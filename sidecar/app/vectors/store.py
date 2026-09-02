"""Embedded Qdrant: collections, upsert, search, delete.

``QdrantClient(path=...)`` runs the engine in-process against local files. No
server, no Docker, no port. The path holds an exclusive lock, which is fine for
one desktop process but means tests must use their own data directory.

**Two collections, not one.** bge and CLIP produce vectors of the same width
(1024) in completely different spaces -- a bge vector and a CLIP vector are not
comparable, and a nearest-neighbour search across a mixture of them returns
noise. So text lives in one collection, images in the other, each is searched
with its own encoder, and the two result lists are fused by rank (see
``search/ranking.py``) rather than by raw score.

**Point ids are the SQLite row ids.** A chunk's point id is its ``chunks.id``,
an image's is its ``assets.id``. That makes deletion a direct addressing
operation instead of a scan, and makes SQLite the single source of truth about
what should exist.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.logging_conf import get_logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from qdrant_client import QdrantClient

log = get_logger(__name__)

_client: QdrantClient | None = None
_client_lock = threading.Lock()


@dataclass(frozen=True)
class VectorHit:
    """One raw hit from one collection, before fusion."""

    point_id: int
    score: float
    collection: str
    payload: dict[str, Any]
    # True for a literal keyword match on an identifier rather than a vector
    # hit. Such a match is evidence independent of cosine, so ranking exempts
    # it from the similarity floor. See search/lexical.py.
    exact: bool = False


def get_client() -> QdrantClient:
    """The single embedded client. Created on first use, not at import."""
    global _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            settings.ensure_dirs()
            log.info("opening embedded Qdrant at %s", settings.qdrant_path)
            _client = QdrantClient(path=str(settings.qdrant_path))
        return _client


def close_client() -> None:
    """Release the storage lock. Called on shutdown and by tests."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None
            log.info("closed embedded Qdrant")


def ensure_collections(text_dim: int, image_dim: int) -> None:
    """Create the collections if absent, and refuse to use mismatched ones.

    A collection built for a 384-dim model and then written with 1024-dim
    vectors is a corruption that surfaces much later as bad results, so the
    dimensions are checked against the loaded models rather than assumed.
    """
    from qdrant_client.models import Distance, VectorParams

    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    for name, dim in ((settings.text_collection, text_dim), (settings.image_collection, image_dim)):
        if name not in existing:
            client.create_collection(
                collection_name=name,
                # Vectors are L2-normalised at write time, so cosine and dot
                # agree; cosine is stated explicitly to survive that changing.
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("created collection %s (dim=%d)", name, dim)
            continue

        actual = client.get_collection(name).config.params.vectors.size  # type: ignore[union-attr]
        if actual != dim:
            raise RuntimeError(
                f"Collection {name!r} stores {actual}-dimensional vectors but the loaded model "
                f"produces {dim}. The collection was built with a different model; delete "
                f"{settings.qdrant_path} to rebuild the index."
            )


def upsert_text(
    ids: list[int], vectors: NDArray[np.float32], payloads: list[dict[str, Any]]
) -> None:
    _upsert(settings.text_collection, ids, vectors, payloads)


def upsert_images(
    ids: list[int], vectors: NDArray[np.float32], payloads: list[dict[str, Any]]
) -> None:
    _upsert(settings.image_collection, ids, vectors, payloads)


def _upsert(
    collection: str, ids: list[int], vectors: NDArray[np.float32], payloads: list[dict[str, Any]]
) -> None:
    if not ids:
        return
    if not (len(ids) == len(vectors) == len(payloads)):
        # A mismatch here silently attaches vectors to the wrong rows, which no
        # later check would catch.
        raise ValueError(
            f"upsert length mismatch: {len(ids)} ids, {len(vectors)} vectors, "
            f"{len(payloads)} payloads"
        )

    from qdrant_client.models import PointStruct

    client = get_client()
    points = [
        PointStruct(id=int(point_id), vector=vector.tolist(), payload=payload)
        for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    log.debug("upserted %d points into %s", len(points), collection)


def search_text(
    vector: NDArray[np.float32],
    limit: int | None = None,
    min_score: float | None = None,
    file_kinds: list[str] | None = None,
) -> list[VectorHit]:
    threshold = settings.min_text_score if min_score is None else min_score
    return _search(settings.text_collection, vector, limit, threshold, file_kinds)


def search_images(
    vector: NDArray[np.float32],
    limit: int | None = None,
    min_score: float | None = None,
    file_kinds: list[str] | None = None,
) -> list[VectorHit]:
    threshold = settings.min_image_score if min_score is None else min_score
    return _search(settings.image_collection, vector, limit, threshold, file_kinds)


def _kind_filter(file_kinds: list[str] | None) -> Any:
    """Restrict to certain file kinds, applied by the engine.

    Filtering after retrieval would silently shrink the result set: the store
    returns its N nearest points, most get discarded, and matches beyond
    position N never appear. Qdrant applies this during the search instead.
    """
    if not file_kinds:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchAny

    return Filter(must=[FieldCondition(key="file_kind", match=MatchAny(any=list(file_kinds)))])


def _search(
    collection: str,
    vector: NDArray[np.float32],
    limit: int | None = None,
    min_score: float | None = None,
    file_kinds: list[str] | None = None,
) -> list[VectorHit]:
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = get_client()
    try:
        results = client.query_points(
            collection_name=collection,
            query=vector.tolist(),
            limit=limit or settings.search_candidates,
            # Applied by the engine, so weak neighbours never leave the store.
            # Without it the store returns its N nearest points regardless of
            # how far away they are, and "nearest" is not the same as "relevant".
            score_threshold=min_score,
            query_filter=_kind_filter(file_kinds),
            with_payload=True,
        ).points
    except (UnexpectedResponse, ValueError) as exc:
        # An absent collection means nothing of that kind has been indexed yet
        # -- a text-only library has no image collection. That is an empty
        # result, not an error the user should see.
        log.debug("search on %s failed (probably empty): %s", collection, exc)
        return []

    return [
        VectorHit(
            point_id=int(point.id),
            score=float(point.score),
            collection=collection,
            payload=dict(point.payload or {}),
        )
        for point in results
    ]


def similarities_for(
    collection: str, ids: list[int], query_vector: NDArray[np.float32]
) -> dict[int, float]:
    """Cosine of specific stored points against a query vector.

    Used for hits found by keyword rather than by vector search: they have no
    similarity of their own, and showing them without one -- or with a made-up
    number -- would be worse than looking it up.
    """
    if not ids:
        return {}

    import numpy as np

    try:
        points = get_client().retrieve(
            collection_name=collection, ids=[int(i) for i in ids], with_vectors=True
        )
    except Exception as exc:  # noqa: BLE001 - absent points are not an error here
        log.debug("could not retrieve vectors for %s: %s", ids, exc)
        return {}

    # Stored vectors are L2-normalised at write time, so a dot product is the
    # cosine.
    return {
        int(point.id): float(np.dot(np.asarray(point.vector, dtype=np.float32), query_vector))
        for point in points
        if point.vector is not None
    }


def delete_points(collection: str, ids: list[int]) -> None:
    if not ids:
        return
    get_client().delete(
        collection_name=collection, points_selector=[int(i) for i in ids], wait=True
    )


def delete_file_points(file_id: int) -> None:
    """Remove everything belonging to one file, from both collections.

    Filtered by payload rather than by id: the SQLite rows are already gone by
    the time this runs (cascade delete), so their ids are no longer available.
    """
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    selector = FilterSelector(
        filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))])
    )
    client = get_client()
    for collection in (settings.text_collection, settings.image_collection):
        try:
            client.delete(collection_name=collection, points_selector=selector, wait=True)
        except Exception as exc:  # noqa: BLE001 - a missing collection is not an error here
            log.debug("could not delete points for file %d from %s: %s", file_id, collection, exc)


def counts() -> dict[str, int]:
    """Points per collection, for the status view."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    result = {}
    for name in (settings.text_collection, settings.image_collection):
        result[name] = client.count(name).count if name in existing else 0
    return result
