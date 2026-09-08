"""Query -> ranked results.

The query is embedded twice, once per space: with bge for the text collection
and with CLIP's text encoder for the image collection. They are different
models producing different vectors, and neither can search the other's
collection.

Under ``ml_low_memory`` that means loading two models per query, so the text
search runs first and completely before CLIP is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from app.config import settings
from app.db.session import session_scope
from app.logging_conf import get_logger
from app.ml import embeddings_clip, embeddings_text, loaders
from app.search import lexical
from app.search.ranking import SearchResult, fuse
from app.vectors import store
from app.vectors.store import VectorHit

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = get_logger(__name__)


class KindFilter(str, Enum):
    """Which files to search.

    Note this filters by the *file* a match came from, not by how the match was
    found: OCR text extracted from a photo is still a match in an image file,
    and a scanned page inside a PDF is still a document.
    """

    ALL = "all"
    DOCUMENTS = "documents"
    IMAGES = "images"

    @property
    def file_kinds(self) -> list[str] | None:
        if self is KindFilter.DOCUMENTS:
            return ["pdf", "docx"]
        if self is KindFilter.IMAGES:
            return ["image"]
        return None


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    text_candidates: int
    image_candidates: int
    searched_images: bool
    # False when even the best hit is only marginally better than the library's
    # baseline similarity. The results are still the closest available -- they
    # just should not be presented as answers. See config.confident_match_score.
    confident: bool = False
    # Total matches before paging, so the UI can offer page navigation.
    total: int = 0
    page: int = 1
    page_size: int = field(default=0)


def search(
    query: str,
    limit: int | None = None,
    include_images: bool = True,
    group_by_file: bool = True,
    kinds: KindFilter = KindFilter.ALL,
    page: int = 1,
    page_size: int | None = None,
) -> SearchResponse:
    """Search text and (optionally) images, and return one page of results."""
    query = query.strip()
    page_size = page_size or settings.search_page_size
    if not query:
        return SearchResponse(query, [], 0, 0, False, page=page, page_size=page_size)

    limit = limit or settings.search_limit
    file_kinds = kinds.file_kinds

    # No slot for the text encoder: bge is kept resident precisely so a query
    # never waits behind a captioning pass. Not guarded against a missing model
    # either -- without it there is no search worth returning, so that must
    # surface rather than degrade into image-only results that would look like a
    # complete answer.
    query_vector = embeddings_text.embed_query(query)
    text_hits = store.search_text(query_vector, file_kinds=file_kinds)

    # Keyword search, fused as a third list. Dense vectors cannot find an exact
    # identifier -- a reference number scores BELOW unrelated documents, because
    # the model has no representation for a long digit string. See
    # search/lexical.py.
    keyword_hits = _keyword_hits(query, query_vector, file_kinds)

    image_hits = []
    searched_images = False
    # Searching the image collection for a documents-only query would be pure
    # waste: every hit would be filtered out again by kind.
    wants_images = include_images and kinds is not KindFilter.DOCUMENTS
    if wants_images and settings.enable_image_search:
        try:
            # A much shorter wait: image results are a bonus on top of text
            # results, so if the worker is busy it is better to return the text
            # matches now than to make the user wait minutes for both.
            with loaders.heavy_model_slot("clip", timeout=settings.image_model_wait_seconds):
                image_vector = embeddings_clip.embed_query(query)
            image_hits = store.search_images(image_vector, file_kinds=file_kinds)
            searched_images = True
        except loaders.ModelBusyError:
            log.info("skipping image search: models busy with background processing")
        except (loaders.ModelNotAvailableError, Exception) as exc:
            # Text results are still worth returning: degrade, do not fail.
            log.warning("image search unavailable: %s", exc)

    ranked = fuse(text_hits + keyword_hits, image_hits, limit=limit, group_by_file=group_by_file)

    total = len(ranked)
    page = max(1, page)
    start = (page - 1) * page_size
    # A page past the end returns nothing rather than clamping: silently showing
    # page 3 when someone asked for page 9 hides that the results changed.
    results = ranked[start : start + page_size]

    log.info(
        "query %r [%s] -> %d result(s) of %d (page %d; %d dense, %d keyword)",
        query,
        kinds.value,
        len(results),
        total,
        page,
        len(text_hits),
        len(keyword_hits),
    )
    return SearchResponse(
        query=query,
        results=results,
        text_candidates=len(text_hits),
        image_candidates=len(image_hits),
        searched_images=searched_images,
        # An exact identifier match is a confident answer whatever its cosine:
        # the number was found, and the model's opinion of the number is not
        # evidence about whether it is the right document.
        confident=bool(ranked)
        and (ranked[0].exact or ranked[0].similarity >= settings.confident_match_score),
        total=total,
        page=page,
        page_size=page_size,
    )


def _keyword_hits(
    query: str, query_vector: NDArray[np.float32], file_kinds: list[str] | None
) -> list[VectorHit]:
    """Literal matches, shaped like vector hits so they can be fused.

    Their ``score`` is the real cosine looked up from the stored vector, not an
    invented number -- and 0.0 when the chunk has no vector at all, which is
    the honest value for a passage excluded from the semantic index.
    """
    with session_scope() as session:
        hits = lexical.search(session, query, limit=settings.search_candidates)

    if not hits:
        return []

    if file_kinds is not None:
        allowed = set(file_kinds)
        hits = [h for h in hits if h.payload.get("file_kind") in allowed]
        if not hits:
            return []

    similarities = store.similarities_for(
        settings.text_collection, [h.chunk_id for h in hits], query_vector
    )

    vector_hits = [
        VectorHit(
            point_id=hit.chunk_id,
            score=similarities.get(hit.chunk_id, 0.0),
            collection=settings.text_collection,
            payload=hit.payload,
            exact=hit.is_exact,
        )
        for hit in hits
    ]

    # An *identifier* hit stands on its own: the number was found, and the
    # model's opinion of the number is not evidence. An ordinary-word hit is a
    # different thing entirely -- it is an OR over the query's words, so
    # "a red Mercedes car" matches every document containing "car", with no
    # score floor of its own because it never went through the vector store.
    # That flooded results with matches at 0.33 that the dense retriever had
    # already rejected at 0.55, and they then set the relative floor and buried
    # the genuine answers. Keyword hits on plain words have to clear the same
    # bar as anything else.
    if vector_hits and not vector_hits[0].exact:
        vector_hits = [h for h in vector_hits if h.score >= settings.min_text_score]

    return vector_hits
