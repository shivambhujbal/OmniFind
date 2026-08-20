"""Embedded Qdrant: collections, upsert, search, delete.

Uses synthetic vectors rather than real models, so this runs on a clean
checkout. Real end-to-end retrieval is covered in `test_search_e2e.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from app.config import settings
from app.vectors import store

DIM = 8


@pytest.fixture
def qdrant(_clean_test_data_dir: None) -> Iterator[None]:
    """A fresh store per test. The embedded client holds an exclusive lock on
    its directory, so it must be closed before the next test opens it."""
    import shutil

    store.close_client()
    shutil.rmtree(settings.qdrant_path, ignore_errors=True)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    yield
    store.close_client()


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for index, value in enumerate(values):
        vector[index] = value
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm else vector


def payload(file_id: int, chunk_id: int, **extra: object) -> dict:
    return {
        "file_id": file_id,
        "chunk_id": chunk_id,
        "kind": "text",
        "file_name": f"file{file_id}.pdf",
        "file_kind": "pdf",
        "snippet": f"chunk {chunk_id}",
        **extra,
    }


def test_collections_are_created_with_the_right_dimensions(qdrant: None) -> None:
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    names = {c.name for c in store.get_client().get_collections().collections}
    assert settings.text_collection in names
    assert settings.image_collection in names


def test_creating_collections_twice_is_harmless(qdrant: None) -> None:
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    assert store.counts()[settings.text_collection] == 0


def test_a_dimension_mismatch_is_refused(qdrant: None) -> None:
    """Writing 1024-dim vectors into a 384-dim collection is silent corruption.

    It surfaces much later as bad results, so it has to fail at setup instead.
    """
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    with pytest.raises(RuntimeError, match="different model"):
        store.ensure_collections(text_dim=DIM * 2, image_dim=DIM)


def test_upsert_then_search_returns_the_nearest(qdrant: None) -> None:
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    store.upsert_text(
        ids=[1, 2, 3],
        vectors=np.stack([unit(1, 0), unit(0, 1), unit(0.7, 0.7)]),
        payloads=[payload(1, 1), payload(1, 2), payload(2, 3)],
    )

    hits = store.search_text(unit(1, 0), limit=3, min_score=0.0)

    assert hits, "search returned nothing"
    assert hits[0].point_id == 1
    assert hits[0].payload["file_id"] == 1
    assert hits[0].score > hits[-1].score


def test_point_ids_are_the_database_ids(qdrant: None) -> None:
    """Deletion addresses points directly instead of scanning for them."""
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    store.upsert_text([42], np.stack([unit(1, 0)]), [payload(7, 42)])

    hits = store.search_text(unit(1, 0), limit=1, min_score=0.0)
    assert hits[0].point_id == 42


def test_upsert_is_idempotent(qdrant: None) -> None:
    """Re-indexing a file must update points, not duplicate them."""
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    for _ in range(3):
        store.upsert_text([1], np.stack([unit(1, 0)]), [payload(1, 1)])

    assert store.counts()[settings.text_collection] == 1


def test_length_mismatch_is_rejected(qdrant: None) -> None:
    """Mismatched inputs would attach vectors to the wrong rows silently."""
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_text([1, 2], np.stack([unit(1, 0)]), [payload(1, 1)])


def test_empty_upsert_is_a_no_op(qdrant: None) -> None:
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    store.upsert_text([], np.zeros((0, DIM), dtype=np.float32), [])
    assert store.counts()[settings.text_collection] == 0


def test_search_on_a_missing_collection_returns_empty_not_an_error(qdrant: None) -> None:
    """A text-only library has never created an image collection.

    That is "no image results", not a failure the user should be shown.
    """
    assert store.search_images(unit(1, 0)) == []


def test_deleting_a_file_removes_its_points_from_both_collections(qdrant: None) -> None:
    """Deleted files must stop appearing in search results."""
    store.ensure_collections(text_dim=DIM, image_dim=DIM)

    store.upsert_text([1, 2], np.stack([unit(1, 0), unit(0, 1)]), [payload(1, 1), payload(2, 2)])
    store.upsert_images([10], np.stack([unit(1, 0)]), [{"file_id": 1, "asset_id": 10}])

    store.delete_file_points(file_id=1)

    # min_score=0: this test is about deletion, not relevance. The synthetic
    # vectors are orthogonal, so the production floor would filter the survivor
    # out and the assertion would pass for entirely the wrong reason.
    remaining = {
        h.payload["file_id"] for h in store.search_text(unit(1, 0), limit=10, min_score=0.0)
    }
    assert remaining == {2}, "file 1's text points should be gone"
    assert store.search_images(unit(1, 0), limit=10, min_score=0.0) == []


def test_counts_report_both_collections(qdrant: None) -> None:
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    store.upsert_text([1], np.stack([unit(1, 0)]), [payload(1, 1)])

    counts = store.counts()
    assert counts[settings.text_collection] == 1
    assert counts[settings.image_collection] == 0


def test_the_relevance_floor_filters_distant_neighbours(qdrant: None) -> None:
    """ "Nearest" is not the same as "relevant".

    Without a threshold the store happily returns its N nearest points however
    far away they are, which is why every document in the library came back for
    every query.
    """
    store.ensure_collections(text_dim=DIM, image_dim=DIM)
    store.upsert_text(
        ids=[1, 2],
        vectors=np.stack([unit(1, 0), unit(0, 1)]),  # identical vs orthogonal
        payloads=[payload(1, 1), payload(2, 2)],
    )

    everything = store.search_text(unit(1, 0), limit=10, min_score=0.0)
    assert len(everything) == 2, "precondition: both points are reachable"

    filtered = store.search_text(unit(1, 0), limit=10, min_score=0.55)
    assert [h.point_id for h in filtered] == [1], "the orthogonal point should be cut"
