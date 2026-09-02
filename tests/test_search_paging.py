"""Filtering by file kind, and paging through results."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.search import service
from app.search.service import KindFilter
from app.vectors.store import VectorHit


def hit(point_id: int, score: float, file_id: int, file_kind: str = "pdf") -> VectorHit:
    return VectorHit(
        point_id=point_id,
        score=score,
        collection="text_chunks",
        payload={
            "file_id": file_id,
            "chunk_id": point_id,
            "kind": "text",
            "file_name": f"file{file_id}.{file_kind}",
            "file_kind": file_kind,
            "snippet": "…",
        },
    )


@pytest.fixture
def stub_search(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the encoders and the store; the logic under test is above them."""
    captured: dict = {}

    def fake_text_search(
        vector, limit=None, min_score=None, file_kinds=None
    ):  # noqa: ANN001, ANN202
        captured["file_kinds"] = file_kinds
        # 25 files, tightly scored so the relative cutoff keeps them all.
        return [hit(i, 0.80 - i * 0.001, file_id=i) for i in range(1, 26)]

    monkeypatch.setattr(
        service.embeddings_text, "embed_query", lambda q: np.zeros(4, dtype=np.float32)
    )
    monkeypatch.setattr(service.store, "search_text", fake_text_search)
    monkeypatch.setattr(settings, "enable_image_search", False)
    return captured


# --- kind filter -------------------------------------------------------


def test_documents_filter_asks_the_store_for_documents_only(stub_search: dict) -> None:
    """Filtering must happen in the engine, not after retrieval.

    Post-filtering would silently shrink the result set: the store returns its
    N nearest points, most get discarded, and matches beyond position N never
    surface at all.
    """
    service.search("anything", kinds=KindFilter.DOCUMENTS)
    assert stub_search["file_kinds"] == ["pdf", "docx"]


def test_images_filter_asks_for_images_only(stub_search: dict) -> None:
    service.search("anything", kinds=KindFilter.IMAGES)
    assert stub_search["file_kinds"] == ["image"]


def test_everything_applies_no_filter(stub_search: dict) -> None:
    service.search("anything", kinds=KindFilter.ALL)
    assert stub_search["file_kinds"] is None


def test_a_documents_query_does_not_touch_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a 2.9GB model for hits that would be filtered out is pure waste."""
    monkeypatch.setattr(settings, "enable_image_search", True)
    monkeypatch.setattr(
        service.embeddings_text, "embed_query", lambda q: np.zeros(4, dtype=np.float32)
    )
    monkeypatch.setattr(service.store, "search_text", lambda *a, **k: [])

    def must_not_run(query: str):  # noqa: ANN202
        raise AssertionError("CLIP was loaded for a documents-only search")

    monkeypatch.setattr(service.embeddings_clip, "embed_query", must_not_run)

    response = service.search("anything", kinds=KindFilter.DOCUMENTS)
    assert response.searched_images is False


# --- paging ------------------------------------------------------------


def test_a_page_holds_at_most_the_page_size(stub_search: dict) -> None:
    response = service.search("anything", page_size=10)

    assert len(response.results) == 10
    assert response.total == 25
    assert response.page == 1


def test_later_pages_continue_where_the_previous_one_stopped(stub_search: dict) -> None:
    first = service.search("anything", page=1, page_size=10)
    second = service.search("anything", page=2, page_size=10)

    assert [r.file_id for r in first.results] != [r.file_id for r in second.results]
    assert not {r.file_id for r in first.results} & {r.file_id for r in second.results}


def test_the_last_page_holds_the_remainder(stub_search: dict) -> None:
    third = service.search("anything", page=3, page_size=10)
    assert len(third.results) == 5
    assert third.total == 25


def test_a_page_past_the_end_is_empty_rather_than_clamped(stub_search: dict) -> None:
    """Silently showing page 3 when asked for page 9 hides that results changed."""
    response = service.search("anything", page=9, page_size=10)

    assert response.results == []
    assert response.total == 25
    assert response.page == 9


def test_total_survives_paging(stub_search: dict) -> None:
    """The count must describe every match, not just this page.

    Otherwise page navigation cannot know how many pages there are.
    """
    for page in (1, 2, 3):
        assert service.search("anything", page=page, page_size=10).total == 25


def test_page_zero_is_treated_as_the_first_page(stub_search: dict) -> None:
    assert service.search("anything", page=0, page_size=10).page == 1


def test_an_empty_query_reports_no_pages(stub_search: dict) -> None:
    response = service.search("   ")
    assert response.results == []
    assert response.total == 0
