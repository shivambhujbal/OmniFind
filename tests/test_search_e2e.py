"""Phase 4 gate: a natural-language query returns the right file.

Real models, real Qdrant, real documents. Skips when the weights are absent.

Captioning is deliberately not exercised here -- it costs ~390s per image on
CPU and is already covered in `test_ml_inference.py`. What matters at this
level is that ingested content becomes searchable and that the ranking puts the
right file first.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.ml import loaders
from app.vectors import store
from tests import fixtures

pytestmark = pytest.mark.slow

requires_text = pytest.mark.skipif(
    not loaders.available_models()["text_embeddings"]["available"],
    reason="bge weights not downloaded",
)
requires_clip = pytest.mark.skipif(
    not loaders.available_models()["clip"]["available"],
    reason="OpenCLIP weights not downloaded",
)


@pytest.fixture
def indexed_library(db_session: Session, tmp_path: Path) -> Iterator[Session]:
    """A small library, ingested and indexed, with a clean vector store."""
    import shutil

    from app.ingestion import pipeline
    from app.ml import indexing

    store.close_client()
    shutil.rmtree(settings.qdrant_path, ignore_errors=True)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)

    documents = [
        fixtures.make_text_pdf(tmp_path / "terns.pdf", pages=2),
        fixtures.make_docx(tmp_path / "survey_notes.docx"),
    ]
    for path in documents:
        file_row = pipeline.register_upload(db_session, path, path.name)
        db_session.commit()
        pipeline.extract_file(db_session, file_row.id)
        indexing.index_file(db_session, file_row.id)

    yield db_session

    store.close_client()
    loaders.load_text_encoder.cache_clear()
    loaders.load_clip.cache_clear()
    gc.collect()


@requires_text
def test_natural_language_query_finds_the_right_document(indexed_library: Session) -> None:
    """The Phase 4 gate, and the whole point of the app.

    The query shares no distinctive words with the document -- "sea birds" and
    "travel" appear nowhere in it. Keyword search would return nothing.
    """
    from app.search import service

    response = service.search("how far do sea birds travel each year?", include_images=False)

    assert response.results, "no results at all"
    assert "terns" in response.results[0].file_name
    assert response.results[0].snippet


@requires_text
def test_a_different_query_finds_the_other_document(indexed_library: Session) -> None:
    """Guards against "the first document always wins" passing test one."""
    from app.search import service

    response = service.search("field observation counts at survey sites", include_images=False)

    assert response.results
    assert "survey" in response.results[0].file_name.lower()


@requires_text
def test_results_carry_enough_to_open_the_source(indexed_library: Session) -> None:
    from app.search import service

    result = service.search("arctic tern migration", include_images=False).results[0]

    assert result.file_id > 0
    assert result.file_name
    assert result.snippet
    # A PDF hit must say which page, or "open at page N" cannot work.
    if result.file_kind == "pdf" and result.match_kind == "text":
        assert result.page_number is not None


@requires_text
def test_one_document_does_not_monopolise_the_results(indexed_library: Session) -> None:
    from app.search import service

    response = service.search("survey", include_images=False, limit=10)
    file_ids = [r.file_id for r in response.results]
    assert len(file_ids) == len(set(file_ids)), "the same file appeared twice"


@requires_text
def test_empty_query_returns_nothing_rather_than_everything(indexed_library: Session) -> None:
    from app.search import service

    assert service.search("   ").results == []


@requires_text
def test_deleted_files_stop_appearing(indexed_library: Session, tmp_path: Path) -> None:
    """A deleted file must leave the index, not linger as a broken result."""
    from app.ingestion import pipeline
    from app.search import service

    before = service.search("arctic tern migration", include_images=False)
    assert before.results
    target = before.results[0].file_id

    pipeline.delete_file(indexed_library, target)

    after = service.search("arctic tern migration", include_images=False)
    assert all(r.file_id != target for r in after.results), "deleted file still in results"


@requires_text
def test_indexing_is_resumable_and_not_duplicated(indexed_library: Session) -> None:
    """Re-running the indexer must be a no-op, not a second copy of everything."""
    from app.db.models import File
    from app.ml import indexing

    counts_before = store.counts()

    for file_row in indexed_library.query(File).all():
        result = indexing.index_file(indexed_library, file_row.id)
        assert result.text_points == 0, "already-indexed chunks were re-embedded"

    assert store.counts() == counts_before


@requires_text
@requires_clip
def test_image_search_finds_a_photo_by_description(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-modal retrieval end to end: text query, image result, no OCR.

    The image contains no readable text of any kind, so a hit can only come
    from the CLIP image vector.

    Image understanding is switched on explicitly: this machine's `.env` turns
    it off (captioning is ~390s per image on CPU), but the capability itself
    still has to be covered so it does not rot before the GPU machine arrives.
    """
    import shutil

    monkeypatch.setattr(settings, "enable_image_understanding", True)

    from PIL import Image, ImageDraw

    from app.ingestion import pipeline
    from app.ml import indexing
    from app.search import service

    store.close_client()
    shutil.rmtree(settings.qdrant_path, ignore_errors=True)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)

    beach = tmp_path / "seaside.png"
    image = Image.new("RGB", (640, 480), "#4aa3df")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 300, 640, 480], fill="#e8d8a0")
    draw.ellipse([480, 40, 580, 140], fill="#ffe066")
    image.save(beach)

    forest = tmp_path / "woodland.png"
    image = Image.new("RGB", (640, 480), "#1e3d1e")
    draw = ImageDraw.Draw(image)
    for x in range(60, 640, 120):
        draw.rectangle([x, 120, x + 30, 480], fill="#5a3a22")
    image.save(forest)

    for path in (beach, forest):
        file_row = pipeline.register_upload(db_session, path, path.name)
        db_session.commit()
        pipeline.extract_file(db_session, file_row.id)
        indexing.index_file(db_session, file_row.id)

    try:
        response = service.search("a sunny beach by the sea")
        assert response.searched_images, "image collection was not searched"
        assert response.results, "no results for an image-only library"
        assert "seaside" in response.results[0].file_name
    finally:
        store.close_client()
        loaders.load_clip.cache_clear()
        loaders.load_text_encoder.cache_clear()
        gc.collect()
