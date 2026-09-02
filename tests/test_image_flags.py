"""The two image switches, and what each one costs when it is off.

Image search and captioning used to share one flag. Measured on the CPU dev
box, a CLIP image vector costs **2.19s** per image and a moondream2 caption
costs **~390s** -- 178x more -- so a single switch made a CPU-only machine give
up picture search it could easily have afforded.

These tests pin the split: captioning off must still index image vectors, and
either flag off must still leave documents (including scans) fully searchable
and pick the skipped work up later with no re-import.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import AssetKind, ChunkKind, ProcessingStatus
from app.ingestion import pipeline
from app.ml import pipeline as ml_pipeline
from tests import fixtures


@pytest.fixture
def captions_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU configuration: match pictures, do not describe them."""
    monkeypatch.setattr(settings, "enable_captioning", False)
    monkeypatch.setattr(settings, "enable_image_search", True)


@pytest.fixture
def images_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both halves off -- documents only."""
    monkeypatch.setattr(settings, "enable_captioning", False)
    monkeypatch.setattr(settings, "enable_image_search", False)


def ingest(session: Session, path: Path) -> object:
    file_row = pipeline.register_upload(session, path, path.name)
    session.commit()
    pipeline.extract_file(session, file_row.id)
    return file_row


def test_ocr_still_runs_with_images_off(
    db_session: Session, tmp_path: Path, images_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the flag: a scan must stay searchable.

    OCR is ~1s with Tesseract and is the difference between a scanned document
    being findable and being invisible. Only captioning is expensive.
    """
    from app.ml import ocr

    captioned: list[object] = []

    def fake_ocr(paths):  # noqa: ANN001, ANN202
        return {p: ocr.OcrResult(text="INVOICE TOTAL 4820", engine="stub") for p in paths}

    def record_caption(paths):  # noqa: ANN001, ANN202
        captioned.extend(paths)
        return {}

    monkeypatch.setattr(ml_pipeline.ocr, "read_images", fake_ocr)
    monkeypatch.setattr(ml_pipeline.captioning, "caption_images", record_caption)

    file_row = ingest(db_session, fixtures.make_scanned_pdf(tmp_path / "invoice.pdf"))
    result = ml_pipeline.process_file(db_session, file_row.id)

    assert result.ocr_chunks > 0, "OCR is governed by neither flag and must still run"
    assert not captioned, "captioning should have been skipped entirely"

    db_session.refresh(file_row)
    assert file_row.status is ProcessingStatus.READY
    kinds = {c.kind for c in file_row.chunks}
    assert ChunkKind.OCR in kinds
    assert ChunkKind.CAPTION not in kinds


def test_captions_are_skipped_not_failed(
    db_session: Session, tmp_path: Path, captions_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping is a normal outcome, not an error state on the file."""
    from app.ml import ocr

    monkeypatch.setattr(
        ml_pipeline.ocr, "read_images", lambda paths: dict.fromkeys(paths, ocr.OcrResult("", "s"))
    )

    file_row = ingest(db_session, fixtures.make_pdf_with_image(tmp_path / "survey.pdf"))
    ml_pipeline.process_file(db_session, file_row.id)
    db_session.refresh(file_row)

    assert file_row.status is ProcessingStatus.READY
    assert file_row.error is None
    assert all(a.caption is None for a in file_row.assets)


def test_image_vectors_are_skipped_but_remain_pending(
    db_session: Session, tmp_path: Path, images_off: None
) -> None:
    """Turning the flag back on must not need a re-import.

    `clip_embedded` stays False, so `requeue_unindexed` picks these assets up
    on the next start once image search is enabled.
    """
    from app.ml import indexing

    file_row = ingest(db_session, fixtures.make_image(tmp_path / "harbour.png"))
    result = indexing._index_images(db_session, file_row)

    assert result == 0
    processable = [a for a in file_row.assets if a.kind is AssetKind.SOURCE_IMAGE]
    assert processable, "precondition: there is an image to embed"
    assert all(not a.clip_embedded for a in processable), "must stay queued for later"


def test_search_does_not_touch_clip_when_disabled(
    images_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No CLIP load at query time -- that is 2.9GB and ~16s for nothing."""
    import numpy as np

    from app.search import service

    def must_not_run(query: str):  # noqa: ANN202
        raise AssertionError("CLIP was loaded despite image search being off")

    monkeypatch.setattr(service.embeddings_clip, "embed_query", must_not_run)
    monkeypatch.setattr(
        service.embeddings_text, "embed_query", lambda q: np.zeros(4, dtype=np.float32)
    )
    monkeypatch.setattr(service.store, "search_text", lambda *a, **k: [])

    response = service.search("anything", include_images=True)
    assert response.searched_images is False


def test_status_reports_documents_only_mode(images_off: None) -> None:
    """The UI has to be able to explain why pictures are not being described."""
    from app.api.models import list_models

    report = list_models()
    assert report.image_search is False
    assert report.captioning is False
    # A missing captioner must not be reported as "not ready" when the app is
    # never going to load it.
    assert report.ready_for_processing == report.models[settings.ocr_engine].available


def test_image_vectors_are_written_when_only_captioning_is_off(
    db_session: Session, tmp_path: Path, captions_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the flag was split.

    Under the old single switch this returned 0: turning captioning off to
    escape 390s per image also gave up the 2.19s CLIP vector, and with it any
    ability to search pictures by appearance at all.
    """
    import numpy as np

    from app.ml import indexing

    upserted: list[list[int]] = []

    monkeypatch.setattr(
        indexing.embeddings_clip,
        "embed_images",
        lambda paths: (np.ones((len(paths), 4), dtype=np.float32), list(paths)),
    )
    monkeypatch.setattr(indexing.embeddings_clip, "embedding_dim", lambda: 4)
    monkeypatch.setattr(indexing.store, "ensure_collections", lambda **kw: None)
    monkeypatch.setattr(
        indexing.store, "upsert_images", lambda ids, vectors, payloads: upserted.append(list(ids))
    )

    file_row = ingest(db_session, fixtures.make_image(tmp_path / "harbour.png"))
    written = indexing._index_images(db_session, file_row)

    assert written > 0, "captioning off must not disable image search"
    assert upserted, "vectors must reach the store"
    assert all(a.clip_embedded for a in file_row.assets if a.kind is AssetKind.SOURCE_IMAGE)


def test_captioning_off_still_leaves_no_caption_chunks(
    db_session: Session, tmp_path: Path, captions_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the split: the expensive part really is skipped."""
    from app.ml import ocr

    captioned: list[object] = []

    def record_caption(paths):  # noqa: ANN001, ANN202
        captioned.extend(paths)
        return {}

    monkeypatch.setattr(
        ml_pipeline.ocr, "read_images", lambda paths: dict.fromkeys(paths, ocr.OcrResult("", "s"))
    )
    monkeypatch.setattr(ml_pipeline.captioning, "caption_images", record_caption)

    file_row = ingest(db_session, fixtures.make_pdf_with_image(tmp_path / "survey.pdf"))
    ml_pipeline.process_file(db_session, file_row.id)

    assert not captioned, "390s/image must not run when captioning is off"
    db_session.refresh(file_row)
    assert ChunkKind.CAPTION not in {c.kind for c in file_row.chunks}


def test_retired_env_var_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale FS_ENABLE_IMAGE_UNDERSTANDING must not be ignored in silence.

    `extra="ignore"` would drop it, quietly re-enabling whichever half the user
    meant to keep off -- on this machine, the 390s-per-image one.
    """
    import app.config as config_module

    monkeypatch.setenv("FS_ENABLE_IMAGE_UNDERSTANDING", "false")
    config_module.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="no longer read"):
            config_module.get_settings()
    finally:
        monkeypatch.delenv("FS_ENABLE_IMAGE_UNDERSTANDING", raising=False)
        config_module.get_settings.cache_clear()
