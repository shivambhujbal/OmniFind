"""Phase 2 gate: dropping a file yields SQLite rows and content on disk,
and unsupported input fails without breaking anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db.models import AssetKind, ChunkKind, File, FileKind, ProcessingStatus
from app.ingestion import chunking, detect, pipeline
from app.storage import files as storage
from tests import fixtures


@pytest.fixture
def session(db_session: Session) -> Session:
    return db_session


def ingest(session: Session, path: Path) -> File:
    """Register + extract, the way the upload endpoint and worker do."""
    file_row = pipeline.register_upload(session, path, path.name)
    session.commit()
    pipeline.extract_file(session, file_row.id)
    session.refresh(file_row)
    return file_row


# --- detection ---------------------------------------------------------


def test_detects_by_content_not_extension(tmp_path: Path) -> None:
    """A PNG named .pdf is a PNG."""
    misnamed = tmp_path / "actually_an_image.pdf"
    fixtures.make_image(tmp_path / "real.png")
    misnamed.write_bytes((tmp_path / "real.png").read_bytes())

    assert detect.detect(misnamed).kind is FileKind.IMAGE


def test_detects_docx_inside_zip_container(tmp_path: Path) -> None:
    path = fixtures.make_docx(tmp_path / "notes.docx")
    assert detect.detect(path).kind is FileKind.DOCX


def test_unsupported_type_raises_a_readable_message(tmp_path: Path) -> None:
    path = fixtures.make_unsupported(tmp_path / "thing.xyz")
    with pytest.raises(detect.UnsupportedFileError) as exc:
        detect.detect(path)
    # The message reaches the user, so it must name the formats we do handle.
    assert ".pdf" in str(exc.value)


# --- chunking ----------------------------------------------------------


def test_chunks_respect_size_and_overlap() -> None:
    text = "\n\n".join(fixtures.LOREM for _ in range(12))
    chunks = chunking.chunk_text(text, size=600, overlap=100)

    assert len(chunks) > 1
    assert all(len(c.text) <= 900 for c in chunks), "chunks far over target"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunking_drops_trivial_fragments() -> None:
    assert chunking.chunk_text("7") == []
    assert chunking.chunk_text("") == []


def test_normalise_rejoins_hyphenated_line_breaks() -> None:
    assert "migration" in chunking.normalise("mig-\nration across the sea")


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        chunking.chunk_text(fixtures.LOREM, size=100, overlap=100)


# --- PDF ---------------------------------------------------------------


def test_text_pdf_produces_chunks_and_rows(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_text_pdf(tmp_path / "terns.pdf", pages=3)
    file_row = ingest(session, path)

    assert file_row.status is ProcessingStatus.EXTRACTED
    assert file_row.kind is FileKind.PDF
    assert file_row.page_count == 3
    assert len(file_row.chunks) > 0
    assert all(c.kind is ChunkKind.TEXT for c in file_row.chunks)
    assert all(c.page_number in (1, 2, 3) for c in file_row.chunks)

    # The original bytes are on disk under the content hash.
    stored = storage.absolute_file_path(file_row.stored_path)
    assert stored.exists()
    assert file_row.sha256 in stored.name


def test_scanned_pdf_yields_page_images_for_ocr(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_scanned_pdf(tmp_path / "invoice.pdf")
    file_row = ingest(session, path)

    assert file_row.status is ProcessingStatus.EXTRACTED
    # No text layer, so no text chunks -- OCR supplies the content in phase 3.
    assert len(file_row.chunks) == 0

    renders = [a for a in file_row.assets if a.kind is AssetKind.PAGE_IMAGE]
    assert renders, "a text-less page must be rendered for OCR"
    assert storage.absolute_derived_path(renders[0].path).exists()


def test_pdf_embedded_images_are_extracted(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_pdf_with_image(tmp_path / "survey.pdf")
    file_row = ingest(session, path)

    embedded = [a for a in file_row.assets if a.kind is AssetKind.EMBEDDED_IMAGE]
    assert embedded, "the embedded picture should have been extracted"
    asset = embedded[0]
    assert asset.width and asset.width >= 128
    assert storage.absolute_derived_path(asset.path).exists()


# --- DOCX --------------------------------------------------------------


def test_docx_text_and_table_are_extracted(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_docx(tmp_path / "notes.docx")
    file_row = ingest(session, path)

    assert file_row.status is ProcessingStatus.EXTRACTED
    assert file_row.title == "Coastal Survey Notes"
    assert len(file_row.chunks) > 0

    body = " ".join(c.text for c in file_row.chunks)
    assert "Observation 1" in body
    assert "Sandy Point" in body, "table contents must be part of the searchable text"
    # DOCX has no stored pagination.
    assert all(c.page_number is None for c in file_row.chunks)


# --- images ------------------------------------------------------------


def test_image_yields_source_asset_and_thumbnail(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_image(tmp_path / "harbour.png")
    file_row = ingest(session, path)

    assert file_row.status is ProcessingStatus.EXTRACTED
    kinds = {a.kind for a in file_row.assets}
    assert AssetKind.SOURCE_IMAGE in kinds
    assert AssetKind.THUMBNAIL in kinds
    # No text yet -- OCR and captioning fill this in during phase 3.
    assert len(file_row.chunks) == 0

    thumb = next(a for a in file_row.assets if a.kind is AssetKind.THUMBNAIL)
    assert storage.absolute_derived_path(thumb.path).exists()


# --- failure and dedup -------------------------------------------------


def test_corrupt_pdf_fails_one_file_not_the_run(session: Session, tmp_path: Path) -> None:
    bad = fixtures.make_corrupt_pdf(tmp_path / "broken.pdf")
    good = fixtures.make_text_pdf(tmp_path / "fine.pdf", pages=1)

    bad_row = ingest(session, bad)
    good_row = ingest(session, good)

    assert bad_row.status is ProcessingStatus.FAILED
    assert bad_row.error, "a failure must record why, for the UI to show"
    assert good_row.status is ProcessingStatus.EXTRACTED, "one bad file must not poison the rest"


def test_identical_bytes_are_not_ingested_twice(session: Session, tmp_path: Path) -> None:
    first = fixtures.make_text_pdf(tmp_path / "a.pdf", pages=1)
    second = tmp_path / "renamed.pdf"
    second.write_bytes(first.read_bytes())

    row_one = ingest(session, first)
    row_two = pipeline.register_upload(session, second, second.name)
    session.commit()

    assert row_two.id == row_one.id, "content hash, not filename, decides identity"


def test_delete_removes_rows_and_bytes(session: Session, tmp_path: Path) -> None:
    path = fixtures.make_pdf_with_image(tmp_path / "gone.pdf")
    file_row = ingest(session, path)
    file_id, sha256, stored = file_row.id, file_row.sha256, file_row.stored_path

    assert pipeline.delete_file(session, file_id) is True

    assert session.get(File, file_id) is None
    assert not storage.absolute_file_path(stored).exists()
    assert not (storage.settings.derived_dir / sha256[:2] / sha256).exists()


def test_scanned_page_is_not_processed_twice(session: Session, tmp_path: Path) -> None:
    """A scanned page yields ONE image to process, not two.

    The rendered page already composites every picture on it, so also extracting
    the embedded image means OCRing and captioning identical content twice --
    two near-duplicate chunks in the index, and double the captioning cost,
    which is ~390s per image on CPU.
    """
    path = fixtures.make_scanned_pdf(tmp_path / "invoice.pdf")
    file_row = ingest(session, path)

    processable = [
        a for a in file_row.assets if a.kind in (AssetKind.PAGE_IMAGE, AssetKind.EMBEDDED_IMAGE)
    ]
    assert (
        len(processable) == 1
    ), f"expected one image per scanned page, got {[a.kind.value for a in processable]}"
    assert processable[0].kind is AssetKind.PAGE_IMAGE


def test_text_page_still_yields_its_embedded_pictures(session: Session, tmp_path: Path) -> None:
    """The scanned-page shortcut must not suppress images on normal pages."""
    path = fixtures.make_pdf_with_image(tmp_path / "survey.pdf")
    file_row = ingest(session, path)

    embedded = [a for a in file_row.assets if a.kind is AssetKind.EMBEDDED_IMAGE]
    assert embedded, "a text page's pictures must still be extracted"
