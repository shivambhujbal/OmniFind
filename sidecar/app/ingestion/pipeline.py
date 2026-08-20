"""Ingestion orchestration: bytes on disk -> rows in SQLite.

This is phase 2's end state. It stores the original, runs the format-specific
extractor, writes ``File``/``Asset``/``Chunk`` rows, and leaves the file in
``EXTRACTED``. Phase 3 picks it up from there for embeddings, OCR and captions.

Failure policy: an unreadable or unsupported file marks that one file ``FAILED``
with a message a person can act on. It never raises out of the worker, because
one bad file in a dropped folder of fifty must not stop the other forty-nine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Asset,
    AssetKind,
    Chunk,
    ChunkKind,
    File,
    FileKind,
    ProcessingStatus,
)
from app.ingestion import chunking, detect, docx, images, pdf, quality
from app.logging_conf import get_logger
from app.storage import files as storage

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    file_id: int
    status: ProcessingStatus
    chunk_count: int
    asset_count: int
    duplicate: bool
    error: str | None = None


def register_upload(session: Session, temp_path: Path, original_name: str) -> File:
    """Store the bytes and create (or find) the ``File`` row.

    Runs on the request thread so the caller learns immediately that the file
    was accepted, and gets a real error for an unsupported type instead of a
    row that fails silently in the background.
    """
    detected = detect.detect(temp_path, original_name)  # raises UnsupportedFileError
    stored = storage.store_upload(temp_path, original_name)

    existing = session.scalar(select(File).where(File.sha256 == stored.sha256))
    if existing is not None:
        log.info("duplicate upload: %s matches file %d", original_name, existing.id)
        return existing

    file_row = File(
        sha256=stored.sha256,
        original_name=original_name,
        stored_path=stored.relative_path,
        kind=detected.kind,
        mime_type=detected.mime_type,
        size_bytes=stored.size_bytes,
        status=ProcessingStatus.PENDING,
    )
    session.add(file_row)
    session.flush()  # assign the id without ending the caller's transaction
    return file_row


def register_existing_file(session: Session, path: Path) -> tuple[File, bool]:
    """Register a file the user owns, without touching the original.

    Returns ``(row, is_new)`` so a folder scan can report how many files were
    genuinely added versus already known -- re-scanning a folder after adding
    two documents should say "2 added", not "500 added".
    """
    detected = detect.detect(path)  # raises UnsupportedFileError
    stored = storage.store_existing_file(path)

    existing = session.scalar(select(File).where(File.sha256 == stored.sha256))
    if existing is not None:
        return existing, False

    file_row = File(
        sha256=stored.sha256,
        original_name=path.name,
        stored_path=stored.relative_path,
        kind=detected.kind,
        mime_type=detected.mime_type,
        size_bytes=stored.size_bytes,
        status=ProcessingStatus.PENDING,
    )
    session.add(file_row)
    session.flush()
    return file_row, True


def extract_file(session: Session, file_id: int) -> IngestResult:
    """Run extraction for one already-registered file."""
    file_row = session.get(File, file_id)
    if file_row is None:
        raise LookupError(f"No file with id {file_id}")

    if file_row.status in (ProcessingStatus.EXTRACTED, ProcessingStatus.READY):
        return IngestResult(
            file_id, file_row.status, len(file_row.chunks), len(file_row.assets), True
        )

    file_row.status = ProcessingStatus.EXTRACTING
    file_row.error = None
    session.commit()

    source = storage.absolute_file_path(file_row.stored_path)
    try:
        if file_row.kind is FileKind.PDF:
            chunk_count, asset_count = _ingest_pdf(session, file_row, source)
        elif file_row.kind is FileKind.DOCX:
            chunk_count, asset_count = _ingest_docx(session, file_row, source)
        else:
            chunk_count, asset_count = _ingest_image(session, file_row, source)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
        session.rollback()
        file_row = session.get(File, file_id)
        assert file_row is not None
        file_row.status = ProcessingStatus.FAILED
        file_row.error = str(exc)[:2000]
        session.commit()
        log.exception("extraction failed for %s", file_row.original_name)
        return IngestResult(file_id, ProcessingStatus.FAILED, 0, 0, False, str(exc))

    file_row.status = ProcessingStatus.EXTRACTED
    session.commit()
    log.info(
        "extracted %s: %d chunks, %d assets",
        file_row.original_name,
        chunk_count,
        asset_count,
    )
    return IngestResult(file_id, ProcessingStatus.EXTRACTED, chunk_count, asset_count, False)


def _add_chunks(session: Session, file_row: File, chunks: list[chunking.TextChunk]) -> int:
    for chunk in chunks:
        session.add(
            Chunk(
                file_id=file_row.id,
                kind=ChunkKind.TEXT,
                ordinal=chunk.ordinal,
                page_number=chunk.page_number,
                text=chunk.text,
                char_count=len(chunk.text),
                # Kept in the database either way -- it is part of the document
                # -- but only indexed if it carries retrievable content.
                searchable=quality.is_searchable(chunk.text),
            )
        )
    return len(chunks)


def _add_image_assets(session: Session, file_row: File, extracted: list[pdf.ExtractedImage]) -> int:
    count = 0
    for image in extracted:
        kind = AssetKind.PAGE_IMAGE if image.is_page_render else AssetKind.EMBEDDED_IMAGE
        session.add(
            Asset(
                file_id=file_row.id,
                kind=kind,
                path=storage.relative_derived(image.path),
                page_number=image.page_number,
                ordinal=image.ordinal,
                width=image.width,
                height=image.height,
            )
        )
        count += 1
    return count


def _ingest_pdf(session: Session, file_row: File, source: Path) -> tuple[int, int]:
    extraction = pdf.extract(source, file_row.sha256)
    file_row.page_count = extraction.page_count
    file_row.title = extraction.title

    chunk_count = _add_chunks(session, file_row, chunking.chunk_pages(extraction.pages))
    asset_count = _add_image_assets(session, file_row, extraction.images)

    if chunk_count == 0 and extraction.scanned_pages:
        log.info(
            "%s has no extractable text on %d page(s); OCR will supply it",
            file_row.original_name,
            len(extraction.scanned_pages),
        )
    return chunk_count, asset_count


def _ingest_docx(session: Session, file_row: File, source: Path) -> tuple[int, int]:
    extraction = docx.extract(source, file_row.sha256)
    file_row.title = extraction.title

    chunk_count = _add_chunks(session, file_row, chunking.chunk_text(extraction.text))
    asset_count = _add_image_assets(session, file_row, extraction.images)
    return chunk_count, asset_count


def _ingest_image(session: Session, file_row: File, source: Path) -> tuple[int, int]:
    info = images.extract(source, file_row.sha256)

    # The uploaded image is itself the asset OCR and captioning will run on.
    session.add(
        Asset(
            file_id=file_row.id,
            kind=AssetKind.SOURCE_IMAGE,
            path=file_row.stored_path,
            ordinal=0,
            width=info.width,
            height=info.height,
        )
    )
    session.add(
        Asset(
            file_id=file_row.id,
            kind=AssetKind.THUMBNAIL,
            path=storage.relative_derived(info.thumbnail_path),
            ordinal=0,
            width=None,
            height=None,
        )
    )
    return 0, 2


def delete_file(session: Session, file_id: int) -> bool:
    """Remove a file's rows and its bytes. Cascades handle chunks and assets."""
    file_row = session.get(File, file_id)
    if file_row is None:
        return False
    sha256, relative = file_row.sha256, file_row.stored_path
    session.delete(file_row)
    session.commit()
    storage.delete_file_data(sha256, relative)

    # Drop the vectors too, or deleted files keep turning up in search results.
    # Imported here so ingestion does not pull in the vector stack.
    from app.vectors import store as vector_store

    vector_store.delete_file_points(file_id)
    return True
