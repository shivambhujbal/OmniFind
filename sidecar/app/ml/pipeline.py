"""Phase 3 processing: turn an EXTRACTED file into a fully described one.

Runs after ingestion, on the background worker thread. For one file it:

1. OCRs every image asset, storing usable text as ``OCR`` chunks
2. Captions every image asset, storing the caption on the asset *and* as a
   ``CAPTION`` chunk
3. Leaves the file in ``READY``

Embedding and Qdrant upsert happen in phase 4; this stage produces the text that
phase 4 indexes. The two are kept apart so a captioning failure does not lose
the OCR work, and so re-running one does not redo the other.

Everything here is idempotent at the row level: an asset that already has a
caption is skipped, so a re-run after a crash resumes rather than duplicating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Asset, AssetKind, Chunk, ChunkKind, File, ProcessingStatus
from app.ingestion import chunking, quality
from app.logging_conf import get_logger
from app.ml import captioning, loaders, ocr
from app.storage import files as storage

log = get_logger(__name__)

# Thumbnails are for display only -- OCRing or captioning a 400px copy of an
# image we already processed at full size wastes minutes and adds nothing.
# What the vision models *read*: OCR runs over every image a document
# contains, because that is what makes a scan or a screenshotted table
# searchable at all.
PROCESSABLE_KINDS = (
    AssetKind.PAGE_IMAGE,
    AssetKind.EMBEDDED_IMAGE,
    AssetKind.SOURCE_IMAGE,
)

# What gets a CLIP vector: standalone image files only.
#
# Deliberately NOT the images inside documents. A figure in a PDF is already
# reachable through the document's own text and through OCR of the figure
# itself, so a CLIP vector adds a way to find it that duplicates one that
# already works -- and it is not free. One 52-page slide deck in the dev
# library extracts to 2054 embedded images: 75 minutes of CPU, 95% of the
# whole backlog, for one file that OCR had already made searchable.
#
# Page renderings are excluded for the same reason: a page image is a picture
# of text, and text is what bge and FTS5 already index well.
CLIP_INDEXABLE_KINDS = (AssetKind.SOURCE_IMAGE,)


@dataclass(frozen=True)
class ProcessResult:
    file_id: int
    status: ProcessingStatus
    ocr_chunks: int
    captions: int
    skipped: int
    error: str | None = None
    # Set when a stage was skipped because its model is not installed. The file
    # is not broken -- it is waiting for a capability. See `process_file`.
    deferred_reason: str | None = None


def asset_path(asset: Asset) -> Path:
    """Absolute path for an asset, which lives in one of two roots."""
    if asset.kind is AssetKind.SOURCE_IMAGE:
        return storage.absolute_file_path(asset.path)
    return storage.absolute_derived_path(asset.path)


def _next_ordinal(session: Session, file_id: int, kind: ChunkKind) -> int:
    """Next free ordinal for a chunk kind.

    Chunk ordinals are unique per (file, kind), so a re-run has to continue the
    sequence rather than restart it and collide.
    """
    existing = session.scalars(
        select(Chunk.ordinal).where(Chunk.file_id == file_id, Chunk.kind == kind)
    ).all()
    return max(existing) + 1 if existing else 0


def process_file(session: Session, file_id: int) -> ProcessResult:
    """Run OCR and captioning over one file's images."""
    file_row = session.get(File, file_id)
    if file_row is None:
        raise LookupError(f"No file with id {file_id}")

    if file_row.status is ProcessingStatus.READY:
        return ProcessResult(file_id, file_row.status, 0, 0, 0)

    assets = [a for a in file_row.assets if a.kind in PROCESSABLE_KINDS]
    if not assets:
        # Text-only document: nothing for the vision models to do.
        file_row.status = ProcessingStatus.READY
        session.commit()
        return ProcessResult(file_id, ProcessingStatus.READY, 0, 0, 0)

    file_row.status = ProcessingStatus.PROCESSING
    file_row.error = None
    session.commit()

    ocr_chunks = 0
    captions = 0
    try:
        # OCR always runs: it is cheap and it is what makes a scanned document
        # findable at all. Captioning is the part that costs minutes per image,
        # so it is gated.
        ocr_chunks = _run_ocr(session, file_row, assets)
        if settings.enable_captioning:
            # Only standalone image files, the same set that gets a CLIP
            # vector. This is what makes captioning affordable at all on CPU:
            # at ~390s each, describing the 2054 figures inside one slide deck
            # is nine days of compute for pictures OCR had already read, while
            # describing the handful of photographs a person actually adds is
            # minutes. The expensive model is pointed at the images that have
            # no other way in.
            captionable = [a for a in assets if a.kind in CLIP_INDEXABLE_KINDS]
            captions = _run_captioning(session, file_row, captionable)
            # Hand the memory back now, while nothing else wants it. Waiting
            # for the next eviction means freeing 4.1GB and allocating 2.9GB in
            # the same instant, which is what killed the sidecar on the first
            # search after a captioning run.
            loaders.release_heavy_model(settings.captioner)
        else:
            log.debug(
                "captioning disabled; %s indexed on its text alone",
                file_row.original_name,
            )
    except loaders.ModelNotAvailableError as exc:
        # NOT a failure. The document's extracted text is intact and fully
        # searchable; only the image-derived text is missing, because a model
        # is not installed. Marking the file FAILED here would hide working
        # content behind a scary red state, and would be wrong the moment the
        # user installs the model.
        #
        # Leaving it in EXTRACTED means `requeue_unprocessed` picks it up
        # automatically on the next start, so provisioning the model is all the
        # user has to do -- no re-import, no manual retry.
        session.rollback()
        file_row = session.get(File, file_id)
        assert file_row is not None
        file_row.status = ProcessingStatus.EXTRACTED
        file_row.error = None
        session.commit()
        log.warning("deferred image processing for %s: %s", file_row.original_name, exc)
        return ProcessResult(
            file_id, ProcessingStatus.EXTRACTED, 0, 0, len(assets), deferred_reason=str(exc)
        )
    except Exception as exc:  # noqa: BLE001 - one file must not stop the queue
        session.rollback()
        file_row = session.get(File, file_id)
        assert file_row is not None
        file_row.status = ProcessingStatus.FAILED
        file_row.error = str(exc)[:2000]
        session.commit()
        log.exception("processing failed for %s", file_row.original_name)
        return ProcessResult(file_id, ProcessingStatus.FAILED, 0, 0, 0, str(exc))

    file_row.status = ProcessingStatus.READY
    session.commit()
    log.info(
        "processed %s: %d OCR chunks, %d captions",
        file_row.original_name,
        ocr_chunks,
        captions,
    )
    return ProcessResult(file_id, ProcessingStatus.READY, ocr_chunks, captions, 0)


def _already_has_chunk(session: Session, asset_id: int, kind: ChunkKind) -> bool:
    return (
        session.scalar(
            select(Chunk.id).where(Chunk.asset_id == asset_id, Chunk.kind == kind).limit(1)
        )
        is not None
    )


def _run_ocr(session: Session, file_row: File, assets: list[Asset]) -> int:
    pending = [a for a in assets if not _already_has_chunk(session, a.id, ChunkKind.OCR)]
    if not pending:
        return 0

    by_path = {asset_path(a): a for a in pending}
    results = ocr.read_images(list(by_path))

    ordinal = _next_ordinal(session, file_row.id, ChunkKind.OCR)
    added = 0
    for path, result in results.items():
        asset = by_path[path]
        # OCR output can be long (a dense scanned page), so it is chunked the
        # same way document text is rather than stored as one giant row.
        for piece in chunking.chunk_text(result.text, page_number=asset.page_number):
            session.add(
                Chunk(
                    file_id=file_row.id,
                    asset_id=asset.id,
                    kind=ChunkKind.OCR,
                    ordinal=ordinal,
                    page_number=asset.page_number,
                    text=piece.text,
                    char_count=len(piece.text),
                    searchable=quality.is_searchable(piece.text),
                )
            )
            ordinal += 1
            added += 1

        # Short OCR output falls below the chunker's minimum but is often the
        # most useful thing on the page (an invoice total, a sign). Keep it.
        if not added and result.text.strip():
            session.add(
                Chunk(
                    file_id=file_row.id,
                    asset_id=asset.id,
                    kind=ChunkKind.OCR,
                    ordinal=ordinal,
                    page_number=asset.page_number,
                    text=result.text.strip(),
                    char_count=len(result.text.strip()),
                    searchable=quality.is_searchable(result.text),
                )
            )
            ordinal += 1
            added += 1

    session.commit()
    return added


def _run_captioning(session: Session, file_row: File, assets: list[Asset]) -> int:
    pending = [a for a in assets if a.caption is None]
    if not pending:
        return 0

    by_path = {asset_path(a): a for a in pending}
    results = captioning.caption_images(list(by_path))

    ordinal = _next_ordinal(session, file_row.id, ChunkKind.CAPTION)
    added = 0
    for path, caption in results.items():
        asset = by_path[path]
        asset.caption = caption.text

        if _already_has_chunk(session, asset.id, ChunkKind.CAPTION):
            continue
        session.add(
            Chunk(
                file_id=file_row.id,
                asset_id=asset.id,
                kind=ChunkKind.CAPTION,
                ordinal=ordinal,
                page_number=asset.page_number,
                text=caption.text,
                char_count=len(caption.text),
                searchable=quality.is_searchable(caption.text),
            )
        )
        ordinal += 1
        added += 1

    session.commit()
    return added
