"""File upload, listing and retrieval."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.schemas import (
    FileDetailOut,
    FileOut,
    JobOut,
    LibraryStatsOut,
    ScanPreviewOut,
    ScanRequest,
    ScanResponseOut,
    UploadResponse,
    UploadResultOut,
)
from app.config import settings
from app.db.models import Asset, Chunk, File, FileKind, ProcessingStatus
from app.db.session import get_db
from app.ingestion import pipeline, scan
from app.ingestion.detect import SUPPORTED_EXTENSIONS, UnsupportedFileError
from app.logging_conf import get_logger
from app.storage import files as storage
from app.tasks import ingest
from app.tasks.queue import task_queue

log = get_logger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


def _spool_to_disk(upload: UploadFile) -> tuple[Path, int]:
    """Stream an upload to a temp file, enforcing the size ceiling as we go.

    Streaming rather than `await upload.read()` because a 200 MB scan read into
    memory on a laptop is a real cost, and the size check has to happen before
    the bytes are committed, not after.
    """
    limit = settings.max_upload_mb * 1024 * 1024
    suffix = Path(upload.filename or "").suffix
    # delete=False because the file outlives this scope: it is handed to the
    # storage layer, which moves it into place. Cleaned up on every error path.
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
    size = 0
    try:
        with handle:
            while block := upload.file.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ValueError(f"File is larger than the {settings.max_upload_mb} MB limit.")
                handle.write(block)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return Path(handle.name), size


@router.post("", response_model=UploadResponse)
def upload_files(
    files: list[UploadFile],
    session: Session = Depends(get_db),
) -> UploadResponse:
    """Accept one or more files, store them, and queue extraction.

    Per-file outcomes rather than a single status code: dropping a folder where
    three of twenty files are unsupported should ingest the seventeen and say
    what happened to the rest.
    """
    results: list[UploadResultOut] = []

    for upload in files:
        name = upload.filename or "unnamed"
        try:
            temp_path, _ = _spool_to_disk(upload)
        except ValueError as exc:
            results.append(UploadResultOut(original_name=name, accepted=False, reason=str(exc)))
            continue

        try:
            file_row = pipeline.register_upload(session, temp_path, name)
            session.commit()
        except UnsupportedFileError as exc:
            temp_path.unlink(missing_ok=True)
            results.append(UploadResultOut(original_name=name, accepted=False, reason=str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
            session.rollback()
            temp_path.unlink(missing_ok=True)
            log.exception("failed to register %s", name)
            results.append(UploadResultOut(original_name=name, accepted=False, reason=str(exc)))
            continue

        already_done = file_row.status in (ProcessingStatus.EXTRACTED, ProcessingStatus.READY)
        job_id = None
        if not already_done:
            job_id = ingest.submit_extraction(file_row.id, name).id

        results.append(
            UploadResultOut(
                original_name=name,
                accepted=True,
                duplicate=already_done,
                file_id=file_row.id,
                job_id=job_id,
            )
        )

    accepted = sum(1 for r in results if r.accepted)
    return UploadResponse(results=results, accepted=accepted, rejected=len(results) - accepted)


@router.post("/scan", response_model=ScanResponseOut)
def scan_folder(request: ScanRequest, session: Session = Depends(get_db)) -> ScanResponseOut:
    """Index every supported file in a local folder.

    The originals are copied into the app's store and never moved or modified;
    this reads the user's folder, it does not take it over.
    """
    try:
        result = scan.scan_folder(session, request.path, recursive=request.recursive)
    except scan.ScanError as exc:
        # 400 with the message as written: these are all things the user can
        # fix by typing a different path.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScanResponseOut(
        folder=result.folder,
        queued=len(result.queued),
        already_present=result.already_present,
        unsupported=result.unsupported,
        too_large=result.too_large,
        failed=result.failed,
    )


@router.get("/scan/preview", response_model=ScanPreviewOut)
def preview_scan(path: str, recursive: bool = False) -> ScanPreviewOut:
    """How many files a scan would consider, without ingesting anything."""
    try:
        folder = scan.validate_folder(path)
        count = len(scan.iter_candidate_files(folder, recursive))
    except scan.ScanError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ScanPreviewOut(folder=str(folder), supported_files=count)


@router.get("", response_model=list[FileOut])
def list_files(
    session: Session = Depends(get_db),
    status: ProcessingStatus | None = None,
    kind: FileKind | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[FileOut]:
    query = (
        select(File)
        .options(selectinload(File.chunks), selectinload(File.assets))
        .order_by(File.created_at.desc())
    )
    if status is not None:
        query = query.where(File.status == status)
    if kind is not None:
        query = query.where(File.kind == kind)

    rows = session.scalars(query.limit(limit).offset(offset)).all()
    return [FileOut.from_file(row) for row in rows]


@router.get("/stats", response_model=LibraryStatsOut)
def library_stats(session: Session = Depends(get_db)) -> LibraryStatsOut:
    status_rows = session.execute(select(File.status, func.count()).group_by(File.status)).all()
    kind_rows = session.execute(select(File.kind, func.count()).group_by(File.kind)).all()

    return LibraryStatsOut(
        total_files=session.scalar(select(func.count()).select_from(File)) or 0,
        by_status={status.value: count for status, count in status_rows},
        by_kind={kind.value: count for kind, count in kind_rows},
        total_chunks=session.scalar(select(func.count()).select_from(Chunk)) or 0,
        total_assets=session.scalar(select(func.count()).select_from(Asset)) or 0,
        pending_jobs=task_queue.pending_count(),
        disk_usage_mb=storage.disk_usage_mb(),
    )


@router.get("/supported-types", response_model=list[str])
def supported_types() -> list[str]:
    return SUPPORTED_EXTENSIONS


@router.get("/{file_id}", response_model=FileDetailOut)
def get_file(file_id: int, session: Session = Depends(get_db)) -> FileDetailOut:
    file_row = session.scalar(
        select(File)
        .where(File.id == file_id)
        .options(selectinload(File.chunks), selectinload(File.assets))
    )
    if file_row is None:
        raise HTTPException(status_code=404, detail=f"No file with id {file_id}")
    return FileDetailOut.from_file(file_row)


@router.get("/{file_id}/content")
def get_file_content(file_id: int, session: Session = Depends(get_db)) -> FileResponse:
    """Serve the original bytes, for previewing or opening a result."""
    file_row = session.get(File, file_id)
    if file_row is None:
        raise HTTPException(status_code=404, detail=f"No file with id {file_id}")

    path = storage.absolute_file_path(file_row.stored_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="The stored file is missing from disk.")
    return FileResponse(path, media_type=file_row.mime_type, filename=file_row.original_name)


@router.get("/{file_id}/assets/{asset_id}/image")
def get_asset_image(
    file_id: int, asset_id: int, session: Session = Depends(get_db)
) -> FileResponse:
    """Serve a derived image (page render, extracted picture, thumbnail)."""
    asset = session.get(Asset, asset_id)
    if asset is None or asset.file_id != file_id:
        raise HTTPException(status_code=404, detail="No such asset for this file.")

    # SOURCE_IMAGE points into the originals store; everything else is derived.
    from app.db.models import AssetKind

    path = (
        storage.absolute_file_path(asset.path)
        if asset.kind is AssetKind.SOURCE_IMAGE
        else storage.absolute_derived_path(asset.path)
    )
    if not path.exists():
        raise HTTPException(status_code=410, detail="The image is missing from disk.")
    return FileResponse(path)


@router.delete("/{file_id}")
def delete_file(file_id: int, session: Session = Depends(get_db)) -> dict[str, bool]:
    if not pipeline.delete_file(session, file_id):
        raise HTTPException(status_code=404, detail=f"No file with id {file_id}")
    return {"deleted": True}


@router.post("/{file_id}/reprocess", response_model=JobOut)
def reprocess_file(file_id: int, session: Session = Depends(get_db)) -> JobOut:
    """Re-run extraction, e.g. after a failure has been diagnosed."""
    file_row = session.get(File, file_id)
    if file_row is None:
        raise HTTPException(status_code=404, detail=f"No file with id {file_id}")

    for chunk in list(file_row.chunks):
        session.delete(chunk)
    for asset in list(file_row.assets):
        session.delete(asset)
    file_row.status = ProcessingStatus.PENDING
    file_row.error = None
    session.commit()

    job = ingest.submit_extraction(file_id, file_row.original_name)
    return JobOut(**job.as_dict())
