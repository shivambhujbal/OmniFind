"""Response models for the HTTP API.

Separate from the SQLAlchemy models on purpose: the wire format is a contract
with the frontend (Streamlit now, React later) and should not shift every time
a column is added.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import Asset, ChunkKind, File, FileKind, ProcessingStatus


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    page_number: int | None
    width: int | None
    height: int | None
    caption: str | None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: ChunkKind
    ordinal: int
    page_number: int | None
    text: str
    char_count: int


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sha256: str
    original_name: str
    title: str | None
    kind: FileKind
    mime_type: str
    size_bytes: int
    page_count: int | None
    status: ProcessingStatus
    error: str | None
    created_at: datetime
    updated_at: datetime

    chunk_count: int = 0
    asset_count: int = 0

    @classmethod
    def from_file(cls, file_row: File) -> FileOut:
        out = cls.model_validate(file_row)
        out.chunk_count = len(file_row.chunks)
        out.asset_count = len(file_row.assets)
        return out


class FileDetailOut(FileOut):
    chunks: list[ChunkOut] = []
    assets: list[AssetOut] = []

    @classmethod
    def from_file(cls, file_row: File) -> FileDetailOut:  # type: ignore[override]
        out = cls.model_validate(file_row)
        out.chunk_count = len(file_row.chunks)
        out.asset_count = len(file_row.assets)
        out.chunks = [ChunkOut.model_validate(c) for c in file_row.chunks]
        out.assets = [_asset_out(a) for a in file_row.assets]
        return out


def _asset_out(asset: Asset) -> AssetOut:
    return AssetOut(
        id=asset.id,
        kind=asset.kind.value,
        page_number=asset.page_number,
        width=asset.width,
        height=asset.height,
        caption=asset.caption,
    )


class UploadResultOut(BaseModel):
    """One entry per uploaded file. A rejected file reports why, in plain words."""

    original_name: str
    accepted: bool
    duplicate: bool = False
    file_id: int | None = None
    job_id: str | None = None
    reason: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResultOut]
    accepted: int
    rejected: int


class ScanRequest(BaseModel):
    """Index a folder the user already has, in place."""

    path: str
    recursive: bool = False


class ScanPreviewOut(BaseModel):
    folder: str
    supported_files: int


class ScanResponseOut(BaseModel):
    folder: str
    # Newly registered and queued for processing.
    queued: int
    # Recognised by content hash as already in the library.
    already_present: int
    unsupported: int
    too_large: int
    failed: list[str]


class JobOut(BaseModel):
    id: str
    name: str
    state: str
    detail: str
    file_id: int | None
    error: str | None
    # Lower runs sooner; see tasks.queue.Priority. Exposed so the UI can explain
    # why a cheap job overtook an expensive one.
    priority: int
    queued_seconds: float
    duration_seconds: float | None


class LibraryStatsOut(BaseModel):
    total_files: int
    by_status: dict[str, int]
    by_kind: dict[str, int]
    total_chunks: int
    total_assets: int
    pending_jobs: int
    disk_usage_mb: dict[str, float]
