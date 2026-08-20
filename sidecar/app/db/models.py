"""SQLAlchemy models.

Three tables:

``files``   one row per ingested document, keyed by content hash
``assets``  images belonging to a file (rendered pages, embedded pictures)
``chunks``  every searchable unit of text, whatever produced it

Chunks deliberately cover text extraction, OCR output and image captions with
one shape. Search ranks them together, so giving each source its own table would
mean three near-identical queries and a union at read time for no benefit.

There is no user, owner or tenant column anywhere: one install, one person,
their own files (hard constraint 5).
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class FileKind(str, enum.Enum):
    """What the file is, which decides the extraction path."""

    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"


class ProcessingStatus(str, enum.Enum):
    """Lifecycle of one file.

    PENDING -> EXTRACTING -> EXTRACTED -> PROCESSING -> READY
    with FAILED reachable from any working state.

    EXTRACTED (phase 2) means text and images are on disk and in the database.
    READY (phase 3/4) additionally means embeddings, OCR and captions exist and
    have been upserted into Qdrant. Keeping the two apart is what lets the UI
    show a file's content before its embeddings finish.
    """

    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ChunkKind(str, enum.Enum):
    """Where a chunk's text came from -- affects ranking, not storage."""

    TEXT = "text"  # extracted directly from the document
    OCR = "ocr"  # recognised from an image or scanned page
    CAPTION = "caption"  # generated description of an image


class AssetKind(str, enum.Enum):
    PAGE_IMAGE = "page_image"  # a PDF page rendered to a bitmap for OCR
    EMBEDDED_IMAGE = "embedded_image"  # a picture pulled out of a PDF/DOCX
    SOURCE_IMAGE = "source_image"  # the uploaded file itself, when it is an image
    THUMBNAIL = "thumbnail"


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Content hash, not filename: re-uploading the same bytes under a different
    # name must not re-run a multi-minute ML pipeline.
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    original_name: Mapped[str] = mapped_column(String(512))
    # Relative to settings.files_dir, so the data directory stays relocatable.
    stored_path: Mapped[str] = mapped_column(String(1024))
    kind: Mapped[FileKind] = mapped_column(Enum(FileKind, native_enum=False), index=True)
    mime_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)

    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    title: Mapped[str | None] = mapped_column(String(512), default=None)

    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, native_enum=False),
        default=ProcessingStatus.PENDING,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )
    assets: Mapped[list[Asset]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (CheckConstraint("size_bytes >= 0", name="ck_files_size_nonneg"),)

    def __repr__(self) -> str:
        return f"<File {self.id} {self.original_name!r} {self.status.value}>"


class Asset(Base):
    """An image derived from, or identical to, a file."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True)

    kind: Mapped[AssetKind] = mapped_column(Enum(AssetKind, native_enum=False))
    # Relative to settings.derived_dir (or files_dir for SOURCE_IMAGE).
    path: Mapped[str] = mapped_column(String(1024))
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)

    # Set in phase 3. Null means "not processed yet", which is how the task
    # queue finds work to do after a restart.
    caption: Mapped[str | None] = mapped_column(Text, default=None)
    clip_embedded: Mapped[bool] = mapped_column(default=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    file: Mapped[File] = relationship(back_populates="assets")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="asset", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("file_id", "kind", "ordinal", name="uq_assets_file_kind_ordinal"),
    )

    def __repr__(self) -> str:
        return f"<Asset {self.id} file={self.file_id} {self.kind.value}>"


class Chunk(Base):
    """One searchable passage of text."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True)
    # Set when the text came from an image (OCR or caption); null for body text.
    asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), default=None, index=True
    )

    kind: Mapped[ChunkKind] = mapped_column(
        Enum(ChunkKind, native_enum=False), default=ChunkKind.TEXT, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)

    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    # Phase 4 writes the Qdrant point id back here so a chunk can be deleted
    # from the vector store without a scan.
    text_embedded: Mapped[bool] = mapped_column(default=False, index=True)

    # False for passages with no retrievable content -- a patent's list of cited
    # numbers, a figure's callout labels, a diagram OCR'd as one repeated word.
    # They stay in the database (they are part of the document, and visible in
    # the file view) but are kept out of the vector index, where they would
    # otherwise surface against unrelated queries. See ingestion/quality.py.
    searchable: Mapped[bool] = mapped_column(default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    file: Mapped[File] = relationship(back_populates="chunks")
    asset: Mapped[Asset | None] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("file_id", "kind", "ordinal", name="uq_chunks_file_kind_ordinal"),
        CheckConstraint("char_count >= 0", name="ck_chunks_char_count_nonneg"),
    )

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return f"<Chunk {self.id} file={self.file_id} {self.kind.value} {preview!r}>"
