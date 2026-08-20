"""Embed a file's chunks and images into Qdrant.

Runs after ``ml/pipeline.py`` has produced OCR text and captions, so a single
pass indexes body text, OCR output and captions together.

Text and images are embedded in two separate passes -- all the text for a file,
then all the images -- rather than interleaved. Under ``ml_low_memory`` each
switch between bge and CLIP evicts the other, so interleaving would reload a
multi-gigabyte model per chunk.

``Chunk.text_embedded`` and ``Asset.clip_embedded`` make this resumable: a
crash halfway leaves the finished half marked, and the next run continues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Asset, Chunk, File
from app.logging_conf import get_logger
from app.ml import embeddings_clip, embeddings_text, loaders
from app.ml.pipeline import PROCESSABLE_KINDS, asset_path
from app.vectors import store

log = get_logger(__name__)

# Embedding is much cheaper than captioning, so batches can be larger; still
# bounded so a 2000-chunk book does not build one enormous request.
TEXT_BATCH = 64
IMAGE_BATCH = 8

# Enough to show as a search snippet without duplicating the whole chunk into
# the vector payload, which would double storage for no benefit.
SNIPPET_CHARS = 400


@dataclass(frozen=True)
class IndexResult:
    file_id: int
    text_points: int
    image_points: int


def ensure_ready() -> None:
    """Create collections sized from the *actual* loaded models.

    Reading the widths from the models rather than trusting config means a
    model swap is caught at startup instead of producing a silently corrupt
    index.
    """
    store.ensure_collections(
        text_dim=embeddings_text.embedding_dim(),
        image_dim=embeddings_clip.embedding_dim(),
    )


def index_file(session: Session, file_id: int) -> IndexResult:
    file_row = session.get(File, file_id)
    if file_row is None:
        raise LookupError(f"No file with id {file_id}")

    text_points = _index_text(session, file_row)
    image_points = _index_images(session, file_row)

    if text_points or image_points:
        log.info(
            "indexed %s: %d text points, %d image points",
            file_row.original_name,
            text_points,
            image_points,
        )
    return IndexResult(file_id, text_points, image_points)


def _index_text(session: Session, file_row: File) -> int:
    pending = session.scalars(
        select(Chunk).where(
            Chunk.file_id == file_row.id,
            Chunk.text_embedded.is_(False),
            Chunk.searchable.is_(True),
        )
    ).all()
    if not pending:
        return 0

    store.ensure_collections(
        text_dim=embeddings_text.embedding_dim(),
        image_dim=settings.clip_embedding_dim,
    )

    written = 0
    for start in range(0, len(pending), TEXT_BATCH):
        batch = pending[start : start + TEXT_BATCH]
        # No slot: bge stays resident, so this never contends with captioning.
        vectors = embeddings_text.embed_documents([c.text for c in batch])

        store.upsert_text(
            ids=[c.id for c in batch],
            vectors=vectors,
            payloads=[_text_payload(file_row, c) for c in batch],
        )
        for chunk in batch:
            chunk.text_embedded = True
        session.commit()
        written += len(batch)

    return written


def _index_images(session: Session, file_row: File) -> int:
    if not settings.enable_image_understanding:
        # Leaves `clip_embedded` False, so turning the flag back on and
        # restarting picks these up through `requeue_unindexed` -- no re-import.
        return 0

    pending = session.scalars(
        select(Asset).where(
            Asset.file_id == file_row.id,
            Asset.clip_embedded.is_(False),
            Asset.kind.in_(PROCESSABLE_KINDS),
        )
    ).all()
    if not pending:
        return 0

    store.ensure_collections(
        text_dim=settings.text_embedding_dim,
        image_dim=embeddings_clip.embedding_dim(),
    )

    written = 0
    for start in range(0, len(pending), IMAGE_BATCH):
        batch = pending[start : start + IMAGE_BATCH]
        by_path: dict[Path, Asset] = {asset_path(a): a for a in batch}

        # embed_images reports which paths it actually read, so an unreadable
        # image cannot shift every later vector onto the wrong asset.
        with loaders.heavy_model_slot("clip"):
            vectors, kept = embeddings_clip.embed_images(list(by_path))
        if not kept:
            continue

        kept_assets = [by_path[p] for p in kept]
        store.upsert_images(
            ids=[a.id for a in kept_assets],
            vectors=vectors,
            payloads=[_image_payload(file_row, a) for a in kept_assets],
        )
        for asset in kept_assets:
            asset.clip_embedded = True
        session.commit()
        written += len(kept_assets)

    return written


def _text_payload(file_row: File, chunk: Chunk) -> dict[str, object]:
    return {
        "file_id": file_row.id,
        "chunk_id": chunk.id,
        "kind": chunk.kind.value,
        "file_name": file_row.original_name,
        "file_kind": file_row.kind.value,
        "title": file_row.title,
        "page_number": chunk.page_number,
        "snippet": chunk.text[:SNIPPET_CHARS],
    }


def _image_payload(file_row: File, asset: Asset) -> dict[str, object]:
    return {
        "file_id": file_row.id,
        "asset_id": asset.id,
        "kind": asset.kind.value,
        "file_name": file_row.original_name,
        "file_kind": file_row.kind.value,
        "title": file_row.title,
        "page_number": asset.page_number,
        "caption": asset.caption,
    }
