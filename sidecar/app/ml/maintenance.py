"""Bring an existing index in line with the current quality rules.

When the low-information rules change, chunks already in the database were
classified under the old ones. Re-importing every file to fix that would mean
re-running OCR and captioning, which on CPU is measured in hours.

This re-runs the *classification* only — pure text analysis, no file I/O, no
models — and removes from the vector store anything that no longer qualifies.
Cheap enough to run at every startup, and idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Asset, Chunk
from app.ingestion import quality
from app.logging_conf import get_logger
from app.vectors import store

log = get_logger(__name__)


@dataclass(frozen=True)
class ReclassifyResult:
    checked: int
    demoted: int
    promoted: int


def reclassify_chunks(session: Session) -> ReclassifyResult:
    """Recompute `searchable` for every chunk and sync the vector store."""
    chunks = session.scalars(select(Chunk)).all()
    if not chunks:
        return ReclassifyResult(0, 0, 0)

    demoted: list[int] = []
    promoted = 0

    for chunk in chunks:
        should_be = quality.is_searchable(chunk.text)
        if should_be == chunk.searchable:
            continue

        chunk.searchable = should_be
        if should_be:
            # Newly eligible: clear the embedded flag so the indexer picks it
            # up on its next pass.
            chunk.text_embedded = False
            promoted += 1
        else:
            demoted.append(chunk.id)
            chunk.text_embedded = False

    if demoted or promoted:
        session.commit()

    if demoted:
        # Delete by point id: a chunk's id *is* its point id, so this is direct
        # addressing rather than a scan.
        store.delete_points(settings.text_collection, demoted)
        log.info("removed %d low-information chunk(s) from the search index", len(demoted))
    if promoted:
        log.info("%d chunk(s) became searchable and will be re-indexed", promoted)

    return ReclassifyResult(len(chunks), len(demoted), promoted)


def prune_image_vectors(session: Session) -> int:
    """Drop CLIP vectors for assets that are no longer indexable.

    The indexable set narrowed to standalone image files (see
    `ml.pipeline.CLIP_INDEXABLE_KINDS`). Anything embedded before that change
    is still sitting in the images collection, where it goes on matching
    queries and crowding out the standalone photographs the collection exists
    for. Deleting the rows in SQLite is not enough: the vector store is a
    separate database and keeps whatever it was given.

    Same contract as `reclassify_chunks` -- classification only, no models, no
    file I/O, idempotent, cheap enough to run at every startup.
    """
    from app.ml.pipeline import CLIP_INDEXABLE_KINDS

    stale = session.scalars(
        select(Asset).where(
            Asset.clip_embedded.is_(True),
            Asset.kind.not_in(CLIP_INDEXABLE_KINDS),
        )
    ).all()
    if not stale:
        return 0

    store.delete_points(settings.image_collection, [a.id for a in stale])
    for asset in stale:
        # Reset rather than leave it True: the flag means "has a vector", and
        # after this it does not. If the rule ever widens again, this is what
        # lets `requeue_unindexed` pick them back up.
        asset.clip_embedded = False
    session.commit()

    log.info("pruned %d image vector(s) for no-longer-indexable assets", len(stale))
    return len(stale)
