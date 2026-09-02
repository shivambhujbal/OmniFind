"""Background job for vector indexing.

A third job after extract and process, for the same reason those are separate:
each has a very different duration, and the UI should be able to say which
stage a file is in rather than showing one opaque spinner. It is also the stage
most likely to need re-running on its own, after a model change or an index
rebuild.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select

from app.config import settings
from app.db.models import Asset, Chunk, File, ProcessingStatus
from app.db.session import session_scope
from app.logging_conf import get_logger
from app.ml import indexing
from app.ml.pipeline import CLIP_INDEXABLE_KINDS
from app.tasks.queue import Job, Priority, task_queue

log = get_logger(__name__)


def _index_job(file_id: int) -> Callable[[Job], None]:
    def run(job: Job) -> None:
        with session_scope() as session:
            file_row = session.get(File, file_id)
            job.detail = file_row.original_name if file_row else f"file {file_id}"
            result = indexing.index_file(session, file_id)
            job.detail = (
                f"{job.detail}: {result.text_points} text, {result.image_points} image vectors"
            )

    return run


def submit_indexing(file_id: int, name: str) -> Job:
    return task_queue.submit(
        f"index:{name}",
        _index_job(file_id),
        file_id=file_id,
        priority=Priority.INDEX,
    )


def requeue_unindexed() -> int:
    """Queue anything with content that has not reached the vector store.

    Driven by the per-row `text_embedded` / `clip_embedded` flags rather than
    the file's status, because a file can be READY (OCR and captions done) yet
    only partly indexed if the process died mid-embedding.
    """
    with session_scope() as session:
        file_ids = set(
            session.scalars(
                select(Chunk.file_id)
                .where(Chunk.text_embedded.is_(False), Chunk.searchable.is_(True))
                .distinct()
            ).all()
        )
        if settings.enable_image_search:
            # Same kind filter as the indexer. Without it every document
            # holding an embedded image would be re-queued on every start,
            # indexed to zero, and queued again on the next one.
            file_ids |= set(
                session.scalars(
                    select(Asset.file_id)
                    .where(
                        Asset.clip_embedded.is_(False),
                        Asset.kind.in_(CLIP_INDEXABLE_KINDS),
                    )
                    .distinct()
                ).all()
            )
        # Only files whose content is final; anything earlier gets indexed when
        # its own pipeline stage completes.
        pending = [
            (f.id, f.original_name)
            for f in session.scalars(
                select(File).where(File.id.in_(file_ids), File.status == ProcessingStatus.READY)
            ).all()
        ]

    for file_id, name in pending:
        submit_indexing(file_id, name)

    if pending:
        log.info("queued %d file(s) for vector indexing", len(pending))
    return len(pending)
