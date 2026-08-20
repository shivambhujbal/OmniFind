"""Background jobs for ingestion.

Kept apart from ``tasks/queue.py`` so the queue stays a generic mechanism and
the domain work lives with the domain.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select

from app.db.models import File, ProcessingStatus
from app.db.session import session_scope
from app.ingestion import pipeline
from app.logging_conf import get_logger
from app.tasks.queue import Job, Priority, task_queue

log = get_logger(__name__)


def _extract_job(file_id: int) -> Callable[[Job], None]:
    def run(job: Job) -> None:
        # A fresh session per job: SQLAlchemy Sessions are not thread-safe, and
        # this runs on the worker thread, not the request thread.
        with session_scope() as session:
            file_row = session.get(File, file_id)
            job.detail = file_row.original_name if file_row else f"file {file_id}"
            result = pipeline.extract_file(session, file_id)
            if result.error:
                # Recorded on the file row too; surface it on the job so the UI
                # can show it without a second lookup.
                raise RuntimeError(result.error)
            job.detail = f"{job.detail}: {result.chunk_count} chunks, {result.asset_count} images"
            name = file_row.original_name if file_row else str(file_id)

        # Chain into ML processing once extraction has committed. Queued rather
        # than called inline so the two show as separate jobs with separate
        # durations -- extraction is milliseconds, captioning can be a minute.
        # Imported here, not at module scope, to avoid an import cycle
        # (tasks.process -> ml.pipeline -> ... -> tasks).
        from app.tasks import process

        process.submit_processing(file_id, name)

    return run


def submit_extraction(file_id: int, name: str) -> Job:
    return task_queue.submit(
        f"extract:{name}",
        _extract_job(file_id),
        file_id=file_id,
        priority=Priority.EXTRACT,
    )


def requeue_unfinished() -> int:
    """Re-queue files left mid-flight by a crash or a forced quit.

    Called from the app lifespan. EXTRACTING means the process died partway
    through; PENDING means it never started. Either way the file is stuck
    forever unless something picks it back up.
    """
    with session_scope() as session:
        stuck = session.scalars(
            select(File).where(
                File.status.in_([ProcessingStatus.PENDING, ProcessingStatus.EXTRACTING])
            )
        ).all()
        pending = [(f.id, f.original_name) for f in stuck]

    for file_id, name in pending:
        submit_extraction(file_id, name)

    if pending:
        log.info("re-queued %d unfinished file(s) from the previous run", len(pending))
    return len(pending)
