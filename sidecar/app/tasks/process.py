"""Background jobs for ML processing (OCR + captioning).

Submitted automatically after extraction succeeds, so dropping a file runs the
whole chain without further input. Kept as a *separate* job from extraction
rather than one long one, because the two have very different durations: text
extraction is milliseconds, captioning a scanned page can be a minute on CPU.
Splitting them lets the UI show "extracted, describing images…" instead of one
opaque spinner, and lets a captioning failure leave the extracted text intact.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select

from app.config import settings
from app.db.models import Asset, File, ProcessingStatus
from app.db.session import session_scope
from app.logging_conf import get_logger
from app.ml import pipeline as ml_pipeline
from app.tasks.queue import Job, Priority, task_queue

log = get_logger(__name__)


def _process_job(file_id: int) -> Callable[[Job], None]:
    def run(job: Job) -> None:
        # A fresh session per job: Sessions are not thread-safe and this runs on
        # the worker thread.
        with session_scope() as session:
            file_row = session.get(File, file_id)
            name = file_row.original_name if file_row else f"file {file_id}"
            job.detail = name
            result = ml_pipeline.process_file(session, file_id)
            if result.error:
                raise RuntimeError(result.error)
            if result.deferred_reason:
                # Not an error: the file's text is fine, a model is just not
                # installed yet. Say so plainly rather than showing a failure.
                job.detail = f"{job.detail}: image processing skipped (model not installed)"
            else:
                job.detail = (
                    f"{job.detail}: {result.ocr_chunks} OCR chunks, {result.captions} captions"
                )

        # Index whatever exists, including when captioning was deferred: the
        # extracted text is searchable on its own, and waiting for a model the
        # user may never install would leave the library permanently unfindable.
        # Imported here rather than at module scope to avoid an import cycle
        # (tasks.index -> ml.indexing -> ml.pipeline -> ... -> tasks).
        from app.tasks import index

        index.submit_indexing(file_id, name)

    return run


def submit_processing(file_id: int, name: str) -> Job:
    return task_queue.submit(
        f"process:{name}",
        _process_job(file_id),
        file_id=file_id,
        priority=Priority.PROCESS,
    )


def requeue_unprocessed() -> int:
    """Queue files that are extracted but not yet described.

    Called at startup alongside ``ingest.requeue_unfinished``. EXTRACTED means
    ingestion finished but ML never ran (or the process died during it);
    PROCESSING means it died partway. Both need picking back up, or the file
    stays permanently half-indexed.
    """
    with session_scope() as session:
        stuck = session.scalars(
            select(File).where(
                File.status.in_([ProcessingStatus.EXTRACTED, ProcessingStatus.PROCESSING])
            )
        ).all()
        pending = [(f.id, f.original_name) for f in stuck]

    for file_id, name in pending:
        submit_processing(file_id, name)

    if pending:
        log.info("queued %d file(s) for ML processing", len(pending))
    return len(pending)


def requeue_uncaptioned() -> int:
    """Pick up images that predate captioning being switched on.

    Turning the flag on is not enough on its own: those files are already
    READY, and `process_file` returns immediately for a READY file rather than
    redoing hours of OCR. So they are stepped back to EXTRACTED, which is the
    state that means "content is on disk, the vision models have not finished
    with it" -- exactly true here -- and the normal queue takes it from there.

    Driven by the per-asset `caption` column rather than by status, for the
    same reason `requeue_unindexed` is: status describes the file, and this is
    a fact about one asset. Only captionable kinds count, or every document
    holding a figure would come back on every start and never settle.
    """
    if not settings.enable_captioning:
        return 0

    with session_scope() as session:
        files = session.scalars(
            select(File)
            .join(Asset, Asset.file_id == File.id)
            .where(
                File.status == ProcessingStatus.READY,
                Asset.caption.is_(None),
                Asset.kind.in_(ml_pipeline.CLIP_INDEXABLE_KINDS),
            )
            .distinct()
        ).all()
        pending = [(f.id, f.original_name) for f in files]
        for file_row in files:
            file_row.status = ProcessingStatus.EXTRACTED
        session.commit()

    for file_id, name in pending:
        submit_processing(file_id, name)

    if pending:
        log.info("queued %d file(s) for captioning", len(pending))
    return len(pending)
