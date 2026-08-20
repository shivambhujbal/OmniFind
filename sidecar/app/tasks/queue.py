"""In-process background task queue.

One process, one user, one machine -- so there is no broker, no Celery and no
Redis (hard constraint: "in-process async task handling"). What is needed is
narrower than any of those: run long jobs off the request thread, one at a time,
and let the UI poll their progress.

**Why a thread and not `asyncio` tasks.** Ingestion and inference are CPU/GPU
bound and synchronous (PyMuPDF, torch, PIL). Running them as coroutines on the
event loop would block it, freezing `/health` and every other request for the
duration of a model forward pass. A single worker thread pulling from a queue
keeps the loop responsive while still serialising the GPU work, which is what we
want anyway: two concurrent ViT-H forward passes on one device are slower than
two sequential ones and can exhaust VRAM.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.logging_conf import get_logger

log = get_logger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Priority(int, Enum):
    """Lower runs sooner. Ties break by submission order, so each level is FIFO.

    Strict FIFO across all work is the wrong policy here because the stages have
    wildly different costs and payoffs. Captioning one image takes ~390s on CPU
    and only *improves* a file that is already searchable; indexing takes about a
    second and is what makes a file searchable at all.

    Under FIFO, dropping six documents in meant five of them sat fully extracted
    but unsearchable for six minutes, waiting behind one scanned page's
    captioning. Indexing first means search works almost immediately and captions
    fill in behind it.
    """

    INDEX = 10
    EXTRACT = 20
    PROCESS = 50


@dataclass
class Job:
    id: str
    name: str
    state: JobState = JobState.QUEUED
    detail: str = ""
    file_id: int | None = None
    error: str | None = None
    priority: int = Priority.PROCESS
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "detail": self.detail,
            "file_id": self.file_id,
            "error": self.error,
            "priority": int(self.priority),
            "queued_seconds": round((self.started_at or time.time()) - self.created_at, 1),
            "duration_seconds": (
                round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at
                else None
            ),
        }


# Jobs are kept in memory only. They describe work in flight, and the durable
# record of what happened is the file's `status` column -- so losing the list on
# restart costs nothing, and a restart re-queues unfinished files from SQLite.
_MAX_HISTORY = 200


class TaskQueue:
    def __init__(self) -> None:
        # (priority, sequence, payload). The sequence keeps each priority level
        # FIFO and, just as importantly, guarantees the heap never has to
        # compare two Job objects -- which are not orderable.
        self._queue: queue.PriorityQueue[
            tuple[int, int, tuple[Job, Callable[[Job], None]] | None]
        ] = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    # --- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="task-worker", daemon=True)
        self._worker.start()
        log.info("task worker started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop after the current job. Called from the FastAPI lifespan shutdown."""
        if self._worker is None:
            return
        self._stopping.set()
        # Priority -1 jumps ahead of every queued job so shutdown is not stuck
        # behind a captioning pass.
        self._queue.put((-1, next(self._sequence), None))
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            # Daemon thread, so the process still exits; say so rather than hang.
            log.warning("task worker did not finish within %.0fs; exiting anyway", timeout)
        else:
            log.info("task worker stopped")
        self._worker = None

    # --- submission -----------------------------------------------------

    def submit(
        self,
        name: str,
        handler: Callable[[Job], None],
        file_id: int | None = None,
        priority: int = Priority.PROCESS,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, file_id=file_id, priority=priority)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim_history()
        self._queue.put((int(priority), next(self._sequence), (job, handler)))
        log.debug("queued job %s (%s, priority %d)", job.id, name, priority)
        return job

    # --- inspection -----------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in reversed(self._order[-limit:])]

    def pending_count(self) -> int:
        with self._lock:
            return sum(
                1 for job in self._jobs.values() if job.state in (JobState.QUEUED, JobState.RUNNING)
            )

    def _trim_history(self) -> None:
        """Drop finished jobs beyond the history cap. Caller holds the lock."""
        while len(self._order) > _MAX_HISTORY:
            oldest = self._order[0]
            if self._jobs[oldest].state in (JobState.QUEUED, JobState.RUNNING):
                break  # never evict work that has not finished
            self._order.pop(0)
            self._jobs.pop(oldest, None)

    # --- worker ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            _priority, _sequence, payload = self._queue.get()
            if payload is None:
                break
            job, handler = payload

            job.state = JobState.RUNNING
            job.started_at = time.time()
            try:
                handler(job)
                job.state = JobState.DONE
            except Exception as exc:  # noqa: BLE001 - a failed job must not kill the worker
                job.state = JobState.FAILED
                job.error = str(exc)[:2000]
                log.exception("job %s (%s) failed", job.id, job.name)
            finally:
                job.finished_at = time.time()
                self._queue.task_done()


task_queue = TaskQueue()
