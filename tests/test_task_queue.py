"""The background queue: ordering, isolation of failures, shutdown."""

from __future__ import annotations

import threading

from app.tasks.queue import Job, Priority, TaskQueue


def collector() -> tuple[list[str], threading.Event, int]:
    return [], threading.Event(), 0


def test_indexing_jumps_ahead_of_captioning() -> None:
    """Search must not wait behind a slow captioning pass.

    Observed under plain FIFO: five documents sat fully extracted but
    unsearchable for six minutes, queued behind one scanned page's captioning.
    Indexing takes about a second and is what makes a file findable at all, so
    it goes first.
    """
    queue = TaskQueue()
    order: list[str] = []
    done = threading.Event()

    def record(label: str, last: bool = False):  # noqa: ANN202
        def run(job: Job) -> None:
            order.append(label)
            if last:
                done.set()

        return run

    # Submit in the awkward order: the slow job first, as really happens.
    queue.submit("process:scan", record("process"), priority=Priority.PROCESS)
    queue.submit("extract:doc", record("extract"), priority=Priority.EXTRACT)
    queue.submit("index:doc", record("index", last=True), priority=Priority.INDEX)

    queue.start()
    try:
        assert done.wait(timeout=10), f"queue did not drain: {order}"
    finally:
        queue.stop()

    assert order.index("index") < order.index("process"), f"index ran too late: {order}"
    assert order.index("extract") < order.index("process"), f"extract ran too late: {order}"


def test_same_priority_stays_first_in_first_out() -> None:
    """Priority must not reorder work within a level."""
    queue = TaskQueue()
    order: list[int] = []
    done = threading.Event()

    def record(index: int, last: bool = False):  # noqa: ANN202
        def run(job: Job) -> None:
            order.append(index)
            if last:
                done.set()

        return run

    for i in range(5):
        queue.submit(f"index:{i}", record(i, last=(i == 4)), priority=Priority.INDEX)

    queue.start()
    try:
        assert done.wait(timeout=10)
    finally:
        queue.stop()

    assert order == [0, 1, 2, 3, 4]


def test_a_failing_job_does_not_kill_the_worker() -> None:
    """One bad file must not stop everything queued behind it."""
    queue = TaskQueue()
    done = threading.Event()

    def explode(job: Job) -> None:
        raise RuntimeError("simulated failure")

    def succeed(job: Job) -> None:
        done.set()

    failing = queue.submit("index:bad", explode, priority=Priority.INDEX)
    queue.submit("index:good", succeed, priority=Priority.INDEX)

    queue.start()
    try:
        assert done.wait(timeout=10), "the job after a failure never ran"
    finally:
        queue.stop()

    assert failing.state.value == "failed"
    assert failing.error and "simulated failure" in failing.error


def test_stop_is_not_blocked_behind_queued_work() -> None:
    """Shutdown must not wait for a captioning pass that has not started."""
    queue = TaskQueue()
    started = threading.Event()
    release = threading.Event()

    def slow(job: Job) -> None:
        started.set()
        release.wait(timeout=5)

    def never(job: Job) -> None:  # pragma: no cover - must not run
        raise AssertionError("queued work ran during shutdown")

    queue.submit("process:slow", slow, priority=Priority.PROCESS)
    queue.start()
    assert started.wait(timeout=5)

    for i in range(3):
        queue.submit(f"process:{i}", never, priority=Priority.PROCESS)

    release.set()
    queue.stop(timeout=5)


def test_pending_count_tracks_outstanding_work() -> None:
    queue = TaskQueue()
    queue.submit("index:a", lambda job: None, priority=Priority.INDEX)
    queue.submit("index:b", lambda job: None, priority=Priority.INDEX)
    assert queue.pending_count() == 2
