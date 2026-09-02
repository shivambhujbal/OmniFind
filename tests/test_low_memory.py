"""Low-memory mode: one large model resident, and safely so.

Regression cover for a crash: the worker was mid-caption holding moondream2
(4.1GB) when an HTTP search request tried to load CLIP. Evicting a model another
thread is actively using frees nothing, CLIP allocated on top, and the whole
sidecar died on the Windows commit limit -- taking the user's in-flight ingest
with it.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.config import settings
from app.ml import loaders


def test_slot_serialises_use_so_eviction_can_actually_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ml_low_memory", True)

    holding = threading.Event()
    release = threading.Event()
    overlapped = []

    def worker() -> None:
        with loaders.heavy_model_slot("captioner"):
            holding.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=worker)
    thread.start()
    assert holding.wait(timeout=5)

    # While the worker holds a model, a second acquirer must NOT get in.
    try:
        with loaders.heavy_model_slot("clip", timeout=0.2):
            overlapped.append("clip loaded while captioner was resident")
    except loaders.ModelBusyError:
        pass

    release.set()
    thread.join(timeout=5)

    assert not overlapped, overlapped[0]


def test_the_slot_is_released_and_reusable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ml_low_memory", True)

    with loaders.heavy_model_slot("text encoder", timeout=1):
        pass
    with loaders.heavy_model_slot("clip", timeout=1):
        pass


def test_the_slot_is_released_even_when_the_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed captioning pass must not wedge the app permanently."""
    monkeypatch.setattr(settings, "ml_low_memory", True)

    with pytest.raises(RuntimeError, match="boom"), loaders.heavy_model_slot("captioner"):
        raise RuntimeError("boom")

    with loaders.heavy_model_slot("clip", timeout=1):
        pass


def test_the_slot_is_reentrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested acquisition on one thread must not deadlock against itself."""
    monkeypatch.setattr(settings, "ml_low_memory", True)

    with loaders.heavy_model_slot("captioner"), loaders.heavy_model_slot("captioner"):
        pass


def test_the_slot_does_nothing_when_low_memory_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a GPU the models coexist; serialising them would only cost throughput."""
    monkeypatch.setattr(settings, "ml_low_memory", False)

    holding = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with loaders.heavy_model_slot("captioner"):
            holding.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=worker)
    thread.start()
    assert holding.wait(timeout=5)

    start = time.monotonic()
    with loaders.heavy_model_slot("clip", timeout=0.2):
        pass
    assert time.monotonic() - start < 0.2, "the slot blocked with low-memory mode off"

    release.set()
    thread.join(timeout=5)


def test_search_returns_text_results_when_the_worker_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image search is a bonus; a busy worker must not fail the whole query.

    Before this, the request thread loaded CLIP anyway and crashed the process.
    """
    monkeypatch.setattr(settings, "ml_low_memory", True)
    monkeypatch.setattr(settings, "image_model_wait_seconds", 0.1)

    from app.search import service
    from app.vectors.store import VectorHit

    def fake_text_query(query: str):  # noqa: ANN202
        import numpy as np

        return np.zeros(4, dtype=np.float32)

    def fake_text_search(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return [
            VectorHit(
                1,
                0.8,
                "text_chunks",
                {"file_id": 1, "chunk_id": 1, "file_name": "a.pdf", "file_kind": "pdf"},
            )
        ]

    def must_not_run(query: str):  # noqa: ANN202
        raise AssertionError("CLIP was loaded while the worker held the slot")

    monkeypatch.setattr(service.embeddings_text, "embed_query", fake_text_query)
    monkeypatch.setattr(service.store, "search_text", fake_text_search)
    monkeypatch.setattr(service.embeddings_clip, "embed_query", must_not_run)

    holding = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with loaders.heavy_model_slot("captioner"):
            holding.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=worker)
    thread.start()
    assert holding.wait(timeout=5)

    try:
        # The text encoder is stubbed, so its own slot acquisition is what would
        # block -- give it room by releasing after the call starts.
        threading.Timer(0.3, release.set).start()
        response = service.search("anything", include_images=True)
    finally:
        release.set()
        thread.join(timeout=5)

    assert response.results, "text results were lost because images were busy"
    assert not response.searched_images, "should have reported image search as skipped"


# --- releasing after use, not at the next load --------------------------


def test_captioner_is_released_as_soon_as_captioning_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeing at the next load is what killed the sidecar.

    Evicting moondream2 (4.1GB) and building ViT-H/14 (2.9GB peak) in the same
    instant exceeded a 19.2GB commit limit with ~5GB free: the process vanished
    mid-log-line with no traceback, straight after "evicted moondream to make
    room for clip". Releasing when the pass ends puts minutes between the free
    and the next allocation.
    """
    from app.ml import loaders

    monkeypatch.setattr(loaders.settings, "ml_low_memory", True)

    cleared: list[str] = []

    class FakeLoader:
        def __init__(self, name: str) -> None:
            self.name = name

        def cache_info(self):  # noqa: ANN202
            return SimpleNamespace(currsize=1)

        def cache_clear(self) -> None:
            cleared.append(self.name)

    monkeypatch.setattr(
        loaders,
        "_HEAVY_LOADERS",
        {"moondream": FakeLoader("moondream"), "clip": FakeLoader("clip")},
    )

    assert loaders.release_heavy_model("moondream2") is True
    assert cleared == ["moondream"], "the engine name must map to the loader key"


def test_release_maps_blip2_without_mangling_the_digit(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "moondream2" -> "moondream" but "blip2" -> "blip2".

    Any rule that strips the trailing digit frees nothing for BLIP-2 and the
    crash comes straight back, silently.
    """
    from app.ml import loaders

    monkeypatch.setattr(loaders.settings, "ml_low_memory", True)
    cleared: list[str] = []

    class FakeLoader:
        def cache_info(self):  # noqa: ANN202
            return SimpleNamespace(currsize=1)

        def cache_clear(self) -> None:
            cleared.append("blip2")

    monkeypatch.setattr(loaders, "_HEAVY_LOADERS", {"blip2": FakeLoader()})

    assert loaders.release_heavy_model("blip2") is True
    assert cleared == ["blip2"]


def test_release_is_a_no_op_on_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeping models resident is the right policy when memory allows."""
    from app.ml import loaders

    monkeypatch.setattr(loaders.settings, "ml_low_memory", False)
    assert loaders.release_heavy_model("moondream2") is False
