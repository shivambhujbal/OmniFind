"""The ONLY module that constructs a model or chooses a device.

Hard constraint 2. Every loader here takes ``device`` with ``settings.ml_device``
as its default; no other module in the codebase mentions ``"cuda"`` or ``"cpu"``,
calls ``.to(device)``, or builds a model. ``tests/test_constraints.py`` fails the
build if that stops being true.

Why it is worth the discipline: adding a CPU fallback later then means editing
one function in ``config.py``, not auditing every call site. See
``docs/cpu-fallback-plan.md``.

**Loading is lazy and cached.** Models are built on first use, not at import, so
the sidecar starts in under a second and a user who only ever ingests text PDFs
never pays to load a 4 GB vision model. Each loader is ``lru_cache``d, so the
weights are read from disk exactly once per process.

**Thread safety.** The task worker is a single thread, so inference itself is
serialised. But an HTTP request embedding a search query can race the worker's
first call to the same loader, and building a model twice concurrently would
double peak memory. A lock around construction prevents that; it is uncontended
after the first call.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import sys
import threading
import types
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Device, settings
from app.logging_conf import get_logger

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

# Guards construction, not inference. See the module docstring.
_BUILD_LOCK = threading.Lock()

# Guards *use* of a heavy model under ml_low_memory, so that eviction can
# actually free memory. Reentrant: a captioning pass may nest calls that each
# ask for the same slot. See `heavy_model_slot`.
_HEAVY_SLOT = threading.RLock()


class ModelNotAvailableError(RuntimeError):
    """A model's files are missing from the local models directory.

    Raised instead of letting a library attempt a download, which would either
    hang or fail obscurely with networking disabled. The message names the
    directory and the command that populates it.
    """

    def __init__(self, name: str, path: Path, detail: str = "") -> None:
        self.name = name
        self.path = path
        message = f"{name} is not available at {path}."
        if detail:
            message += f" {detail}"
        message += " Run `python scripts/download_models.py` on a machine with internet."
        super().__init__(message)


class ModelBusyError(RuntimeError):
    """Another thread is using a large model and it cannot be evicted yet.

    Only possible under ``ml_low_memory``, where one heavy model is resident at
    a time. Callers that have something useful to return without the model
    should catch this and degrade; callers that do not should surface it.
    """


@contextlib.contextmanager
def heavy_model_slot(name: str, timeout: float | None = None) -> Iterator[None]:
    """Hold the right to have ``name`` resident, for the duration of the block.

    Eviction alone is not enough to keep one model resident at a time. The
    worker thread can be halfway through a captioning pass -- holding a live
    reference to moondream2 -- when an HTTP request asks for CLIP. Clearing the
    cache then frees nothing, CLIP allocates on top of moondream2, and the
    process dies on the commit limit. That is exactly how the sidecar crashed
    mid-search during an ingest.

    So *use* is serialised, not just construction: whoever holds this slot is
    the only one with a heavy model in play, which makes eviction meaningful.

    Outside low-memory mode this is a no-op -- on a GPU the models coexist
    happily and serialising them would only cost throughput.
    """
    if not settings.ml_low_memory:
        yield
        return

    acquired = _HEAVY_SLOT.acquire(timeout=timeout if timeout is not None else -1)
    if not acquired:
        raise ModelBusyError(
            f"Cannot load {name}: another model is in use and this machine runs in "
            "low-memory mode (one large model at a time). Try again once processing "
            "has finished."
        )
    try:
        yield
    finally:
        _HEAVY_SLOT.release()


def _evict_other_heavy_models(keeping: str) -> None:
    """Under ``ml_low_memory``, keep at most one large model resident.

    bge (1.3GB) + ViT-H/14 (2.9GB) + moondream2 (4.1GB) is ~8.3GB. That fits in
    VRAM on the target GPU but not in system RAM on a 16GB laptop, where the
    third load dies on the Windows commit limit with an access violation.

    ``cache_clear()`` alone is not enough: it drops the cache's reference, but
    the multi-gigabyte tensors survive until the collector runs, so the next
    model allocates on top of the old one. Hence the explicit ``gc.collect()``.

    This is a memory policy, not device detection -- it is driven by its own
    config flag, so it stays orthogonal to the GPU/CPU seam.
    """
    if not settings.ml_low_memory:
        return

    evicted = []
    for name, loader in _HEAVY_LOADERS.items():
        if name == keeping:
            continue
        if loader.cache_info().currsize:
            loader.cache_clear()
            evicted.append(name)

    if evicted:
        gc.collect()
        empty_device_cache()
        log.info("evicted %s to make room for %s (low-memory mode)", ", ".join(evicted), keeping)


def release_heavy_model(name: str) -> bool:
    """Free one heavy model as soon as its work is finished.

    Eviction alone defers the free to the *next* load, so the memory is handed
    back and asked for again microseconds later. On this machine that is where
    the process dies: with a 19.2GB commit limit and ~5GB free, releasing
    moondream2 (4.1GB) and immediately building ViT-H/14 (2.9GB peak) exceeded
    the limit part-way through construction -- the sidecar vanished with no
    traceback, mid-log-line, straight after "evicted moondream to make room for
    clip".

    Releasing at the end of the pass that needed the model puts minutes between
    the free and the next allocation instead of microseconds, which is enough
    for the OS to reclaim the commit. Costs an ~8s reload per captioned file
    against ~390s of captioning: about 2%.
    """
    if not settings.ml_low_memory:
        return False

    loader = _HEAVY_LOADERS.get(_LOADER_KEYS.get(name, name))
    if loader is None or not loader.cache_info().currsize:
        return False

    loader.cache_clear()
    gc.collect()
    empty_device_cache()
    log.info("released %s after use (low-memory mode)", name)
    return True


def _require_dir(name: str, path: Path, *required_files: str) -> Path:
    if not path.is_dir():
        raise ModelNotAvailableError(name, path)
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        raise ModelNotAvailableError(name, path, f"Missing: {', '.join(missing)}.")
    return path


def resolve_dtype(device: Device) -> torch.dtype:
    """Pick the compute dtype for a device.

    float16 on CPU is a trap: most x86 CPUs have no native fp16 arithmetic, so
    torch emulates it and the model runs several times *slower* than fp32 while
    also risking overflow in attention. On CUDA it is the point of the exercise.
    This is the one device-dependent choice beyond placement itself, which is why
    it lives next to the device switch rather than at the call sites.
    """
    import torch

    return torch.float16 if device == "cuda" else torch.float32


def empty_device_cache(device: Device | None = None) -> None:
    """Release cached VRAM after a large model runs. No-op on CPU."""
    device = device or settings.ml_device
    if device != "cuda":
        return
    import torch

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Text embeddings -- BAAI/bge-large-en-v1.5
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_text_encoder(device: Device | None = None) -> SentenceTransformer:
    """sentence-transformers encoder, loaded from a local directory.

    Passing a filesystem path rather than a repo id is what keeps this offline:
    a repo id sends SentenceTransformer to the hub to resolve the model even
    when the files are already cached.
    """
    device = device or settings.ml_device
    # Not evicting anything: bge stays resident so text search does not wait
    # behind captioning. See the note on _HEAVY_LOADERS.
    path = _require_dir(
        "Text embedding model",
        settings.models_dir / "bge-large-en-v1.5",
        "config.json",
        "modules.json",
    )

    with _BUILD_LOCK:
        import torch
        from sentence_transformers import SentenceTransformer

        log.info("loading text encoder from %s on %s", path, device)
        model_kwargs = {}
        bin_path = path / "pytorch_model.bin"
        if bin_path.exists():
            model_kwargs["use_safetensors"] = False

        try:
            model = SentenceTransformer(str(path), device=device, model_kwargs=model_kwargs)
        except (torch.cuda.OutOfMemoryError, Exception) as exc:
            if str(device).startswith("cuda"):
                log.warning("Failed to load text encoder on %s (%s). Falling back to CPU.", device, exc)
                model = SentenceTransformer(str(path), device="cpu", model_kwargs=model_kwargs)
            else:
                raise
        model.eval()

    log.info("text encoder ready (dim=%d)", model.get_sentence_embedding_dimension())
    return model


# ---------------------------------------------------------------------------
# Image + text embeddings -- OpenCLIP ViT-H/14
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_clip(device: Device | None = None) -> tuple[Any, Any, Any]:
    """Return ``(model, preprocess, tokenizer)`` for OpenCLIP.

    Uses the ``local-dir:`` schema, never ``pretrained=``. The ``pretrained=``
    path resolves through the hub even when the weights sit on disk, which is a
    network call the moment the app is offline.

    The directory must contain ``open_clip_config.json`` (open_clip raises
    ``FileNotFoundError`` without it), a checkpoint, and the tokenizer files --
    when the config comes from a local dir, open_clip always builds its
    tokenizer from that same dir.
    """
    device = device or settings.ml_device
    _evict_other_heavy_models("clip")
    path = _require_dir(
        "OpenCLIP model",
        settings.clip_model_path,
        "open_clip_config.json",
    )

    with _BUILD_LOCK:
        import open_clip

        identifier = f"local-dir:{path}"
        log.info("loading OpenCLIP from %s on %s", identifier, device)

        try:
            model, preprocess = _build_clip_low_memory(open_clip, identifier, path, device)
        except (OSError, MemoryError, Exception) as exc:
            if str(device).startswith("cuda"):
                log.warning("OpenCLIP failed to load on %s (%s). Falling back to CPU...", device, exc)
                try:
                    model, preprocess = _build_clip_low_memory(open_clip, identifier, path, "cpu")
                except Exception as inner:
                    raise ModelNotAvailableError(
                        "OpenCLIP model",
                        path,
                        f"Unable to load OpenCLIP model due to memory limits: {inner}. Text search continues to work normally.",
                    ) from inner
            else:
                raise ModelNotAvailableError(
                    "OpenCLIP model",
                    path,
                    f"Unable to load OpenCLIP model due to memory limits: {exc}. Text search continues to work normally.",
                ) from exc

        model.eval()
        tokenizer = open_clip.get_tokenizer(identifier)

    log.info("OpenCLIP ready")
    return model, preprocess, tokenizer


def _build_clip_low_memory(
    open_clip: Any, identifier: str, path: Path, device: Device
) -> tuple[Any, Any]:
    """Build ViT-H/14 with one allocation instead of two.

    open_clip's default path allocates the whole architecture with random
    weights (~3.9GB for ViT-H/14 in fp32) and *then* reads the checkpoint
    (~3.7GB) before copying across -- about 7.6GB peak for a 3.9GB model. On a
    16GB laptop that exceeds the Windows commit limit and dies in
    ``nn.Linear.__init__`` with an access violation rather than a clean
    MemoryError.

    Building on the meta device costs no memory at all (it records shapes and
    dtypes, allocating no storage), after which each parameter is materialised
    straight from the checkpoint. Peak becomes one copy of the model, and load
    time drops from "crash" to about 15 seconds. This is also strictly better on
    CUDA, where it avoids a redundant host-side copy.
    """
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    checkpoint = _find_clip_checkpoint(path)

    with torch.device("meta"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            identifier, device="meta", load_weights=False
        )

    _check_windows_memory_for_checkpoint(checkpoint)

    with safe_open(str(checkpoint), framework="pt") as handle:
        available = set(handle.keys())
        for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
            if name in available:
                set_module_tensor_to_device(model, name, device, value=handle.get_tensor(name))

    _materialise_non_persistent_buffers(model, device)
    return model, preprocess


def _check_windows_memory_for_checkpoint(checkpoint: Path) -> None:
    """Ensure Windows has sufficient commit charge (paging file) to map the weights file."""
    if sys.platform != "win32":
        return
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    file_size = checkpoint.stat().st_size
    required_bytes = file_size + 300 * 1024 * 1024
    if stat.ullAvailPageFile < required_bytes:
        avail_mb = round(stat.ullAvailPageFile / (1024 * 1024))
        req_mb = round(required_bytes / (1024 * 1024))
        raise ModelNotAvailableError(
            "OpenCLIP model",
            checkpoint.parent,
            f"Insufficient Windows commit memory ({avail_mb}MB available, ~{req_mb}MB required) to map {checkpoint.name}. "
            "Please increase Windows paging file size to enable image search.",
        )


def _find_clip_checkpoint(path: Path) -> Path:
    """Locate the weights file, preferring safetensors as open_clip does."""
    for name in ("open_clip_model.safetensors", "model.safetensors"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise ModelNotAvailableError(
        "OpenCLIP model", path, "No open_clip_model.safetensors checkpoint found."
    )


def _materialise_non_persistent_buffers(model: Any, device: Device) -> None:
    """Rebuild buffers that are computed rather than stored.

    Buffers registered ``persistent=False`` (open_clip's causal ``attn_mask``)
    are deliberately absent from the checkpoint, so a meta build leaves them
    with no storage. Left that way the text tower silently produces wrong
    embeddings -- observed as "a red circle" retrieving the blue square -- with
    no error anywhere. By definition such a buffer is reconstructible from its
    module, and open_clip supplies the constructor.
    """
    import torch

    unresolved: list[str] = []
    for module in model.modules():
        for name, buffer in list(module.named_buffers(recurse=False)):
            if buffer is None or not buffer.is_meta:
                continue

            # Preferred: the module knows how to build it.
            builder = None
            for candidate in (f"build_{name}", "build_causal_mask", "build_attention_mask"):
                attribute = getattr(module, candidate, None)
                if callable(attribute):
                    builder = attribute
                    break
            if builder is not None:
                with torch.no_grad():
                    setattr(module, name, builder().to(device))
                continue

            # `CLIP.attn_mask` is a shared reference to the text tower's mask
            # (open_clip model.py: `register_buffer('attn_mask', text.attn_mask)`),
            # and that tower is not kept as a submodule -- so no builder exists
            # on the object that owns the buffer. A meta tensor still carries
            # its shape, and a causal additive mask is fully determined by it:
            # upper triangle -inf, rest 0. This reproduces open_clip's
            # `build_causal_mask` exactly, without hardcoding the context length.
            if name == "attn_mask" and buffer.dim() == 2 and buffer.shape[0] == buffer.shape[1]:
                with torch.no_grad():
                    mask = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                    mask.fill_(float("-inf"))
                    mask.triu_(1)
                    setattr(module, name, mask)
                continue

            unresolved.append(f"{type(module).__name__}.{name}")

    if unresolved:
        # Refuse rather than run with uninitialised buffers: that path returns
        # confident, wrong embeddings and poisons the search index silently.
        raise ModelNotAvailableError(
            "OpenCLIP model",
            settings.clip_model_path,
            f"Could not rebuild computed buffers {unresolved}; refusing to run with "
            "uninitialised tensors, which would produce silently wrong embeddings.",
        )


# ---------------------------------------------------------------------------
# Captioning -- moondream2 (primary) / BLIP-2 (fallback)
# ---------------------------------------------------------------------------


def _import_from_vendored_dir(package_name: str, directory: Path, module: str) -> Any:
    """Import a module out of a vendored model repo, bypassing the HF cache.

    ``trust_remote_code=True`` copies a repo's ``.py`` files into a *global*
    ``<HF_HOME>/modules/transformers_modules`` tree and imports from there. That
    directory is machine-local, is not what PyInstaller bundles, and is empty on
    a clean install -- so the app works on the dev box and fails on the user's.
    Observed concretely: transformers reported
    ``No such file or directory: .../transformers_modules/moondream2/lora.py``
    even though ``lora.py`` was sitting in the vendored directory.

    The repo's modules import each other *relatively* (``from .config import
    ...``), so the directory has to be registered as a real package rather than
    loaded as a set of standalone files -- a bare
    ``spec_from_file_location`` on one file makes every relative import fail.
    Synthesising a package whose ``__path__`` is the vendored directory makes
    those imports resolve to the shipped files and nothing else.
    """
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    try:
        return importlib.import_module(f"{package_name}.{module}")
    except ImportError as exc:
        raise ModelNotAvailableError(
            package_name, directory, f"Cannot import {module}.py from the vendored repo: {exc}."
        ) from exc


def _moondream_tokenizer_file() -> Path:
    """The tokenizer moondream2 actually needs -- from a separate repo.

    ``moondream.py`` hardcodes
    ``Tokenizer.from_pretrained("moondream/starmie-v1")`` in
    ``MoondreamModel.__init__``: a network call baked into the model repo, so
    vendoring the weights alone does not make the app work offline. With
    ``HF_HUB_OFFLINE=1`` it raises ``OfflineModeIsEnabled``; without it the app
    would quietly phone home on every cold start.

    The obvious fix -- point it at the ``tokenizer.json`` sitting next to the
    weights -- is WRONG and dangerously so. That file is an older revision's
    tokenizer with a 50295-token vocabulary, while the model generates over
    51200. It loads without complaint and produces fluent nonsense
    ("Vasure quant reached Ret sotti AUD allocate..."), which is far worse than
    an error: every image gets a caption, every caption is garbage, and the
    search index quietly fills with noise.

    So the starmie-v1 repo is vendored separately and required here.
    """
    return settings.moondream_tokenizer_path


def _redirect_vendored_tokenizer_to_local(path: Path) -> None:
    """Make the vendored model load its tokenizer from disk, not the hub."""
    from tokenizers import Tokenizer

    tokenizer_file = _moondream_tokenizer_file()
    if not tokenizer_file.exists():
        raise ModelNotAvailableError(
            "moondream2 tokenizer",
            tokenizer_file.parent,
            f"moondream2 needs the {settings.moondream_tokenizer_hub_id} tokenizer, which is a "
            "separate repo from the weights. The tokenizer.json shipped with the weights has the "
            "wrong vocabulary and would produce nonsense captions.",
        )

    module = sys.modules.get("moondream2_vendored.moondream")
    if module is None:
        return

    # A plain shim, not a subclass: `tokenizers.Tokenizer` is a Rust extension
    # type and "is not an acceptable base type". The vendored module uses the
    # name for exactly one call (`Tokenizer.from_pretrained` in
    # `MoondreamModel.__init__`), so forwarding the constructors is enough;
    # anything else still reaches the real class through `__getattr__`.
    class LocalTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, *args: Any, **kwargs: Any) -> Any:
            log.debug("redirecting tokenizer %r to %s", identifier, tokenizer_file)
            return Tokenizer.from_file(str(tokenizer_file))

        @staticmethod
        def from_file(file: str, *args: Any, **kwargs: Any) -> Any:
            return Tokenizer.from_file(file)

        def __getattr__(self, name: str) -> Any:
            return getattr(Tokenizer, name)

    # Rebinding a name on a dynamically imported module; mypy cannot know
    # the vendored module's attributes.
    module.Tokenizer = LocalTokenizer  # type: ignore[attr-defined]


def _verify_tokenizer_covers_vocabulary(model: Any, path: Path) -> None:
    """Fail loudly if the tokenizer cannot decode what the model emits.

    A tokenizer whose vocabulary is smaller than the model's decodes
    out-of-range ids into unrelated words, so the pipeline produces a caption
    for every image and every one is nonsense. Nothing raises, nothing logs,
    and the vector index fills with plausible-looking noise -- which is why
    this is checked at load time rather than trusted.
    """
    tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
    if tokenizer is None:
        return

    try:
        tokenizer_vocab = int(tokenizer.get_vocab_size())
        model_vocab = int(model.model.config.text.vocab_size)
    except (AttributeError, TypeError, ValueError):
        return  # a layout we do not recognise; do not block on a weak check

    if tokenizer_vocab < model_vocab:
        raise ModelNotAvailableError(
            "moondream2 tokenizer",
            path,
            f"Tokenizer vocabulary ({tokenizer_vocab}) is smaller than the model's "
            f"({model_vocab}), so generated captions would decode to nonsense. "
            f"Vendor {settings.moondream_tokenizer_hub_id} with "
            "`python scripts/download_models.py --only moondream --force`.",
        )
    log.debug("tokenizer vocab %d covers model vocab %d", tokenizer_vocab, model_vocab)


@lru_cache(maxsize=1)
def load_moondream(device: Device | None = None) -> tuple[Any, Any]:
    """Return ``(model, tokenizer)`` for moondream2, fully offline.

    Three specific hazards are handled here; see docs/model-vendoring.md:
    the tokenizer is built with ``Tokenizer.from_file`` (never
    ``from_pretrained``, which consults the hub), the model class is imported
    from the vendored directory (never the ``transformers_modules`` cache), and
    the revision is pinned in config.
    """
    device = device or settings.ml_device
    _evict_other_heavy_models("moondream")
    path = _require_dir(
        "moondream2",
        settings.models_dir / "moondream2",
        "config.json",
        "tokenizer.json",
    )

    with _BUILD_LOCK:
        from tokenizers import Tokenizer

        log.info("loading moondream2 from %s on %s", path, device)

        # from_file, not from_pretrained: the latter reaches for the hub even
        # against a local directory.
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        # The concrete class from the vendored repo, NOT AutoModelForCausalLM.
        # Going through Auto* consults `auto_map` in config.json and routes back
        # through the transformers_modules cache -- the exact failure this
        # avoids. config.json names `hf_moondream.HfMoondream`.
        vendored = _import_from_vendored_dir("moondream2_vendored", path, "hf_moondream")
        model_class = vendored.HfMoondream
        _redirect_vendored_tokenizer_to_local(path)

        # "auto" keeps the checkpoint's own dtype (bfloat16 for this repo).
        # Forcing float32 on CPU would double the resident size from ~3.8GB to
        # ~7.7GB for no accuracy gain -- and on a 16GB laptop that is the
        # difference between loading and dying on the Windows commit limit.
        # On CUDA the explicit half-precision choice still applies.
        dtype = resolve_dtype(device) if device == "cuda" else "auto"
        model = model_class.from_pretrained(
            str(path),
            revision=settings.moondream_revision,
            torch_dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        _verify_tokenizer_covers_vocabulary(model, path)

    log.info("moondream2 ready (%s)", dtype)
    return model, tokenizer


@lru_cache(maxsize=1)
def load_blip2(device: Device | None = None) -> tuple[Any, Any]:
    """Return ``(model, processor)`` for the BLIP-2 fallback captioner."""
    device = device or settings.ml_device
    _evict_other_heavy_models("blip2")
    path = _require_dir(
        "BLIP-2",
        settings.models_dir / "blip2-opt-2.7b",
        "config.json",
        "preprocessor_config.json",
    )

    with _BUILD_LOCK:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        log.info("loading BLIP-2 from %s on %s", path, device)
        dtype = resolve_dtype(device)
        processor = Blip2Processor.from_pretrained(str(path), local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            str(path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()

    log.info("BLIP-2 ready (%s)", dtype)
    return model, processor


# ---------------------------------------------------------------------------
# OCR -- PaddleOCR (primary) / Tesseract (CPU fallback)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_paddleocr(device: Device | None = None) -> Any:
    """PaddleOCR with explicit local model directories.

    PaddleX resolves models by *name* against a cache directory and downloads
    what is missing. Passing explicit ``*_model_dir`` paths removes the lookup
    entirely, so there is nothing to download even if the environment variables
    were somehow lost.
    """
    device = device or settings.ml_device
    official = settings.models_dir / "paddlex" / "official_models"
    _require_dir("PaddleOCR models", official)

    detection = official / "PP-OCRv5_server_det"
    recognition = official / "PP-OCRv5_server_rec"
    for name, sub in (("detection", detection), ("recognition", recognition)):
        if not sub.is_dir():
            raise ModelNotAvailableError("PaddleOCR", official, f"Missing {name} model {sub.name}.")

    with _BUILD_LOCK:
        from paddleocr import PaddleOCR

        log.info("loading PaddleOCR from %s on %s", official, device)
        engine = PaddleOCR(
            text_detection_model_dir=str(detection),
            text_recognition_model_dir=str(recognition),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    log.info("PaddleOCR ready")
    return engine


@lru_cache(maxsize=1)
def load_tesseract() -> Any:
    """Return the configured ``pytesseract`` module.

    Tesseract is a native binary, not a Python package, so "loading" it means
    locating the executable and failing early with a message that says how to
    install it -- rather than at the first OCR call, halfway through ingesting a
    folder. No device parameter: Tesseract is CPU-only by nature.
    """
    import pytesseract

    # settings.tesseract_path resolves an explicit override, then the standard
    # install locations, then PATH -- see config.py.
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path

    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # pytesseract raises several unrelated types here
        raise ModelNotAvailableError(
            "Tesseract",
            Path(pytesseract.pytesseract.tesseract_cmd),
            "Tesseract is a separate program, not a Python package. Install it "
            "(`winget install UB-Mannheim.TesseractOCR`) and set FS_TESSERACT_EXE "
            "if it is not on PATH.",
        ) from exc

    log.info("Tesseract ready (v%s)", version)
    return pytesseract


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


# The models big enough that two at once is a memory problem, and therefore the
# ones that get evicted and serialised under low-memory mode.
#
# bge is deliberately NOT in this set even though it is 1.3GB. It is the
# smallest of the four and the only one every single search needs, so evicting
# it would make text search wait behind a captioning pass -- up to ~390s on CPU,
# during which the app looks hung. Keeping it resident costs 1.3GB and means
# text search stays instant while images are being captioned in the background.
# bge + moondream2 is ~5.4GB, which fits; it is CLIP + moondream2 (~7GB) that
# does not.
_HEAVY_LOADERS: dict[str, Any] = {
    "clip": load_clip,
    "moondream": load_moondream,
    "blip2": load_blip2,
}

# `settings.captioner` names an engine; `_HEAVY_LOADERS` keys name a loader.
# They are nearly the same strings, which is exactly why this is written down
# rather than derived -- "moondream2" -> "moondream" but "blip2" -> "blip2",
# so any rule that strips the digit silently frees the wrong model, or none.
_LOADER_KEYS: dict[str, str] = {"moondream2": "moondream", "blip2": "blip2"}

# Everything that can hold significant memory, for reporting.
_ALL_LARGE_LOADERS: dict[str, Any] = {"text": load_text_encoder, **_HEAVY_LOADERS}


def resident_models() -> list[str]:
    """Which large models currently hold memory. For the status view."""
    return [name for name, loader in _ALL_LARGE_LOADERS.items() if loader.cache_info().currsize]


def available_models() -> dict[str, dict[str, Any]]:
    """Report which models are present, without loading any of them.

    Feeds the status view so a user can see what is provisioned before hitting
    a failure mid-ingest.
    """
    checks: dict[str, tuple[Path, tuple[str, ...]]] = {
        "text_embeddings": (
            settings.models_dir / "bge-large-en-v1.5",
            ("config.json", "modules.json"),
        ),
        "clip": (settings.clip_model_path, ("open_clip_config.json",)),
        "moondream2": (
            settings.models_dir / "moondream2",
            ("config.json", "tokenizer.json"),
        ),
        "blip2": (settings.models_dir / "blip2-opt-2.7b", ("config.json",)),
        # PaddleOCR has no single sentinel file -- it needs a detection and a
        # recognition model directory, matching what `load_paddleocr` requires.
        "paddleocr": (
            settings.models_dir / "paddlex" / "official_models",
            ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
        ),
    }

    report: dict[str, dict[str, Any]] = {}
    for name, (path, required) in checks.items():
        if not path.is_dir():
            report[name] = {"path": str(path), "available": False, "missing": list(required)}
            continue

        missing = [f for f in required if not (path / f).exists()]
        # An empty directory is not an installed model. `ensure_dirs` and the
        # download script both create these eagerly, so existence alone would
        # report PaddleOCR as ready when nothing has been fetched.
        empty = not any(path.iterdir())
        if empty and not missing:
            missing = ["(directory is empty)"]

        report[name] = {
            "path": str(path),
            "available": not missing,
            "missing": missing,
        }

    # Tesseract is an executable, so presence is a different question.
    try:
        load_tesseract()
        report["tesseract"] = {"path": "system", "available": True, "missing": []}
    except ModelNotAvailableError as exc:
        report["tesseract"] = {"path": str(exc.path), "available": False, "missing": ["tesseract"]}

    return report


def unload_all() -> None:
    """Drop every cached model and release device memory.

    Used by tests and by a future "free memory" action; not part of normal
    operation, where models stay warm for the life of the process.
    """
    for loader in (
        load_text_encoder,
        load_clip,
        load_moondream,
        load_blip2,
        load_paddleocr,
        load_tesseract,
    ):
        loader.cache_clear()
    gc.collect()
    empty_device_cache()
    log.info("unloaded all models")
