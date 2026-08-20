# CPU fallback plan (not implemented)

Device auto-detection is deliberately **out of scope** for the current phase.
This note records exactly where it would go, so adding it later is a localised
change rather than a refactor.

## The seam

There is exactly one device value in the system:

```
sidecar/app/config.py :: Settings.ml_device   # Literal["cuda", "cpu"], default "cuda"
```

It is read in exactly one module:

```
sidecar/app/ml/loaders.py
```

Every loader in `loaders.py` takes the device as a parameter with
`settings.ml_device` as its default, and every model-owning module
(`embeddings_text.py`, `embeddings_clip.py`, `captioning.py`, `ocr.py`) obtains
its model *only* by calling a loader. No other file constructs a model, calls
`.to(device)`, or mentions `"cuda"` / `"cpu"`.

Enforced by test: `tests/test_no_hardcoded_device.py` fails if a device literal
appears outside `config.py` and `loaders.py`.

## What auto-detection would add

Replace the static default with a resolver, in `config.py` only:

```python
@computed_field
@property
def resolved_device(self) -> Device:
    if self.ml_device != "auto":
        return self.ml_device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
```

Then widen the `Device` literal to include `"auto"` and have `loaders.py` read
`settings.resolved_device`. Nothing else changes.

The reason this is deferred rather than done now: importing `torch` inside
`config.py` makes configuration import-heavy and couples settings to the ML
stack, which matters for the PyInstaller build (phase 6) and for tests that
should not pay a multi-second torch import. That trade-off is worth making once,
at deployment time, with the real target hardware in hand.

## Consequences that are NOT just a device string

Switching to CPU changes behaviour beyond placement, and a real fallback has to
handle these:

| Concern | GPU path | CPU path |
| --- | --- | --- |
| dtype | `float16` for captioning is fine | `float16` is slow/unsupported on most CPUs — must use `float32`, or keep the checkpoint's `bfloat16` |
| Batch size | large text/image batches | must shrink or the process pages to disk |
| OCR engine | PaddleOCR (GPU build) | Tesseract — PaddleOCR's CPU build is a separate wheel and much slower |
| Captioning latency | ~1s/image | **~390s/image measured** for moondream2 — the UI must show progress, not block |
| Peak load memory | VRAM, and the host copy is transient | the binding constraint — see below |
| `torch.cuda.empty_cache()` | meaningful | no-op, must be guarded |

`loaders.py` already selects dtype from the device, so the dtype row is handled.
Batch sizes and the OCR engine are separate config values
(`Settings.ocr_engine`) rather than device-derived, so a future resolver should
set them together.

## Measured on the CPU dev box (16 GB RAM, no CUDA)

These are not projections; they are what this hardware actually did.

| Model | Load time | Resident | Notes |
| --- | --- | --- | --- |
| bge-large-en-v1.5 | ~23 s | ~1.3 GB | fine |
| OpenCLIP ViT-H/14 | ~16 s | ~2.9 GB | only after the meta-device rewrite; the default path crashed |
| moondream2 | ~30 s | ~4.1 GB | bfloat16 from the checkpoint |
| Tesseract OCR | instant | negligible | 0.96 mean confidence on rendered text |

Two findings that matter for the eventual CPU story:

1. **Peak memory at load, not steady state, is what kills you.** Building a
   model with random weights and then reading the checkpoint needs two copies.
   For ViT-H/14 that is ~7.6 GB for a 3.9 GB model, which exceeded the Windows
   commit limit and produced an access violation rather than a `MemoryError`.
   The meta-device load in `_build_clip_low_memory` fixed this and helps on GPU
   too. Any future model added to the pipeline should be loaded the same way.

2. **Only one heavy model fits at a time.** bge + ViT-H/14 + moondream2 is
   ~8.3 GB resident, which does not co-exist with everything else on a 16 GB
   machine. The test suite works around this with an autouse fixture that
   unloads the models a test does not need (`tests/test_ml_inference.py`), and
   `cache_clear()` alone is not enough — `gc.collect()` is required or the
   tensors are still alive when the next model allocates.

   **The production pipeline does not do this yet.** On CPU hardware it would
   need an eviction policy in `loaders.py` — load-on-demand with an LRU of one
   heavy model — paid for with a reload between stages. On the target CUDA box
   the question does not arise, which is why it is not implemented.

3. **Captioning is impractical on CPU at this model size.** 390 s per image
   means a 50-image PDF is five hours. A CPU fallback that keeps moondream2
   would need a smaller captioner; the `Captioner` interface already allows
   that swap without touching the pipeline.

## Current dev-box status

The machine this was built on has an AMD integrated GPU and a CPU-only torch
build, so `.env` sets `FS_ML_DEVICE=cpu` and `FS_OCR_ENGINE=tesseract`. The
code's defaults remain `cuda` / `paddleocr` per spec; only the local `.env`
differs.
