# Model vendoring

The app must work with networking disabled (hard constraint 1). That means every
model is a **local directory of real files**, resolved by path, with the ML
libraries put into offline mode before they are imported. This document records
how each model is fetched, what it must contain, and the specific failure mode
each measure prevents.

## Where models live

`<data_dir>/models`, where `data_dir` defaults to `%LOCALAPPDATA%\FileSearch`
(`Settings.data_dir`, override with `FS_DATA_DIR`). The directory doubles as
`HF_HOME`, so any incidental Hugging Face cache lands inside the same tree and
ships or gets wiped as one unit.

```
models/
├── bge-large-en-v1.5/          ~1.3 GB
├── openclip-vit-h14/           ~3.8 GB
├── moondream2/                 ~3.7 GB
├── moondream2-tokenizer/       ~4 MB   (moondream/starmie-v1 -- see below)
├── blip2-opt-2.7b/             ~8 GB   (optional fallback, --only blip2)
└── paddlex/official_models/    ~0.2 GB
```

## Fetching

`scripts/download_models.py` runs **on the build/dev machine only** and is the
only code in the repo permitted to make network calls. The shipped app never
invokes it.

```bash
python scripts/download_models.py            # text + clip + moondream + paddle
python scripts/download_models.py --verify   # report what is present, fetch nothing
python scripts/download_models.py --only blip2
```

It uses `snapshot_download(..., local_dir=...)`. Passing `local_dir` is what
matters: it writes real files rather than the blob-cache-plus-symlinks layout,
which does not survive being copied into an installer and needs developer mode
or admin rights to create on Windows at all. (The old explicit
`local_dir_use_symlinks=False` argument is deprecated and ignored in
huggingface-hub >= 0.26 — real files are now the only behaviour — so it is not
passed.)

The script sets `HF_HUB_OFFLINE=0` **before** importing `app.config`. This works
because `Settings.apply_offline_env` uses `os.environ.setdefault` — an explicit
value set first wins. That is the whole mechanism for having one repo where
runtime is offline and the vendoring script is not.

## Offline environment

Set in `sidecar/app/__init__.py`, i.e. at package import, not in `main.py`:

| Variable | Value | Prevents |
| --- | --- | --- |
| `HF_HOME` | `<models dir>` | writes to `~/.cache/huggingface` that don't ship |
| `HF_HUB_OFFLINE` | `1` | hub revision checks on every `from_pretrained` |
| `TRANSFORMERS_OFFLINE` | `1` | the same check inside transformers |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | outbound telemetry pings |
| `SENTENCE_TRANSFORMERS_HOME` | `<models dir>` | a second cache tree outside the app |
| `PADDLE_PDX_MODEL_SOURCE` | `local` | PaddleX resolving models from BOS |
| `PADDLE_PDX_CACHE_HOME` | `<models dir>/paddlex` | writes to `~/.paddlex` |

**These must be set before `huggingface_hub` / `transformers` are imported.**
Both libraries read them once, at their own import time; setting them afterwards
silently does nothing and you get a network call in production. Putting them in
`app/__init__.py` means any `import app.<anything>` has already applied them.

## Per-model notes

### bge-large-en-v1.5 (text embeddings)

Loaded with `SentenceTransformer(str(path), device=...)` — a filesystem path, not
a repo id, so no hub resolution happens. The repo also ships ONNX and OpenVINO
exports of the same weights; `ignore_patterns` drops them, otherwise the download
is roughly triple the size for no benefit.

Queries are prefixed with `Represent this sentence for searching relevant
passages: ` (`Settings.text_query_prefix`) — this is how the model was trained
for asymmetric retrieval. Documents are embedded **without** the prefix. Getting
this backwards degrades recall noticeably.

### OpenCLIP ViT-H/14 (image + text embeddings)

```python
open_clip.create_model("local-dir:" + str(settings.clip_model_path), device=settings.ml_device)
```

Never the `pretrained=` argument: that path resolves through the hub even when
the weights are already cached locally.

The `local-dir:` schema (open_clip >= 3.0) requires the directory to contain:

- `open_clip_config.json` — mandatory; without it the call raises `FileNotFoundError`
- a checkpoint, preferring `open_clip_model.safetensors`
- tokenizer files (`tokenizer.json` et al.) — when the config comes from a local
  dir, open_clip always builds `HFTokenizer` from that same dir

The `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` repo publishes the same weights three
times (HF-format `pytorch_model.bin`, `open_clip_pytorch_model.bin`, and
`open_clip_model.safetensors`), roughly 4 GB each. The `allow_patterns` list in
the spec fetches only the safetensors copy plus config and tokenizer.

#### Loading it without two copies in memory

open_clip's default path allocates the whole architecture with random weights
(~3.9 GB for ViT-H/14 in fp32) and *then* reads the 3.7 GB checkpoint before
copying across — roughly **7.6 GB peak for a 3.9 GB model**. On a 16 GB laptop
that exceeds the Windows commit limit and the process dies inside
`nn.Linear.__init__` with an access violation (`0xC0000005`), not a clean
`MemoryError`. Increasing the pagefile also fixes it, but that is the user's
machine to configure, and the waste is real on every machine.

`_build_clip_low_memory` builds the model on the **meta device** — shapes and
dtypes, no storage at all — and then materialises each parameter directly from
the checkpoint with `accelerate.utils.set_module_tensor_to_device`. Peak drops
to one copy (~2.9 GB measured), and load time goes from "crash" to ~16 s. This
is strictly better on CUDA too, where it avoids a redundant host-side copy.

**The catch: non-persistent buffers.** Buffers registered `persistent=False`
are absent from the checkpoint by design, so a meta build leaves them with no
storage. open_clip's causal `attn_mask` is one. Left uninitialised the text
tower returns confident, wrong embeddings — observed as *"a red circle"*
retrieving the blue square — with nothing raised or logged.

`_materialise_non_persistent_buffers` handles this: it asks the owning module to
rebuild the buffer (`build_causal_mask` / `build_attention_mask`), and for
`CLIP.attn_mask` — which is a shared reference to the text tower's mask, on a
module that has no builder — it reconstructs the causal mask from the meta
tensor's own shape. Anything it cannot rebuild is a hard failure rather than a
silent one.

### moondream2 (captioning, primary)

Four footguns, all hit in practice and all handled in `ml/loaders.py`.

1. **The tokenizer is in a different repo, and the bundled one is wrong.**
   `moondream.py` hardcodes `Tokenizer.from_pretrained("moondream/starmie-v1")`
   inside `MoondreamModel.__init__` — a network call baked into the model repo,
   so vendoring the weights alone does not make the app offline-capable.

   The obvious fix, pointing it at the `tokenizer.json` sitting beside the
   weights, is wrong and **dangerously quiet**: that file is an older
   revision's tokenizer with a **50295**-token vocabulary, while the model
   generates over **51200**. Nothing errors. Every image gets a caption and
   every caption is fluent nonsense:

   > `Vasure quant reached Ret sotti AUD allocatewikiA remarkably surdue colossal65ippyitude`

   That is worse than a crash — the search index fills with plausible-looking
   noise. So `moondream/starmie-v1` is vendored separately
   (`Settings.moondream_tokenizer_hub_id`, ~4 MB) and
   `_verify_tokenizer_covers_vocabulary` refuses to load if the tokenizer
   vocabulary is smaller than the model's.

2. **`transformers_modules` cache.** `trust_remote_code=True` copies the repo's
   `.py` files into a global `<HF_HOME>/modules/transformers_modules` tree and
   imports from *there*. That cache is machine-local, is not what PyInstaller
   bundles, and is empty on a clean install. Observed failure:
   `No such file or directory: .../transformers_modules/moondream2/lora.py`,
   with `lora.py` sitting in the vendored directory the whole time.

   The loader imports the concrete class (`hf_moondream.HfMoondream`, named in
   `config.json`'s `auto_map`) straight from the vendored directory. Note the
   repo's modules import each other **relatively** (`from .config import ...`),
   so a bare `spec_from_file_location` on one file is not enough — the
   directory is registered as a package whose `__path__` points at it.
   Going through `AutoModelForCausalLM` would consult `auto_map` and route back
   into the cache, so the concrete class is used instead.

3. **Pinned revision.** `Settings.moondream_revision` (currently `2025-06-21`).
   This repo changes its module layout between releases; an unpinned fetch
   produces a directory the loader cannot import.

4. **dtype.** The checkpoint is bfloat16. On CPU the loader passes
   `torch_dtype="auto"` to keep it that way; forcing float32 would double the
   resident size from ~3.8 GB to ~7.7 GB for no accuracy gain. On CUDA the
   explicit half-precision choice from `resolve_dtype` still applies.

### BLIP-2 (captioning, fallback)

Standard `Blip2Processor` / `Blip2ForConditionalGeneration` from a local path.
Sits behind the same `Captioner` protocol as moondream2, selected by
`Settings.captioner`. Not downloaded by default — it is another ~8 GB and only
matters if moondream2 misbehaves on the target hardware.

### PaddleOCR

PaddleOCR/PaddleX downloads detection and recognition models on first use unless
the official-models directory is pre-populated. The script fills
`<models dir>/paddlex/official_models/`, and `PADDLE_PDX_MODEL_SOURCE=local`
plus `PADDLE_PDX_CACHE_HOME` point the library there. The loader additionally
passes explicit `*_model_dir` arguments so no name-based lookup happens at all.

Tesseract is the CPU fallback engine, selected with `FS_OCR_ENGINE=tesseract`.
It is a native binary rather than a Python package: install it separately and
point `FS_TESSERACT_EXE` at the executable if it is not on `PATH`.

## Provisioning the installer (phase 6)

Two options; **bundling** is the plan of record:

| | Bundled in installer | First-run download |
| --- | --- | --- |
| Installer size | ~9 GB | ~80 MB |
| First launch | works immediately, offline | needs internet once, ~9 GB, needs UI + resume + failure handling |
| Constraint 1 | satisfied outright | satisfied only after a successful first run |

Bundling keeps the no-network guarantee unconditional and removes an entire
class of first-run failure states. The cost is installer size, which is
acceptable for a single-user desktop tool distributed directly.

## Verifying the guarantee

The phase gate, after models are in place:

1. Disable Wi-Fi, or add an outbound firewall rule blocking the sidecar process.
2. Start the backend and ingest a PDF and a photo.
3. Run a search.

Any outbound connection attempt is a blocker. To see them rather than infer
them, run the sidecar under a network monitor, or temporarily set
`HF_HUB_OFFLINE=0` and watch for hub requests in the logs — with the vendoring
above, there should be none either way.
