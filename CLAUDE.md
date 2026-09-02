# CLAUDE.md — Local Desktop Multimodal File Search

## Project Context
A **local, single-user Windows desktop application** for context-aware, multimodal file
search. The user points it at their PDFs, DOCX files and images and searches by
*content and meaning* ("what's in this file"), not by filename.

This is **not** a hosted or multi-tenant product. Each install runs entirely on one
person's laptop, against their own files: no server, no accounts, no multi-tenancy,
and **no network dependency after install**.

**Current phase: development and testing.** Build directly against a CUDA GPU — do not
write GPU/CPU auto-detection yet. The only requirement now is that model loading stays
centralised enough that adding a CPU fallback later is a small, localised change.
See `docs/cpu-fallback-plan.md`.

## Tech Stack
| Layer | Choice |
| --- | --- |
| Shell | Tauri (Rust) — hosts the UI, spawns the Python backend as a sidecar, builds one Windows installer. **Phase 6.** |
| Frontend | React + TS + Vite in the Tauri webview, self-hosted `@fontsource/*`, no CDN. **Phase 6.** |
| Frontend (now) | **Streamlit** — temporary dev client for exercising the backend, talks to the same loopback HTTP API |
| Backend | Python 3.11+ / FastAPI, PyInstaller-packaged, binds **127.0.0.1 only** |
| Database | SQLite (one file in the app data dir), SQLAlchemy + Alembic. No Postgres. |
| Vector store | Qdrant **embedded** — `QdrantClient(path=...)`. No server, no Docker. |
| File storage | Local filesystem under the app data dir. No MinIO/S3. |
| Background work | In-process `asyncio` queue + worker. No Celery, no Redis, no broker. |
| Text embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers |
| Image/text embeddings | OpenCLIP ViT-H/14 via `local-dir:` schema |
| Captioning | `vikhyatk/moondream2`, BLIP-2 swappable behind the same interface |
| OCR | PaddleOCR (GPU) primary, Tesseract CPU fallback, one interface |

## Hard Constraints
1. **No network calls at runtime, ever.** The app works with no internet after install.
   All models load from local paths.
2. **Model loading is centralised in `sidecar/app/ml/loaders.py`.** That is the *only*
   place a model is constructed or a device is chosen. Every loader reads the device
   from `settings.ml_device` (default `"cuda"`) rather than hardcoding it.
3. **Offline-safe loading.** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
   `HF_HOME=<models dir>` set in the sidecar environment (see `app/__init__.py` — they
   must be set *before* the HF libraries import). OpenCLIP via
   `open_clip.create_model('local-dir:...')`, never `pretrained=`. Moondream2: vendored
   repo, `Tokenizer.from_file(...)` not `from_pretrained(...)`, model class imported from
   the vendored dir, `revision` pinned. PaddleOCR: pre-populated local model dir plus
   `PADDLE_PDX_MODEL_SOURCE`.
4. **`scripts/download_models.py` runs on the build/dev machine only.** It uses
   `snapshot_download(..., local_dir=..., local_dir_use_symlinks=False)`. The shipped app
   never invokes it at runtime.
5. **No multi-tenant code.** One implicit local user. No auth, no tenant ids, no per-user
   filtering anywhere — it would be dead complexity.
6. **Backend network surface = loopback only.** Tauri passes a local port; the app refuses
   to bind anything but `127.0.0.1` (`Settings.validate_host`).
7. **Verification gate after every phase:** the app must build and run with the network
   disabled, after the one-time model setup. Any outbound connection during normal use is
   a blocker.

## Coding Standards
- **Python:** type hints throughout; `ruff` + `black`; `mypy` clean; `pytest` for ingestion
  and search logic. Every path and the `ml_device` setting come from `config.py` — never
  hardcoded anywhere else.
- **Rust (Tauri):** `main.rs` handles process lifecycle only. No business logic in Rust.
- **Frontend:** TypeScript strict mode, function components + hooks, ESLint + Prettier.
- **No dead multi-tenant/auth scaffolding.** A future multi-user need is a separate
  redesign, not something half-implemented now.
- **Document the CPU-fallback seam** in `docs/cpu-fallback-plan.md` as `ml/loaders.py` is
  built — but do not implement auto-detection in this phase.
- Keep this file under ~200 lines; detail belongs in `docs/`.

## Layout
```
sidecar/app/       config.py (all paths + ml_device) · main.py (FastAPI, loopback only)
  api/ db/ storage/ vectors/ ml/ ingestion/ tasks/ search/
streamlit_app/     dev frontend (temporary)
scripts/           download_models.py (build machine only) · run_backend.ps1 · run_frontend.ps1
docs/              model-vendoring.md · cpu-fallback-plan.md
src-tauri/         phase 6
frontend/          phase 6 (React)
```

## Phases & gates
1. **Shell + sidecar skeleton** — FastAPI on a loopback port, `/health`, frontend confirms
   the round trip. *Gate: launching shows "backend connected" with no manual steps.*
2. **Local storage + ingestion** — SQLite schema (files, chunks, status), filesystem
   manager, upload endpoint extracting PDF/DOCX text and images. *Gate: a dropped PDF
   yields a SQLite row and extracted content on disk; unsupported types fail safely.*
3. **ML loaders + pipeline** — `loaders.py`, bge, OpenCLIP, OCR, captioning, wired into the
   background queue. *Gate: a scanned PDF and a photo produce text embeddings, image
   embeddings, OCR text and a caption — with the network disabled.*
4. **Vector search** — embedded Qdrant, collections, upsert, query endpoint with ranked,
   deduplicated results. *Gate: a natural-language query returns relevant content.*
5. **Frontend** — upload with progress, search bar, results with thumbnail/snippet/caption
   and "open file". *Gate: full flow works with no console errors.*
6. **Packaging** — PyInstaller sidecar, Tauri + React shell, Windows installer, documented
   model provisioning. *Gate: clean VM, network off, ingestion + search work end to end.*

## Dev environment notes
- venv: `C:\Users\shiva\.venvs\fyp` (outside OneDrive on purpose — weights and SQLite must
  not be sync-thrashed).
- App data: `%LOCALAPPDATA%\FileSearch\` (`models/`, `files/`, `derived/`, `qdrant/`,
  `logs/`, `filesearch.db`).
- **This dev box has no CUDA device** (AMD integrated GPU, CPU-only torch). Code defaults
  stay `cuda` / `paddleocr` per spec; the local `.env` sets `FS_ML_DEVICE=cpu` and
  `FS_OCR_ENGINE=tesseract` so it can actually run. Flipping those back is the entire
  change needed on a CUDA machine.
- **It also cannot fit ViT-H/14.** 15.2GB RAM with a 4GB pagefile gives a ~19GB commit
  limit; measured with nothing else running, 5.84GB free fell to 3.38GB after bge and
  building ViT-H/14 killed the process outright — no traceback, mid-log-line, every time.
  Same for moondream2 (4.1GB). The `.env` overrides `FS_CLIP_MODEL_DIR_NAME`,
  `FS_CLIP_MODEL_HUB_ID` and `FS_CLIP_EMBEDDING_DIM` to ViT-B/32 (~600MB, 512-dim);
  code defaults stay the spec model. **Changing the CLIP model invalidates the images
  collection** — the widths differ and the spaces are incomparable regardless, so drop it
  and let `requeue_unindexed` rebuild.

Run: `scripts\run_backend.ps1`, then `scripts\run_frontend.ps1`.

- Streamlit posts usage telemetry to an external webhook by default; `.streamlit/config.toml`
  disables it and `tests/test_no_telemetry.py` fails if that config goes missing.
- Alembic's `fileConfig` wipes the root logger's handlers, so `db/bootstrap.py` opts out with
  `configure_logger=False`; guarded by `tests/test_logging_survives_migrations.py`.
- Only one heavy model fits in RAM here. The inference tests unload between models, and
  `cache_clear()` needs `gc.collect()` to actually free the tensors.

## Status
Phases 1-4 complete. 291 fast tests plus `slow` inference tests against the real weights; ruff and
mypy clean. Phase 5 (full frontend flow) is next.

**Two image switches, not one.** Measured on this CPU: a CLIP image vector costs **2.19s**
per image batched, a moondream2 caption **~390s** — 178x more. One flag for both made a CPU-only
machine give up picture search it could easily afford, so they are separate:
`FS_ENABLE_IMAGE_SEARCH=true` and `FS_ENABLE_CAPTIONING=false` here. OCR is governed by neither
and always runs. Assets keep `clip_embedded=False` until indexed, so flipping either flag picks
the backlog up via `requeue_unindexed` with no re-import. `FS_ENABLE_IMAGE_UNDERSTANDING` is
retired and now raises rather than being ignored — `extra="ignore"` would have dropped it in
silence and re-enabled the 390s half.

**Only standalone image files get CLIP vectors** (`CLIP_INDEXABLE_KINDS`). Pictures *inside* a
document are excluded: OCR already reaches them, so a vector duplicates a working route at real
cost — one 52-page slide deck extracted to 2054 embedded images, 95% of a 79-minute backlog, for
a file that was already searchable. OCR's own coverage (`PROCESSABLE_KINDS`) is unchanged and
still spans page and embedded images. `ml.maintenance.prune_image_vectors` clears vectors written
before the rule narrowed; Qdrant is a separate database, so dropping the SQLite flag alone would
leave them matching queries forever.

### Phase 4 design notes
- **Two collections, never one.** bge and CLIP vectors are both 1024-wide and completely
  incomparable; a nearest-neighbour search across a mixture of them returns noise.
- **Fuse by rank, not score.** bge scores a good match ~0.7, CLIP ~0.25. Comparing them directly
  would rank all text above all images regardless of relevance, so results are combined with
  reciprocal rank fusion (`search/ranking.py`).
- **A file's score is the sum of its best hit per space.** Plain max throws away cross-modal
  corroboration; summing every passage lets long documents win on length alone.
- **Queue priority: index (10) < extract (20) < process (50).** Under plain FIFO, five documents
  sat extracted-but-unsearchable behind one scanned page's captioning. Indexing first made the
  library searchable in 15s instead of 7 minutes.

**A bare noun is not a caption.** CLIP never saw "person" in training, it saw "a photo of a
person in a park". One-word queries score low against *every* image, so the floor deleted the
whole result set — `human` returned 0 of 794. `embed_query` now averages the query over caption
templates (query side only, no re-indexing): person 1→27, human 0→15, vehicle 4→6. Also
`image_relative_score_floor` (0.90), separate from the text one (0.95): CLIP's range is a third
as wide, so one ratio showed 4 of 27 retrieved images.

### Ingestion and UI notes
- **`store_existing_file` copies; `store_upload` moves.** The upload path consumes its source
  (a temp file this app spooled) and deletes it on a duplicate. Doing that to a folder the user
  owns would destroy their documents, so folder scanning has its own path.
  `tests/test_folder_scan.py` pins it.
- **Folder scans are non-recursive by default** and skip `.git`, `node_modules` and the app's
  own data directory.
- Search results are filtered by file kind **in Qdrant**, not after retrieval — post-filtering
  silently shrinks the candidate pool.

### Result quality (learned the hard way on a real library)
- **Exact identifiers need FTS5, not embeddings.** Searching a reference number scored the chunk
  containing it at 0.45 while noise scored 0.57 - inverted, not merely weak. `search/lexical.py`
  is a third retriever fused into the same RRF; exact hits outrank semantic ones and suppress
  non-confident ones beneath them.
- **Never show the RRF score.** It is `1/(k+rank)` — rank 1 is always 0.0164 — so it made an
  excellent match look identical to a barely-passing one. `SearchResult.similarity` carries the
  real cosine; that is what the UI displays.
- **Filter low-information chunks** (`ingestion/quality.py`). Content-free text (patent reference
  lists, figure callouts, a diagram OCR'd as one repeated word) lands near the centre of the
  embedding space and therefore matches *everything*. ~1% of a real library.
- **There is no clean relevance threshold.** Noise ceiling 0.572, signal floor 0.581 — a 0.009 gap.
  Short queries score high against everything. Do not try to tune this away; the app surfaces
  the uncertainty instead (`confident_match_score`).
- **Migrations must be tested against a populated database.** Adding a NOT NULL column without a
  server default passes on an empty table and fails on a real one; `tests/test_migrations.py`
  covers it now.

Phase 3 findings worth remembering:
- **moondream2's tokenizer lives in a different repo.** The `tokenizer.json` beside the weights
  has a 50295-token vocabulary against the model's 51200 and yields fluent nonsense with no
  error at all. `moondream/starmie-v1` is vendored separately and the vocabulary is verified
  at load time.
- **Large models need one allocation, not two.** ViT-H/14 loaded the default way needs ~7.6GB
  peak for a 3.9GB model and dies on the Windows commit limit. `_build_clip_low_memory` builds
  on the meta device and materialises from the checkpoint (~2.9GB peak). Buffers registered
  `persistent=False` must be rebuilt afterwards or the text tower silently returns wrong
  embeddings.
- **A missing model is not a failed file.** It leaves the file in `extracted` so
  `requeue_unprocessed` picks it up once the model is installed, and the already-extracted text
  stays searchable in the meantime.
