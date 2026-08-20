# Claude Code Prompt — Local Desktop Multimodal File Search App

> **Instructions to Claude Code:** Build the project described here from scratch. Treat "Hard Constraints" as inviolable rules for every file you write in every phase. First create the repo skeleton and a `CLAUDE.md` with the "Project Context", "Tech Stack", "Hard Constraints", and "Coding Standards" sections below, then proceed phase by phase, running the verification gate after each phase.

## 1. Project Overview & Goals
A **local, single-user Windows desktop application** for context-aware, multimodal file search: the user points it at their PDFs, DOCX files, and images, and searches by *content/meaning* ("what's in this file") rather than filename.

This is **not** a hosted/multi-tenant product. Each install runs entirely on one person's laptop, against their own files, with **no server, no accounts, no multi-tenancy, and no network dependency after install**.

**Current phase: development and testing.** Assume a CUDA-capable GPU is available in the dev environment and build against it directly — do not build GPU/CPU auto-detection logic yet. That decision is deferred to deployment; the only requirement now is that model loading is centralized enough that adding a CPU fallback later is a small, localized change, not a rewrite.

## 2. Tech Stack
- **Shell:** Tauri (Rust) — hosts the UI, spawns and manages the Python backend as a **sidecar process**, produces a single Windows installer (MSI/NSIS).
- **Frontend:** React + TypeScript + Vite, rendered inside the Tauri webview. Self-hosted fonts/icons (`@fontsource/*`), no CDN dependencies.
- **Backend:** Python 3.11 + FastAPI, compiled to a standalone executable via **PyInstaller**, launched by Tauri on app start and terminated on app close. Binds **only to `127.0.0.1`** on a local port — never exposed to the network.
- **Database:** SQLite (single file under the app's local data directory). SQLAlchemy + Alembic migrations. No Postgres.
- **Vector store:** Qdrant **embedded mode** — `QdrantClient(path="<local-dir>")`, running in-process against local files. No Qdrant server, no Docker.
- **File storage:** local filesystem, under an app-managed data directory. No MinIO/S3.
- **Background processing:** in-process async task handling (FastAPI `BackgroundTasks` or an internal `asyncio` queue + worker loop). No Celery, no Redis, no external broker — there's exactly one process, one user, one machine.
- **ML models (GPU, for this dev phase):**
  - Text embeddings: `BAAI/bge-large-en-v1.5` via `sentence-transformers`, `device="cuda"`.
  - Image/text embeddings: OpenCLIP **ViT-H/14** (open_clip ≥3.0.0), loaded via `local-dir:` schema, `device="cuda"`.
  - Image captioning: `vikhyatk/moondream2` (transformers ≥4.53) on CUDA, with BLIP-2 as a swappable fallback behind the same interface.
  - OCR: PaddleOCR (GPU build) as primary, Tesseract as a CPU-only fallback engine (already CPU-only by nature — no change needed there).

## 3. Hard Constraints
1. **No network calls at runtime, period.** The app must work with no internet connection after installation. All models load from local paths only.
2. **Centralize model loading** in `sidecar/app/ml/loaders.py`. This is the *only* place that constructs a model or picks a device. Read the device from a single config value (`settings.ml_device`, defaulting to `"cuda"` for this dev phase) — every loader function takes/uses that value rather than hardcoding `"cuda"` inline. This is what makes a future CPU-fallback change small.
3. **Offline-safe model loading**, same as previously established:
   - Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=<local models dir>` in the sidecar process environment.
   - OpenCLIP loads via `open_clip.create_model('local-dir:/models/openclip-vit-h14', device=settings.ml_device)` — never the hub-resolving `pretrained=` path.
   - Moondream2: vendor the full repo including `tokenizer.json`; load the tokenizer via `Tokenizer.from_file(...)`, never `from_pretrained(...)`; import the model class directly from the vendored directory to avoid the `transformers_modules` cache issue; pin the `revision`.
   - PaddleOCR: pre-populate `~/.paddlex/official_models` (or the equivalent local dir bundled with the app) at build time; set `PADDLE_PDX_MODEL_SOURCE` and explicit model dir paths so it never attempts a network lookup.
4. **`scripts/download_models.py` runs once, on the build/dev machine only** (or as a first-run "setup" step with clear UI feedback if you choose to download on first launch instead of bundling). It populates the local models directory using `snapshot_download(..., local_dir=..., local_dir_use_symlinks=False)`. This script is never invoked by the shipped app at runtime.
5. **No multi-tenant code.** There is one implicit local user. Do not add auth, tenant IDs, or per-user filtering anywhere — it doesn't apply here and would just be dead complexity.
6. **Backend network surface = localhost only.** Tauri passes the sidecar a local port; the FastAPI app must refuse to bind to `0.0.0.0`.
7. **Verification gate (run after every phase):** the app must build and run correctly with the machine's Wi-Fi/network disabled (or via a firewall rule blocking the sidecar process) after the initial model setup step has completed. Any attempted outbound connection during normal use is a blocker.

## 4. Required Project Structure
```
project-root/
├── CLAUDE.md
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json          # sidecar registration, window config
│   └── src/main.rs              # spawns/monitors the Python sidecar, passes local port
├── sidecar/
│   ├── pyinstaller.spec
│   ├── requirements.txt
│   └── app/
│       ├── main.py              # FastAPI app; binds 127.0.0.1:<port> only
│       ├── config.py            # settings incl. ml_device, local data/model paths
│       ├── db/                  # SQLite models, session, Alembic
│       ├── storage/             # local filesystem file manager
│       ├── vectors/             # Qdrant embedded client wrapper
│       ├── ml/
│       │   ├── loaders.py       # SINGLE place models are constructed; reads settings.ml_device
│       │   ├── embeddings_text.py
│       │   ├── embeddings_clip.py
│       │   ├── captioning.py    # Moondream2 + BLIP-2 behind one interface
│       │   └── ocr.py           # PaddleOCR + Tesseract behind one interface
│       ├── ingestion/           # PDF/DOCX/image extraction, chunking
│       ├── tasks/               # in-process background queue (asyncio-based)
│       └── search/              # query embedding + Qdrant search + ranking
├── frontend/
│   ├── package.json / package-lock.json
│   └── src/                     # upload UI, search UI, results view
├── scripts/
│   └── download_models.py       # dev/build-machine only
└── docs/
    ├── model-vendoring.md
    └── cpu-fallback-plan.md     # notes for the future GPU/CPU auto-detect work, not implemented yet
```

## 5. Phased Implementation Plan
**Phase 1 — Shell + sidecar skeleton.** Tauri app that launches a minimal FastAPI sidecar on a local port, with a `/health` endpoint the React UI calls on startup to confirm the round trip works. Sidecar lifecycle (start on launch, clean shutdown on quit) fully wired. **Gate:** launching the Tauri app shows a "backend connected" state with no manual steps.

**Phase 2 — Local storage + ingestion.** SQLite schema (files, chunks, processing status) via SQLAlchemy/Alembic. Local filesystem file manager. Upload endpoint that stores the original file, extracts PDF/DOCX text and images, and creates DB rows. **Gate:** dropping a PDF into the app results in a row in SQLite and extracted content on disk, no ingestion of unsupported types breaks anything.

**Phase 3 — ML loaders + processing pipeline.** Implement `ml/loaders.py` with `settings.ml_device="cuda"`, then bge embeddings, OpenCLIP embeddings, PaddleOCR/Tesseract, and Moondream2/BLIP-2 captioning, each wired into the in-process background task queue triggered after ingestion. **Gate:** uploading a scanned PDF and a photo produces text embeddings, image embeddings, OCR text, and a caption, all visible in SQLite/logs — with network disabled after the one-time model setup.

**Phase 4 — Vector search.** Qdrant embedded client (`path=` based), collection setup, upsert from Phase 3 outputs, and a search endpoint that embeds the query and returns ranked, deduplicated results with source file references. **Gate:** a natural-language query returns relevant chunks/images from ingested files.

**Phase 5 — React frontend.** Drag-and-drop upload with progress, a search bar, and a results view (thumbnail/snippet/caption + "open file" action). Self-hosted fonts/icons only, rendered correctly inside the Tauri webview. **Gate:** full flow — add files, search, open a result — works with no DevTools errors.

**Phase 6 — Packaging.** PyInstaller build of the sidecar, Tauri bundling into a Windows installer, and a documented model-provisioning approach (bundled in the installer vs. first-run download — pick one and implement it cleanly). Write `docs/model-vendoring.md`. **Gate:** install on a clean Windows machine/VM, disconnect from the network, launch, and confirm ingestion + search work end to end.

## 6. Coding Standards
- **Python:** type hints throughout; `ruff` + `black`; `mypy` clean; `pytest` for ingestion/search logic. All paths and the `ml_device` setting come from `config.py`, never hardcoded.
- **Rust (Tauri):** keep `main.rs` focused on process lifecycle only — no business logic in Rust.
- **Frontend:** TypeScript strict mode, functional components/hooks, ESLint + Prettier.
- **No dead multi-tenant/auth scaffolding** — if a future need for multi-user support arises, that's a separate redesign, not something to half-implement now.
- **Document the CPU-fallback seam** in `docs/cpu-fallback-plan.md` as you build `ml/loaders.py` — a short note on exactly where device selection would change — but do not implement device auto-detection in this phase.
- Keep `CLAUDE.md` under ~200 lines; push detail into `docs/`.
