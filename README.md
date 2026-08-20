# Local File Search

Search your own PDFs, DOCX files and images by **what is in them**, not by
filename. Everything runs on your machine: no server, no account, no internet
connection needed after setup.

See [CLAUDE.md](CLAUDE.md) for the architecture and constraints,
[docs/model-vendoring.md](docs/model-vendoring.md) for how models are made
offline-safe, and [docs/cpu-fallback-plan.md](docs/cpu-fallback-plan.md) for the
GPU/CPU seam.

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Backend skeleton, `/health`, frontend round trip | **done** |
| 2 | SQLite schema, filesystem storage, PDF/DOCX/image ingestion | **done** |
| 3 | ML loaders, embeddings, OCR, captioning, background queue | **done** |
| 4 | Embedded Qdrant, vector search endpoint | **done** |
| 5 | Full frontend flow | next |
| 6 | PyInstaller + Tauri + React, Windows installer | |

The frontend is **Streamlit for now** — a stand-in that exercises the same
loopback HTTP API the Tauri + React shell will use in phase 6.

### Documents-only mode

Image *understanding* — captioning pictures and searching them by appearance —
is switched **off** on this machine (`FS_ENABLE_IMAGE_UNDERSTANDING=false`).
Captioning one image with moondream2 costs about **390 seconds** on this CPU,
which makes an image-heavy document impractical to ingest. It is a one-line
change to switch back on once the project moves to a GPU machine, and files
already ingested are picked up automatically — no re-import.

**OCR is not affected.** Reading text out of a scan takes about a second with
Tesseract, so scanned documents are still fully searchable; the sample library
below finds a scanned invoice by its total.

## Setup

One-time, on a machine with internet:

```bash
python -m venv C:\Users\you\.venvs\fyp
```

```bash
C:\Users\you\.venvs\fyp\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
C:\Users\you\.venvs\fyp\Scripts\pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and adjust. Then vendor the models (~9 GB, build
machine only — the app never does this itself):

```bash
python scripts/download_models.py
```

```bash
python scripts/download_models.py --verify
```

OCR needs **Tesseract**, which is a separate program rather than a Python
package:

```bash
winget install UB-Mannheim.TesseractOCR
```

`config.py` finds it in the standard install locations automatically; set
`FS_TESSERACT_EXE` if yours is somewhere else.

## Running

Two terminals. Backend:

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
```

Frontend:

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1
```

The Streamlit sidebar shows **Backend connected** once the round trip works.

## Using it

**Library tab** — paste a folder path and press *Index folder*. The app reports
how many supported files it found before you commit, then shows a progress bar
while the queue drains. Subfolders are excluded by default: pointing at
`Documents` and descending can pull in tens of thousands of files.

Your originals are **copied** into the app's store and never moved, renamed or
deleted — re-indexing the same folder is safe and reports what was already
known rather than adding it twice. Individual file upload is still there, under
*Or upload individual files*.

**Search tab** — describe what you want in your own words. Filter with
*Everything / Documents / Images*, and page through results ten at a time.
Each result shows how strong the match is; when even the best hit is weak the
app says so rather than presenting it as an answer.

## Checks

```bash
C:\Users\you\.venvs\fyp\Scripts\python -m pytest -m "not slow"
```

The `slow` tests run real inference against the vendored weights and skip
automatically when the weights are absent. They take several minutes -- CPU
captioning is roughly 390s per image -- so they are excluded above. To run
them:

```bash
pytest -m slow
```

```bash
C:\Users\you\.venvs\fyp\Scripts\python -m ruff check . ; C:\Users\you\.venvs\fyp\Scripts\python -m mypy sidecar
```

`tests/test_constraints.py` is the executable form of the hard constraints: it
fails if a device string leaks outside `config.py`/`loaders.py`, if a model is
constructed outside `ml/loaders.py`, if multi-tenant scaffolding appears, or if
the backend can be bound to a non-loopback address.

## Where things live

Nothing large is stored in this repo. The app data directory defaults to
`%LOCALAPPDATA%\FileSearch\`:

```
models/          vendored weights (~9 GB)
files/           originals of ingested files
derived/         extracted page images, thumbnails
qdrant/          embedded vector store
logs/            sidecar.log
filesearch.db    SQLite
```

Keep it off OneDrive/Dropbox — syncing a live SQLite file and gigabytes of
weights is slow and can corrupt the database.
