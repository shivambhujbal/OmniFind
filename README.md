# OmniFind — Local File Search

Search your own PDFs, DOCX files and images by **what is in them**, not by filename. Everything runs locally on your machine: no external server, no account, no internet connection needed after initial setup. 100% private and air-gapped.

See [CLAUDE.md](CLAUDE.md) for the architecture and constraints, [docs/model-vendoring.md](docs/model-vendoring.md) for how models are made offline-safe, and [docs/cpu-fallback-plan.md](docs/cpu-fallback-plan.md) for the GPU/CPU seam.

---

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Backend skeleton, `/health`, frontend round trip | **done** |
| 2 | SQLite schema, filesystem storage, PDF/DOCX/image ingestion | **done** |
| 3 | ML loaders, embeddings, OCR, captioning, background queue | **done** |
| 4 | Embedded vector store (ChromaDB / Qdrant), vector search endpoint | **done** |
| 5 | Modern React + TypeScript + Vite Frontend (Stitch Design System) | **done** |
| 6 | PyInstaller + Tauri shell, Windows installer | next |

---

## Frontend Architecture

The project now includes a **modern React 19 + TypeScript + Vite** single-page application located in `frontend/`, styled with a custom light-mode design system:

- **Hero Search**: Minimalist centered search interface with global `Ctrl + K` keyboard shortcut.
- **Search Results View**: Structured result cards with match relevance percentage, highlighted text snippets, image thumbnails, and metadata breadcrumbs.
- **Indexing Status**: Live background queue tracking, progress bar, document/image processed counters, and local folder scanner.
- **Right-Hand Sidebar**: Clean navigation panel for Home, Search History, Indexed Folders, Indexing Status, Settings, and About.
- **System & About Modals**: Live backend inspection of ML hardware device (`cuda` or `cpu`), Python runtime, offline environment verification, and storage breakdown.

The original Streamlit app remains available at `streamlit_app/` as a reference dev harness. Both frontends communicate over the same loopback HTTP API (`http://127.0.0.1:8756`).

---

### Images on a CPU

Two switches, because the two halves of "image understanding" differ in cost by **178x** on CPU:

| | Cost per image, this CPU | Setting |
| --- | --- | --- |
| Match pictures by appearance (CLIP vector) | **2.19 s** | `FS_ENABLE_IMAGE_SEARCH=true` |
| Describe each picture (moondream2 caption) | **~390 s** | `FS_ENABLE_CAPTIONING=false` |

Image *search* runs efficiently on CPU; captioning is disabled by default on CPU. On a GPU, both take about a second. Files already ingested are picked up automatically when either flag changes without re-importing.

**OCR is governed by neither.** Reading text out of a scan takes ~1s with Tesseract, so scanned documents are fully searchable regardless.

---

## Setup

### 1. Python Environment (One-time)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\pip install -r requirements-dev.txt
```

### 2. Vendored Offline Models (~9 GB)

```powershell
python scripts/download_models.py
python scripts/download_models.py --verify
```

### 3. OCR Engine (Tesseract)

```powershell
winget install UB-Mannheim.TesseractOCR
```

### 4. Frontend Node Dependencies

```powershell
cd frontend
npm install
cd ..
```

---

## Running

Launch the backend and frontend in separate PowerShell terminals:

### 1. Start the Backend API (FastAPI sidecar)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
```

Runs on `http://127.0.0.1:8756`.

### 2. Start the React Frontend (Vite dev server)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_react_frontend.ps1
```

Opens at `http://localhost:5173`. Proxies `/api/*` requests directly to the FastAPI sidecar.

*(Optional) Legacy Streamlit frontend:*
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1
```

---

## Testing & Quality Checks

```powershell
# Python unit tests
.\.venv\Scripts\python -m pytest -m "not slow"

# Python linting and type checking
.\.venv\Scripts\python -m ruff check . ; .\.venv\Scripts\python -m mypy sidecar

# Frontend TypeScript check
cd frontend ; npx tsc --noEmit ; cd ..
```

---

## Where Things Live

All application data is stored locally in `%LOCALAPPDATA%\FileSearch\`:

```
models/          vendored weights (~9 GB)
files/           originals of ingested files
derived/         extracted page images, thumbnails
qdrant/          embedded vector store
logs/            sidecar.log
filesearch.db    SQLite database
```
