"""Central configuration for the file-search sidecar.

Every path, device choice and model identifier used anywhere in the backend is
defined here. Nothing else in the codebase is allowed to hardcode a filesystem
path, a torch device string or a Hugging Face repo id.

Settings are read from the process environment (prefix ``FS_``) or a ``.env``
file at the repository root, so the Tauri shell can override them when it
spawns the sidecar without any code change.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <root>/sidecar/app/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

Device = Literal["cuda", "cpu"]
OcrEngine = Literal["paddleocr", "tesseract"]
Captioner = Literal["moondream2", "blip2"]


def _default_data_dir() -> Path:
    """App-managed data directory, outside the source tree.

    Deliberately not inside the repo (or OneDrive): it holds the SQLite file,
    the Qdrant storage and multi-gigabyte model weights, none of which should
    be synced or version-controlled.
    """
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "FileSearch"
    return Path.home() / ".filesearch"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FS_",
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Process / network -------------------------------------------------
    # Hard constraint: the sidecar is local-only. `host` is not configurable by
    # design; see `validate_host` below.
    host: str = "127.0.0.1"
    port: int = 8756
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Field(default_factory=_default_data_dir)

    # Models are large (~9GB), read-only and shared, whereas the rest of the
    # data directory is small, per-install and read-write. Letting them be
    # located separately means tests can redirect the database and file store
    # to a scratch directory without also hiding the weights, and a deployment
    # can put the weights on a different drive. Defaults to <data_dir>/models.
    models_dir_override: Path | None = None

    # --- ML ----------------------------------------------------------------
    # THE device switch. `ml/loaders.py` is the only module that reads it, and
    # every loader takes it as a parameter. Changing this one value (env var
    # FS_ML_DEVICE) moves the whole pipeline between GPU and CPU.
    # See docs/cpu-fallback-plan.md.
    ml_device: Device = "cuda"

    text_embedding_model: str = "BAAI/bge-large-en-v1.5"
    text_embedding_dim: int = 1024
    text_query_prefix: str = "Represent this sentence for searching relevant passages: "

    clip_model_dir_name: str = "openclip-vit-h14"
    clip_model_hub_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_embedding_dim: int = 1024

    # Image *understanding* -- captions and CLIP image vectors -- on or off.
    #
    # OCR is deliberately NOT covered by this flag. Reading text out of a scan
    # takes about a second with Tesseract and is the difference between a
    # scanned document being searchable and being invisible, so it belongs to
    # the document path, not the image path.
    #
    # What this turns off is the expensive half: captioning one image with
    # moondream2 costs ~390s on CPU, and CLIP adds a 2.9GB model to the working
    # set. On a GPU both are ~1s and this stays on.
    enable_image_understanding: bool = True

    captioner: Captioner = "moondream2"
    moondream_hub_id: str = "vikhyatk/moondream2"
    # Pinned per hard constraint 3: never track a moving `main`.
    moondream_revision: str = "2025-06-21"
    # moondream2's own code loads its tokenizer from a SEPARATE repo, and the
    # tokenizer.json shipped alongside the weights is an older, smaller vocab
    # (50295 vs the model's 51200). Decoding with it yields fluent nonsense, so
    # this repo has to be vendored too. See docs/model-vendoring.md.
    moondream_tokenizer_hub_id: str = "moondream/starmie-v1"
    moondream_tokenizer_dir_name: str = "moondream2-tokenizer"
    blip2_hub_id: str = "Salesforce/blip2-opt-2.7b"

    ocr_engine: OcrEngine = "paddleocr"
    tesseract_exe: Path | None = None
    ocr_languages: str = "en"

    # Keep at most one large model resident, evicting the others on load.
    # Needed when the models live in system RAM rather than VRAM: bge (1.3GB) +
    # ViT-H/14 (2.9GB) + moondream2 (4.1GB) is ~8.3GB, which does not fit
    # alongside everything else on a 16GB machine. Costs a reload when the
    # pipeline moves between stages, which is seconds against a captioning pass
    # measured in minutes. Off by default: on the target GPU it is pure loss.
    ml_low_memory: bool = False

    # --- Vector search -----------------------------------------------------
    text_collection: str = "text_chunks"
    image_collection: str = "images"
    search_limit: int = 50
    # Results shown on one page. More than this and a list stops being scannable.
    search_page_size: int = 10

    # Minimum similarity for a hit to be shown at all.
    #
    # Measured on a real mixed library (recipes, resumes, a patent, field
    # notes). bge-large does NOT spread its vectors over the full [0,1] range:
    # unrelated text still scores 0.42-0.52 against any query, while a genuine
    # match scores 0.62-0.79. Without a floor every document in the library
    # comes back for every query, because the store happily returns its 50
    # nearest neighbours however far away they are.
    #
    #   "recipe of smoked paprika" -> correct doc 0.785, everything else <=0.470
    #   "quantum chromodynamics"   -> nothing relevant, best is 0.523
    #
    # 0.55 sits in the gap: it keeps real matches, and returns *nothing* for a
    # query the library cannot answer -- which is the honest response.
    min_text_score: float = 0.55

    # CLIP's scale is different again: a good image match is ~0.25-0.38 and an
    # unrelated one ~0.15. Provisional -- retune on real photos once image
    # understanding is switched on.
    min_image_score: float = 0.22

    # Keep only results scoring at least this fraction of the BEST hit.
    #
    # The absolute floor removes results that match nothing; this removes the
    # tail that merely matches less. Real queries against a real library:
    #
    #   "notes of devops subject" -> 0.668, then 0.551 0.550 0.550
    #   "EEIM notes"              -> 0.605, then 0.561
    #
    # In both cases everything after the first is the library's baseline
    # similarity, not an answer. A ratio adapts to the query: a vague one whose
    # best hit only reaches 0.60 gets a proportionally lower cutoff than a sharp
    # one that reaches 0.79, where a fixed margin would behave differently for
    # each. Never applies to the top result.
    relative_score_floor: float = 0.95

    # Above this, the best hit is a confident answer; between `min_text_score`
    # and this, it is "the closest thing I have" and the UI says so.
    #
    # There is deliberately no single threshold that separates "no answer" from
    # "weak answer". Measured over 20 queries on a real library, the best score
    # for a question the library CANNOT answer reached 0.572, while the worst
    # score for one it CAN answer was 0.581 -- a gap of 0.009. Short queries
    # score high against everything, which is a property of the embedding model,
    # not something a threshold can fix. Pretending otherwise would either hide
    # real answers or present noise as fact, so the uncertainty is surfaced
    # instead of hidden.
    confident_match_score: float = 0.62
    # Per-collection candidate depth before fusion. Wide, because results are
    # then collapsed per file: a single 200-page document can otherwise occupy
    # every candidate slot with its own chunks and crowd every other file out
    # of the running before grouping ever happens.
    search_candidates: int = 200
    # Reciprocal-rank-fusion constant. 60 is the value from the original paper
    # and is not sensitive; it damps the advantage of rank-1 hits.
    rrf_k: int = 60

    # How long a query waits for a large model to become free under
    # ml_low_memory. Text is worth waiting for -- without it there are no
    # results at all. Images are a bonus, so the wait is short and the query
    # returns text-only rather than blocking behind a captioning pass.
    search_model_wait_seconds: float = 600.0
    image_model_wait_seconds: float = 5.0

    # --- Ingestion ---------------------------------------------------------
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    max_upload_mb: int = 200

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Hard constraint 6: refuse to bind to anything but loopback."""
        if v not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"Refusing to bind sidecar to {v!r}: this backend is loopback-only.")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_dir(self) -> Path:
        """Vendored model weights. Also used as HF_HOME.

        Override with FS_MODELS_DIR_OVERRIDE to keep the weights somewhere other
        than inside the data directory.
        """
        return self.models_dir_override or (self.data_dir / "models")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def files_dir(self) -> Path:
        """Originals of every ingested file, content-addressed."""
        return self.data_dir / "files"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def derived_dir(self) -> Path:
        """Extracted artefacts: page images, embedded images, thumbnails."""
        return self.data_dir / "derived"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_path(self) -> Path:
        """Embedded-mode Qdrant storage directory (no server, no Docker)."""
        return self.data_dir / "qdrant"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_path(self) -> Path:
        return self.data_dir / "filesearch.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clip_model_path(self) -> Path:
        return self.models_dir / self.clip_model_dir_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def moondream_tokenizer_path(self) -> Path:
        return self.models_dir / self.moondream_tokenizer_dir_name / "tokenizer.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tesseract_path(self) -> str:
        """Where to find the Tesseract executable.

        An explicit ``FS_TESSERACT_EXE`` always wins. Otherwise look in the
        standard Windows install locations before falling back to a bare
        ``tesseract`` and hoping it is on PATH -- the UB-Mannheim installer adds
        itself to the *user* PATH, which processes started before the install
        (and services) do not see. Auto-discovery here means a normal install
        just works instead of needing a config edit and a restart.
        """
        if self.tesseract_exe is not None:
            return str(self.tesseract_exe)

        candidates = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        found = shutil.which("tesseract")
        return found or "tesseract"

    def ensure_dirs(self) -> None:
        """Create the app data directories. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.models_dir,
            self.files_dir,
            self.derived_dir,
            self.qdrant_path,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def apply_offline_env(self) -> None:
        """Force every ML library into offline mode (hard constraints 1 & 3).

        Must run before ``huggingface_hub``/``transformers`` are imported, which
        is why ``app/__init__.py`` calls it. Setting these after import is a
        no-op because the libraries snapshot them at import time.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.models_dir))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # sentence-transformers writes its own cache; keep it under models_dir.
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self.models_dir))
        # PaddleOCR: resolve models from the bundled directory, never the net.
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "local")
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.models_dir / "paddlex"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
