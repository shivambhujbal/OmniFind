"""Vendor every model into the local models directory. BUILD MACHINE ONLY.

This is the one script in the repo that is *allowed* to touch the network, and
the shipped app never invokes it (hard constraint 4). Run it once on the dev or
build machine; afterwards the whole pipeline works with networking disabled.

    python scripts/download_models.py                 # everything
    python scripts/download_models.py --only text clip
    python scripts/download_models.py --verify        # check, download nothing

Layout produced under <data_dir>/models:

    bge-large-en-v1.5/          sentence-transformers text encoder
    openclip-vit-h14/           open_clip `local-dir:` bundle
    moondream2/                 vendored repo incl. tokenizer.json + modeling code
    blip2-opt-2.7b/             captioning fallback (optional, --only blip2)
    paddlex/official_models/    PaddleOCR detection/recognition models
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# This script is the deliberate exception to the offline rule. `app.__init__`
# calls `apply_offline_env`, which uses `setdefault` -- so setting these first
# wins, and importing `app.config` below stays safe.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sidecar"))

from app.config import settings  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_name: str
    description: str
    revision: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    # Files that must exist afterwards for the runtime loader to succeed.
    required_files: list[str] = field(default_factory=list)

    @property
    def local_dir(self) -> Path:
        return settings.models_dir / self.local_name


SPECS: list[ModelSpec] = [
    ModelSpec(
        key="text",
        repo_id=settings.text_embedding_model,
        local_name="bge-large-en-v1.5",
        description="Text embeddings (sentence-transformers)",
        # This repo publishes the SAME weights in five formats: safetensors,
        # pytorch_model.bin, ONNX, OpenVINO and TF. Each of the first two is
        # ~1.3GB. sentence-transformers prefers safetensors when present, so
        # everything else is pure download time and installer size.
        ignore_patterns=[
            "pytorch_model.bin",
            "*.onnx",
            "*.onnx_data",
            "onnx/*",
            "openvino/*",
            "*.h5",
            "*.msgpack",
            "*.ot",
        ],
        required_files=["config.json", "tokenizer.json", "modules.json", "model.safetensors"],
    ),
    ModelSpec(
        key="clip",
        repo_id=settings.clip_model_hub_id,
        local_name=settings.clip_model_dir_name,
        description="OpenCLIP ViT-H/14 image+text embeddings",
        # The repo ships the same weights three times (HF format, open_clip bin,
        # open_clip safetensors, ~4GB each). Take only what `local-dir:` reads.
        allow_patterns=[
            "open_clip_config.json",
            "open_clip_model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
        required_files=["open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json"],
    ),
    ModelSpec(
        key="moondream",
        repo_id=settings.moondream_hub_id,
        local_name="moondream2",
        description="Image captioning (primary)",
        revision=settings.moondream_revision,
        # Same duplicate-format problem as bge; keep safetensors only.
        ignore_patterns=["*.gguf", "*.onnx", "*.mlpackage/*", "pytorch_model.bin", "*.h5"],
        # tokenizer.json is mandatory: the loader uses Tokenizer.from_file and
        # never from_pretrained, so a missing file here is a hard failure.
        required_files=["config.json", "tokenizer.json"],
    ),
    ModelSpec(
        key="moondream",
        repo_id=settings.moondream_tokenizer_hub_id,
        local_name=settings.moondream_tokenizer_dir_name,
        description="moondream2 tokenizer (separate repo -- see model-vendoring.md)",
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        required_files=["tokenizer.json"],
    ),
    ModelSpec(
        key="blip2",
        repo_id=settings.blip2_hub_id,
        local_name="blip2-opt-2.7b",
        description="Image captioning (swappable fallback, optional)",
        ignore_patterns=["*.h5", "*.msgpack", "*.bin"],
        required_files=["config.json", "preprocessor_config.json"],
    ),
]

# PaddleOCR resolves these by name at runtime; pre-populating the directory plus
# PADDLE_PDX_MODEL_SOURCE=local stops it from ever attempting a lookup.
PADDLE_MODELS = [
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "PP-LCNet_x1_0_textline_ori",
    "PP-LCNet_x1_0_doc_ori",
    "UVDoc",
]


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def verify(spec: ModelSpec) -> tuple[bool, str]:
    if not spec.local_dir.exists():
        return False, "directory missing"
    missing = [name for name in spec.required_files if not (spec.local_dir / name).exists()]
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, f"{_dir_size_mb(spec.local_dir):.0f} MB"


def download(spec: ModelSpec, force: bool = False) -> None:
    from huggingface_hub import snapshot_download

    ok, detail = verify(spec)
    if ok and not force:
        print(f"  [skip]  {spec.local_name}  ({detail}, already complete)")
        return
    if force and spec.local_dir.exists():
        shutil.rmtree(spec.local_dir)

    at_rev = f"@{spec.revision}" if spec.revision else ""
    print(f"  [get]   {spec.local_name}  <- {spec.repo_id}{at_rev}")
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        # `local_dir` now always writes real files (no blob-cache symlinks),
        # which is what we need: the models dir has to be copyable into an
        # installer as-is. The old `local_dir_use_symlinks=False` is deprecated
        # and ignored, so it is not passed.
        local_dir=str(spec.local_dir),
        allow_patterns=spec.allow_patterns,
        ignore_patterns=spec.ignore_patterns,
        max_workers=4,
    )

    ok, detail = verify(spec)
    print(f"  [{'ok' if ok else 'INCOMPLETE'}]  {spec.local_name}  ({detail})")


def download_paddle() -> None:
    """Pre-populate the PaddleX official-models directory."""
    target = settings.models_dir / "paddlex" / "official_models"
    target.mkdir(parents=True, exist_ok=True)
    try:
        from paddlex.utils.download import download_and_extract  # type: ignore[import-not-found]
    except ImportError:
        print(
            "  [skip]  paddleocr models: paddlex not installed.\n"
            "          Install the GPU OCR stack first (sidecar/requirements-ocr-gpu.txt),\n"
            "          or stay on the Tesseract engine (FS_OCR_ENGINE=tesseract)."
        )
        return

    base = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/" "official_inference_model/paddle3.0.0"
    )
    for name in PADDLE_MODELS:
        if (target / name).exists():
            print(f"  [skip]  paddle/{name} (present)")
            continue
        print(f"  [get]   paddle/{name}")
        download_and_extract(f"{base}/{name}_infer.tar", str(target), name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    keys = [s.key for s in SPECS] + ["paddle"]
    parser.add_argument(
        "--only", nargs="+", choices=keys, help="Subset to fetch (default: all but blip2)."
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if complete.")
    parser.add_argument("--verify", action="store_true", help="Report status; download nothing.")
    args = parser.parse_args()

    settings.ensure_dirs()
    print(f"models dir: {settings.models_dir}\n")

    if args.verify:
        all_ok = True
        for spec in SPECS:
            ok, detail = verify(spec)
            all_ok = all_ok and (ok or spec.key == "blip2")
            mark = "ok" if ok else "--"
            print(f"  [{mark}]  {spec.local_name:<22} {detail:<18} {spec.description}")
        paddle_dir = settings.models_dir / "paddlex" / "official_models"
        # Existence is not enough: the directory is created eagerly, so an empty
        # one would report as installed.
        paddle_ok = paddle_dir.is_dir() and any(paddle_dir.iterdir())
        all_ok = all_ok and paddle_ok
        detail = f"{_dir_size_mb(paddle_dir):.0f} MB" if paddle_ok else "not downloaded"
        print(f"  [{'ok' if paddle_ok else '--'}]  {'paddlex':<22} {detail}")
        print(f"\ntotal: {_dir_size_mb(settings.models_dir):.0f} MB")
        return 0 if all_ok else 1

    # blip2 is a fallback captioner and another ~8GB; opt in explicitly.
    selected = args.only or [s.key for s in SPECS if s.key != "blip2"] + ["paddle"]

    for spec in SPECS:
        if spec.key in selected:
            download(spec, force=args.force)
    if "paddle" in selected:
        download_paddle()

    print(f"\ntotal models size: {_dir_size_mb(settings.models_dir):.0f} MB")
    print("Now disable networking and run the phase gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
