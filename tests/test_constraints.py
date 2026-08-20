"""Tests for the project's hard constraints.

These are not unit tests of behaviour -- they are executable versions of the
rules in CLAUDE.md, so a violation fails CI instead of being noticed six months
later when someone tries to add a CPU fallback.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "sidecar" / "app"

# Only these two modules may mention a torch device. config.py *defines* the
# setting; loaders.py is the single place that consumes it (hard constraint 2).
DEVICE_ALLOWED = {"config.py", "loaders.py"}

DEVICE_LITERAL = re.compile(r"""['"](cuda|cuda:\d+)['"]""")


def app_modules() -> list[Path]:
    return sorted(p for p in APP_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("module", app_modules(), ids=lambda p: str(p.relative_to(APP_ROOT)))
def test_no_hardcoded_device(module: Path) -> None:
    """Constraint 2: device strings live in config.py, chosen in loaders.py."""
    if module.name in DEVICE_ALLOWED:
        pytest.skip(f"{module.name} is the designated device seam")
    source = module.read_text(encoding="utf-8")
    hits = DEVICE_LITERAL.findall(source)
    assert not hits, (
        f"{module.relative_to(REPO_ROOT)} hardcodes a device {hits}. "
        "Read settings.ml_device and pass it to a loader in ml/loaders.py instead."
    )


@pytest.mark.parametrize("module", app_modules(), ids=lambda p: str(p.relative_to(APP_ROOT)))
def test_only_loaders_construct_models(module: Path) -> None:
    """Constraint 2: no module outside loaders.py builds a model."""
    if module.name in DEVICE_ALLOWED:
        pytest.skip(f"{module.name} is the designated model-loading seam")
    source = module.read_text(encoding="utf-8")
    forbidden = [
        "SentenceTransformer(",
        "open_clip.create_model",
        "from_pretrained(",
        "AutoModel",
        "PaddleOCR(",
    ]
    hits = [needle for needle in forbidden if needle in source]
    assert not hits, (
        f"{module.relative_to(REPO_ROOT)} constructs a model {hits}. "
        "All model construction belongs in ml/loaders.py."
    )


def test_no_multi_tenant_scaffolding() -> None:
    """Constraint 5: one implicit local user, no auth or tenant plumbing."""
    forbidden = re.compile(r"\b(tenant_id|user_id|owner_id|current_user|oauth2|jwt)\b", re.I)
    offenders = []
    for module in app_modules():
        if forbidden.search(module.read_text(encoding="utf-8")):
            offenders.append(str(module.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"Multi-tenant/auth scaffolding found in {offenders}. "
        "This app has exactly one implicit local user."
    )


def test_host_is_loopback_only() -> None:
    """Constraint 6: the sidecar refuses any non-loopback bind address."""
    from app.config import Settings

    for bad in ("0.0.0.0", "192.168.1.10", ""):
        with pytest.raises(ValueError, match="loopback-only"):
            Settings(host=bad)

    assert Settings(host="127.0.0.1").host == "127.0.0.1"


def test_offline_env_is_applied_on_import() -> None:
    """Constraint 3: importing the package puts the ML stack offline."""
    import os

    import app  # noqa: F401  -- the import itself is what is under test

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["PADDLE_PDX_MODEL_SOURCE"] == "local"
    assert os.environ["HF_HOME"]


def test_download_script_is_not_reachable_from_the_app() -> None:
    """Constraint 4: nothing under app/ may invoke the vendoring script.

    Checked against the parsed import graph rather than the raw text, so that
    documentation mentioning the script by name does not trip the rule.
    """
    import ast

    offenders = []
    for module in app_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            if any("download_models" in name for name in names):
                offenders.append(str(module.relative_to(REPO_ROOT)))
                break

    assert not offenders, f"{offenders} imports download_models. That script is build-machine only."


def test_no_hub_downloads_in_the_app() -> None:
    """Constraint 1: nothing under app/ may fetch from the network."""
    forbidden = re.compile(
        r"\b(snapshot_download|hf_hub_download|urlopen|requests\.(get|post)|urlretrieve)\b"
    )
    offenders = [
        str(m.relative_to(REPO_ROOT))
        for m in app_modules()
        if forbidden.search(m.read_text(encoding="utf-8"))
    ]
    assert (
        not offenders
    ), f"{offenders} performs a network fetch. The app must work with networking off."
