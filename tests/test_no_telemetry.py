"""Constraint 1 at the frontend boundary.

Streamlit posts usage telemetry to an external webhook on every session unless
told not to. The app promises that nothing leaves the machine, so the config
that disables it is part of the product, not a local preference -- and a config
file is easy to lose in a refactor without anyone noticing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".streamlit" / "config.toml"


def test_streamlit_config_exists() -> None:
    assert CONFIG.exists(), "the .streamlit/config.toml that disables telemetry is missing"


def test_usage_stats_are_disabled() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False


def test_frontend_binds_loopback_only() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1"
    assert config["browser"]["serverAddress"] == "127.0.0.1"


def test_upload_limits_agree_across_the_stack() -> None:
    """A file accepted by the UI must not be rejected by the backend."""
    from app.config import settings

    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == settings.max_upload_mb
