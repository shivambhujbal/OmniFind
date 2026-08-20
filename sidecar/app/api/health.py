"""Health / readiness endpoints.

The frontend calls ``/health`` on startup to confirm the sidecar round trip,
and ``/health/detail`` to show the user what is and is not provisioned.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


class PathReport(BaseModel):
    path: str
    exists: bool
    size_mb: float | None = None


class HealthDetailResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    python: str
    platform: str
    ml_device: str
    offline_env: dict[str, str]
    paths: dict[str, PathReport]


def _dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def _report(path: Path, measure: bool = False) -> PathReport:
    return PathReport(
        path=str(path),
        exists=path.exists(),
        size_mb=_dir_size_mb(path) if measure else None,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap liveness probe -- no disk walking, safe to poll."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
    )


@router.get("/health/detail", response_model=HealthDetailResponse)
def health_detail() -> HealthDetailResponse:
    return HealthDetailResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        ml_device=settings.ml_device,
        offline_env={
            key: os.environ.get(key, "<unset>")
            for key in (
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PADDLE_PDX_MODEL_SOURCE",
            )
        },
        paths={
            "data_dir": _report(settings.data_dir),
            "models_dir": _report(settings.models_dir, measure=True),
            "files_dir": _report(settings.files_dir),
            "qdrant_path": _report(settings.qdrant_path),
            "database": _report(settings.db_path),
        },
    )
