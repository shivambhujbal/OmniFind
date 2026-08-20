"""Phase 1 gate: the sidecar starts and answers the round trip the UI makes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_is_cheap_and_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["uptime_seconds"] >= 0


def test_health_detail_reports_offline_env_and_paths(client: TestClient) -> None:
    body = client.get("/health/detail").json()

    assert body["offline_env"]["HF_HUB_OFFLINE"] == "1"
    assert body["offline_env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert body["ml_device"] in ("cuda", "cpu")

    # The UI renders these; every one must be present and absolute.
    for name in ("data_dir", "models_dir", "files_dir", "qdrant_path", "database"):
        assert name in body["paths"], f"missing path report: {name}"
        assert body["paths"][name]["path"]


def test_cors_allows_loopback_only(client: TestClient) -> None:
    """The Streamlit dev client is a different loopback port, so it needs CORS.

    Anything not on loopback must not get an allow-origin header back.
    """
    allowed = client.get("/health", headers={"Origin": "http://127.0.0.1:8501"})
    assert allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:8501"

    denied = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in denied.headers


def test_lifespan_creates_data_directories(client: TestClient) -> None:
    from app.config import settings

    for path in (settings.data_dir, settings.models_dir, settings.files_dir, settings.qdrant_path):
        assert path.exists(), f"lifespan did not create {path}"
