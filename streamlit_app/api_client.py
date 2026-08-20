"""Thin HTTP client for the sidecar.

The Streamlit UI is a *stand-in for the Tauri/React shell* during backend
development. It talks to the sidecar over the same loopback HTTP API the real
frontend will use, so nothing here becomes throwaway knowledge -- the endpoint
contract is the deliverable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_PORT = int(os.environ.get("FS_PORT", "8756"))
DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


@dataclass(frozen=True)
class BackendError(Exception):
    """Raised when the sidecar is unreachable or returns a non-2xx response."""

    message: str

    def __str__(self) -> str:
        return self.message


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        response = None
        try:
            response = requests.request(method, url, timeout=timeout or self.timeout, **kwargs)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise BackendError(f"Cannot reach the sidecar at {self.base_url}.") from exc
        except requests.exceptions.Timeout as exc:
            raise BackendError(f"Timed out after {timeout or self.timeout}s on {path}.") from exc
        except requests.exceptions.HTTPError as exc:
            assert response is not None
            # FastAPI puts the human-readable reason in `detail`; show that
            # rather than a raw JSON blob.
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text[:300]
            raise BackendError(f"{response.status_code}: {detail}") from exc
        return response.json()

    def _get(self, path: str, timeout: float | None = None, **kwargs: Any) -> Any:
        return self._request("GET", path, timeout, **kwargs)

    # --- health ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health", timeout=3.0)

    def health_detail(self) -> dict[str, Any]:
        return self._get("/health/detail", timeout=30.0)

    # --- files ----------------------------------------------------------

    def upload(self, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Send files as multipart. Long timeout: hashing a 200 MB PDF is slow."""
        payload = [("files", (name, data)) for name, data in uploads]
        return self._request("POST", "/files", timeout=300.0, files=payload)

    def scan_preview(self, path: str, recursive: bool = False) -> dict[str, Any]:
        return self._get(
            "/files/scan/preview", timeout=120.0, params={"path": path, "recursive": recursive}
        )

    def scan_folder(self, path: str, recursive: bool = False) -> dict[str, Any]:
        # Registration hashes and copies every file, so a large folder takes a
        # while even though the heavy work is queued.
        return self._request(
            "POST", "/files/scan", timeout=1800.0, json={"path": path, "recursive": recursive}
        )

    def list_files(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._get("/files", params={"limit": limit})

    def get_file(self, file_id: int) -> dict[str, Any]:
        return self._get(f"/files/{file_id}")

    def stats(self) -> dict[str, Any]:
        return self._get("/files/stats", timeout=30.0)

    def supported_types(self) -> list[str]:
        return self._get("/files/supported-types")

    def delete_file(self, file_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/files/{file_id}")

    def reprocess_file(self, file_id: int) -> dict[str, Any]:
        return self._request("POST", f"/files/{file_id}/reprocess")

    def file_content_url(self, file_id: int) -> str:
        return f"{self.base_url}/files/{file_id}/content"

    def asset_image_url(self, file_id: int, asset_id: int) -> str:
        return f"{self.base_url}/files/{file_id}/assets/{asset_id}/image"

    # --- models ---------------------------------------------------------

    def models(self) -> dict[str, Any]:
        return self._get("/models", timeout=30.0)

    # --- search ---------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 50,
        include_images: bool = True,
        group_by_file: bool = True,
        kinds: str = "all",
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        # Generous timeout: the first query of a session pays for loading the
        # encoders, which is tens of seconds on CPU.
        return self._get(
            "/search",
            timeout=180.0,
            params={
                "q": query,
                "limit": limit,
                "include_images": include_images,
                "group_by_file": group_by_file,
                "kinds": kinds,
                "page": page,
                "page_size": page_size,
            },
        )

    def search_status(self) -> dict[str, Any]:
        return self._get("/search/status", timeout=30.0)

    # --- jobs -----------------------------------------------------------

    def jobs(self, limit: int = 25) -> list[dict[str, Any]]:
        return self._get("/jobs", params={"limit": limit})
