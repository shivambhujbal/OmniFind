"""Streamlit dev frontend for the local file-search backend.

Temporary stand-in for the Tauri + React shell (phase 6). It exercises the same
loopback HTTP API, so it doubles as a manual test harness for every phase.
"""

from __future__ import annotations

import streamlit as st

from api_client import DEFAULT_BASE_URL, ApiClient, BackendError
from views import library, models, search

st.set_page_config(page_title="Local File Search", layout="wide")


def get_client() -> ApiClient:
    base_url = st.session_state.get("base_url", DEFAULT_BASE_URL)
    return ApiClient(base_url)


def render_sidebar() -> tuple[ApiClient, dict | None]:
    st.sidebar.title("Local File Search")
    st.sidebar.caption("Dev frontend — Streamlit stand-in for the Tauri shell")

    st.session_state.setdefault("base_url", DEFAULT_BASE_URL)
    st.sidebar.text_input("Sidecar URL", key="base_url")

    client = get_client()
    st.sidebar.divider()

    try:
        health = client.health()
    except BackendError as exc:
        st.sidebar.error("Backend disconnected")
        st.sidebar.caption(str(exc))
        return client, None

    st.sidebar.success("Backend connected")
    st.sidebar.caption(f"v{health['version']} · up {health['uptime_seconds']:.0f}s")
    return client, health


def render_status(client: ApiClient) -> None:
    st.subheader("Backend status")
    try:
        detail = client.health_detail()
    except BackendError as exc:
        st.error(str(exc))
        st.info(
            "Start the sidecar in another terminal:\n\n"
            "```\npowershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1\n```"
        )
        return

    left, mid, right = st.columns(3)
    left.metric("Version", detail["version"])
    mid.metric("ML device", detail["ml_device"])
    right.metric("Python", detail["python"])

    st.markdown("**Offline environment** — all must be set for the no-network guarantee")
    st.json(detail["offline_env"])

    st.markdown("**Data locations**")
    rows = []
    for name, report in detail["paths"].items():
        rows.append(
            {
                "location": name,
                "path": report["path"],
                "present": "yes" if report["exists"] else "no",
                "size (MB)": report["size_mb"] if report["size_mb"] is not None else "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    client, health = render_sidebar()

    st.title("Local File Search")
    st.caption(
        "Context-aware search across your PDFs, DOCX files and images. "
        "Everything runs on this machine."
    )

    if health is None:
        st.error("The sidecar is not running.")
        st.code("powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1", language="text")
        st.stop()

    tab_library, tab_search, tab_models, tab_status = st.tabs(
        ["Library", "Search", "Models", "Status"]
    )

    with tab_library:
        library.render(client)
    with tab_search:
        search.render(client)
    with tab_models:
        models.render(client)
    with tab_status:
        render_status(client)


if __name__ == "__main__":
    main()
