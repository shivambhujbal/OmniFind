"""Library tab: point the app at a folder, watch it index, inspect the result."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from api_client import ApiClient, BackendError

STATUS_LABEL = {
    "pending": "queued",
    "extracting": "extracting",
    "extracted": "extracted",
    "processing": "processing",
    "ready": "ready",
    "failed": "failed",
}

KIND_LABEL = {"pdf": "PDF", "docx": "DOCX", "image": "Image"}

# How often to re-check progress while the queue drains. Short enough to feel
# live, long enough not to hammer the backend from a rerun loop.
POLL_SECONDS = 1.5


def _human_size(size_bytes: int) -> str:
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.1f} MB" if mb >= 1 else f"{size_bytes / 1024:.0f} KB"


def _run_scan(client: ApiClient, path: str, recursive: bool) -> None:
    """Register a folder, then follow the queue to completion."""
    with st.spinner("Reading folder and copying files into the library…"):
        try:
            result = client.scan_folder(path, recursive=recursive)
        except BackendError as exc:
            st.error(str(exc))
            return

    parts = [f"{result['queued']} file(s) added"]
    if result["already_present"]:
        parts.append(f"{result['already_present']} already in the library")
    if result["unsupported"]:
        parts.append(f"{result['unsupported']} unsupported")
    if result["too_large"]:
        parts.append(f"{result['too_large']} over the size limit")
    st.success(" · ".join(parts))

    for failure in result["failed"][:10]:
        st.warning(failure)

    if result["queued"]:
        _follow_progress(client, expected=result["queued"])


def _follow_progress(client: ApiClient, expected: int) -> None:
    """Show a live progress bar until the background queue drains.

    Progress is measured against the number of jobs still outstanding rather
    than a timer: each file passes through extract, process and index, so the
    queue depth is what actually tells the user how far along they are.
    """
    progress = st.progress(0.0, text="Processing…")
    status = st.empty()

    peak = 0
    while True:
        try:
            stats = client.stats()
        except BackendError as exc:
            status.warning(f"Lost contact with the backend: {exc}")
            return

        pending = stats["pending_jobs"]
        peak = max(peak, pending)
        if pending == 0:
            progress.progress(1.0, text="Done")
            status.success(
                f"{stats['total_files']} file(s) in the library · "
                f"{stats['total_chunks']} searchable passages"
            )
            return

        done = peak - pending
        progress.progress(
            min(0.99, done / peak) if peak else 0.0,
            text=f"Processing… {pending} job(s) remaining",
        )
        status.caption(
            f"{stats['by_status'].get('ready', 0)} ready · "
            f"{stats['by_status'].get('extracted', 0)} extracted · "
            f"{stats['by_status'].get('failed', 0)} failed"
        )
        time.sleep(POLL_SECONDS)


def render_add(client: ApiClient) -> None:
    st.subheader("Add files")
    st.caption(
        "Point the app at a folder on this machine. Your files are copied into "
        "the library and never moved or changed."
    )

    path = st.text_input(
        "Folder path",
        placeholder=r"C:\Users\you\Documents\Notes",
        key="scan_path",
    )
    options, action = st.columns([3, 1])
    recursive = options.checkbox(
        "Include subfolders",
        value=False,
        help="Off by default — a folder like Documents can hold tens of thousands of files.",
    )

    if path:
        try:
            preview = client.scan_preview(path, recursive=recursive)
            count = preview["supported_files"]
            if count:
                options.caption(f"{count} supported file(s) found in {preview['folder']}")
            else:
                options.caption(f"No supported files in {preview['folder']}")
        except BackendError as exc:
            options.caption(str(exc))
            count = 0
    else:
        count = 0

    if action.button("Index folder", type="primary", disabled=not path or not count):
        _run_scan(client, path, recursive)

    with st.expander("Or upload individual files"):
        _render_upload(client)


def _render_upload(client: ApiClient) -> None:
    try:
        accepted = client.supported_types()
    except BackendError:
        accepted = [".pdf", ".docx", ".png", ".jpg"]

    uploads = st.file_uploader(
        "Choose files",
        type=[ext.lstrip(".") for ext in accepted],
        accept_multiple_files=True,
    )
    if not uploads:
        return

    # Streamlit keeps uploaded files in widget state across reruns, so without
    # this guard the same batch is re-sent on every interaction.
    signature = tuple(sorted((u.name, u.size) for u in uploads))
    if st.session_state.get("last_upload") == signature:
        st.caption(f"{len(uploads)} file(s) already submitted.")
        return

    if not st.button(f"Add {len(uploads)} file(s)"):
        return

    with st.spinner("Uploading…"):
        try:
            response = client.upload([(u.name, u.getvalue()) for u in uploads])
        except BackendError as exc:
            st.error(str(exc))
            return

    st.session_state["last_upload"] = signature
    if response["accepted"]:
        st.success(f"Accepted {response['accepted']} file(s).")
    for result in response["results"]:
        if not result["accepted"]:
            st.warning(f"**{result['original_name']}** — {result['reason']}")
        elif result["duplicate"]:
            st.info(f"**{result['original_name']}** — already in the library.")

    if any(r["accepted"] and not r["duplicate"] for r in response["results"]):
        _follow_progress(client, expected=response["accepted"])


def render_stats(client: ApiClient) -> None:
    try:
        stats = client.stats()
    except BackendError as exc:
        st.error(str(exc))
        return

    columns = st.columns(4)
    columns[0].metric("Files", stats["total_files"])
    columns[1].metric("Text passages", stats["total_chunks"])
    columns[2].metric("Images", stats["total_assets"])
    columns[3].metric("Jobs running", stats["pending_jobs"])

    usage = stats["disk_usage_mb"]
    st.caption(
        "Disk — originals {originals} MB · extracted {derived} MB · "
        "vectors {vectors} MB · models {models} MB".format(**usage)
    )

    if stats["pending_jobs"]:
        st.info(f"{stats['pending_jobs']} job(s) still running in the background.")
        if st.button("Refresh"):
            st.rerun()


def _render_file_detail(client: ApiClient, file_id: int) -> None:
    try:
        detail = client.get_file(file_id)
    except BackendError as exc:
        st.error(str(exc))
        return

    if detail["error"]:
        st.error(f"Extraction failed: {detail['error']}")

    meta = st.columns(4)
    meta[0].metric("Status", STATUS_LABEL.get(detail["status"], detail["status"]))
    meta[1].metric("Passages", detail["chunk_count"])
    meta[2].metric("Images", detail["asset_count"])
    meta[3].metric("Pages", detail["page_count"] or "—")

    actions = st.columns(3)
    actions[0].link_button("Open file", client.file_content_url(file_id))
    if actions[1].button("Reprocess", key=f"reprocess-{file_id}"):
        client.reprocess_file(file_id)
        st.rerun()
    if actions[2].button("Remove", key=f"delete-{file_id}"):
        client.delete_file(file_id)
        st.rerun()

    chunks_tab, images_tab = st.tabs(
        [f"Text ({detail['chunk_count']})", f"Images ({detail['asset_count']})"]
    )

    with chunks_tab:
        if not detail["chunks"]:
            st.caption("No extractable text. If this is a scan, OCR supplies the text.")
        for chunk in detail["chunks"][:40]:
            page = f"page {chunk['page_number']}" if chunk["page_number"] else "—"
            st.caption(f"#{chunk['ordinal']} · {page} · {chunk['kind']}")
            st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
            st.divider()

    with images_tab:
        assets = [a for a in detail["assets"] if a["kind"] != "thumbnail"]
        if not assets:
            st.caption("No images in this file.")
        columns = st.columns(3)
        for index, asset in enumerate(assets[:12]):
            with columns[index % 3]:
                st.image(client.asset_image_url(file_id, asset["id"]), use_container_width=True)
                label = asset["kind"].replace("_", " ")
                if asset["page_number"]:
                    label += f" · page {asset['page_number']}"
                st.caption(label)
                if asset["caption"]:
                    st.caption(f"“{asset['caption']}”")


def render_file_list(client: ApiClient) -> None:
    try:
        files: list[dict[str, Any]] = client.list_files()
    except BackendError as exc:
        st.error(str(exc))
        return

    if not files:
        st.info("Nothing in the library yet. Add a folder above to get started.")
        return

    st.subheader(f"Library ({len(files)})")
    for file_row in files:
        kind = KIND_LABEL.get(file_row["kind"], file_row["kind"])
        status = STATUS_LABEL.get(file_row["status"], file_row["status"])
        title = file_row["title"] or file_row["original_name"]

        with st.expander(
            f"{title}  —  {kind} · {status} · {file_row['chunk_count']} passages · "
            f"{_human_size(file_row['size_bytes'])}"
        ):
            if title != file_row["original_name"]:
                st.caption(file_row["original_name"])
            _render_file_detail(client, file_row["id"])


def render(client: ApiClient) -> None:
    render_add(client)
    st.divider()
    render_stats(client)
    st.divider()
    render_file_list(client)
