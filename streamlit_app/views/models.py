"""Models tab: what is provisioned, and what that means for search.

Exists so a user can see "captioning unavailable" before they wait through an
ingest that then quietly skips it, rather than after.
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClient, BackendError

# What each model actually buys the user, in their terms rather than ours.
CAPABILITY = {
    "text_embeddings": "Meaning-based search over document text",
    "clip": "Finding images by description, without any text in them",
    "moondream2": "Automatic captions for images",
    "blip2": "Automatic captions (alternative engine)",
    "paddleocr": "Reading text out of scans and photos (GPU)",
    "tesseract": "Reading text out of scans and photos (CPU)",
}


def render(client: ApiClient) -> None:
    try:
        report = client.models()
    except BackendError as exc:
        st.error(str(exc))
        return

    top = st.columns(3)
    top[0].metric("Device", report["device"])
    top[1].metric("OCR engine", report["ocr_engine"])
    top[2].metric("Captioner", report["captioner"])

    if not report.get("image_understanding", True):
        st.info(
            "**Image understanding is off.** Documents and scans are fully searchable — "
            "OCR still reads text out of scanned pages. What is skipped is describing "
            "pictures and searching them by appearance, which costs several minutes per "
            "image on a CPU. Set `FS_ENABLE_IMAGE_UNDERSTANDING=true` on a GPU machine."
        )
    elif report["ready_for_processing"]:
        st.success("All selected engines are installed. Images will be read and captioned.")
    else:
        st.warning(
            "Some engines are missing. Documents are still ingested and their text is "
            "searchable — images just won't be read or captioned until the models are "
            "installed. Files are picked up automatically on the next start, with no "
            "need to re-add them."
        )
        st.code("python scripts/download_models.py", language="text")

    rows = []
    for name, detail in report["models"].items():
        rows.append(
            {
                "model": name,
                "gives you": CAPABILITY.get(name, ""),
                "installed": "yes" if detail["available"] else "no",
                "missing": ", ".join(detail["missing"]) or "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Where these live"):
        for name, detail in report["models"].items():
            st.caption(f"**{name}** — `{detail['path']}`")
