"""Search tab: query, filter, ranked results, page navigation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClient, BackendError

MATCH_LABEL = {
    "text": "document text",
    "ocr": "text read from an image",
    "caption": "image description",
    "page_image": "scanned page",
    "embedded_image": "picture in the document",
    "source_image": "image",
}

KIND_LABEL = {"pdf": "PDF", "docx": "DOCX", "image": "Image"}

# Label shown to the user -> value the API expects.
KIND_FILTERS = {"Everything": "all", "Documents": "documents", "Images": "images"}

EXAMPLES = [
    "what did the invoice total come to?",
    "how far do sea birds travel each year?",
    "how do I look after my bread culture?",
]


def _display_name(result: dict[str, Any]) -> str:
    """What to head the result with.

    A real metadata title wins. Otherwise the filename, with separators turned
    back into spaces -- "Recipe_Development_Smoked_Paprika_Stew.pdf" is the
    truth but "Recipe Development Smoked Paprika Stew.pdf" is legible. The
    extension stays: it is information, and nothing else distinguishes a .docx
    from a .doc.
    """
    if result.get("title"):
        return str(result["title"])

    name = str(result.get("file_name") or "untitled")
    stem, dot, extension = name.rpartition(".")
    if not dot:
        return name.replace("_", " ")
    return f"{stem.replace('_', ' ')}.{extension}"


def _strength(similarity: float) -> str:
    """Describe the match in words as well as a number.

    The raw cosine is the honest signal but needs calibration to read: with
    bge, unrelated text still scores ~0.45, so "0.56" is a weak match and
    "0.78" a strong one. A bare number invites the reader to assume 0-1 means
    0-100%.
    """
    if similarity >= 0.72:
        return "strong"
    if similarity >= 0.62:
        return "good"
    return "weak"


def _location(result: dict[str, Any]) -> str:
    parts = [MATCH_LABEL.get(result["match_kind"], result["match_kind"])]
    if result.get("page_number"):
        parts.append(f"page {result['page_number']}")
    similarity = result.get("similarity")
    if similarity is not None:
        parts.append(f"{_strength(similarity)} match ({similarity:.2f})")
    return " · ".join(parts)


def _render_result(client: ApiClient, result: dict[str, Any], rank: int) -> None:
    kind = KIND_LABEL.get(result["file_kind"], result["file_kind"])

    with st.container(border=True):
        header, action = st.columns([5, 1])
        header.markdown(f"**{rank}. {_display_name(result)}**")
        header.caption(f"{kind} · {_location(result)}")
        action.link_button("Open", client.file_content_url(result["file_id"]))

        if result.get("asset_id"):
            preview, text = st.columns([1, 3])
            preview.image(
                client.asset_image_url(result["file_id"], result["asset_id"]),
                use_container_width=True,
            )
            if result.get("snippet"):
                text.write(result["snippet"])
        elif result.get("snippet"):
            st.write(result["snippet"])

        supporting = result.get("supporting") or []
        if supporting:
            with st.expander(f"{len(supporting)} more match(es) in this file"):
                for extra in supporting:
                    st.caption(_location(extra))
                    if extra.get("snippet"):
                        st.write(extra["snippet"])
                    st.divider()


def _render_pagination(response: dict[str, Any]) -> None:
    """Page navigation, shown only when there is more than one page."""
    total_pages = response["total_pages"]
    if total_pages <= 1:
        return

    page = response["page"]
    previous, indicator, following = st.columns([1, 3, 1])

    if previous.button("← Previous", disabled=page <= 1, use_container_width=True):
        st.session_state["search_page"] = page - 1
        st.rerun()

    start = (page - 1) * response["page_size"] + 1
    end = min(page * response["page_size"], response["total"])
    indicator.markdown(
        f"<div style='text-align:center'>Showing {start}–{end} of {response['total']} "
        f"&nbsp;·&nbsp; page {page} of {total_pages}</div>",
        unsafe_allow_html=True,
    )

    if following.button("Next →", disabled=page >= total_pages, use_container_width=True):
        st.session_state["search_page"] = page + 1
        st.rerun()


def _reset_page() -> None:
    """A changed query or filter invalidates the current page number.

    Staying on page 4 after narrowing to Images is how a user ends up staring
    at an empty screen and concluding there are no results.
    """
    st.session_state["search_page"] = 1


def render(client: ApiClient) -> None:
    try:
        status = client.search_status()
        indexed = sum(status["collections"].values())
    except BackendError as exc:
        st.error(str(exc))
        return

    if indexed == 0:
        st.info(
            "Nothing is indexed yet. Add a folder in the Library tab — files become "
            "searchable once processing finishes."
        )
        return

    st.caption(
        " · ".join(f"{name}: {count} vectors" for name, count in status["collections"].items())
    )

    query = st.text_input(
        "Search your files",
        placeholder="Describe what you're looking for, in your own words",
        key="search_query",
        on_change=_reset_page,
    )

    images_on = status.get("image_understanding", True)
    filters, options = st.columns([2, 3])

    kind_label = filters.radio(
        "Show",
        list(KIND_FILTERS),
        horizontal=True,
        key="search_kind",
        on_change=_reset_page,
        help=(
            "Images can only be matched by their text (OCR) while image " "understanding is off."
            if not images_on
            else "Filter by the kind of file a match came from."
        ),
    )
    kinds = KIND_FILTERS[kind_label]

    with options.expander("Options"):
        group_by_file = st.checkbox("One result per file", value=True, on_change=_reset_page)
        page_size = st.slider("Results per page", 5, 50, 10, step=5, on_change=_reset_page)

    if not query:
        st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))
        return

    page = st.session_state.get("search_page", 1)

    with st.spinner("Searching…"):
        try:
            response = client.search(
                query,
                include_images=images_on,
                group_by_file=group_by_file,
                kinds=kinds,
                page=page,
                page_size=page_size,
            )
        except BackendError as exc:
            st.error(str(exc))
            return

    results = response["results"]
    if not results:
        if response["total"]:
            # Landed past the end, usually after narrowing a filter.
            st.info("No results on this page.")
            _reset_page()
            st.rerun()
        st.warning(
            "Nothing in your library matches that closely. Try describing the "
            "content differently, or in more detail."
        )
        return

    if not response.get("confident", True):
        # Phrased as a caveat, not a verdict. The match is often still correct
        # -- short or acronym-heavy queries score low even when they hit -- so
        # announcing "nothing matched" over the right answer would be its own
        # kind of wrong. There is no threshold that cleanly separates "no
        # answer" from "weak answer" (measured gap: 0.009), so the uncertainty
        # is surfaced and the user decides.
        st.caption(
            "Weak match — short queries often score low even when they are right. "
            "If this isn't what you meant, try describing the content in more detail."
        )

    summary = f"{response['total']} result(s) from {response['text_candidates']} text"
    if response["searched_images"]:
        summary += f" and {response['image_candidates']} image candidates"
    elif not images_on and kinds != "documents":
        summary += " — documents only; image understanding is off"
    st.caption(summary)

    offset = (response["page"] - 1) * response["page_size"]
    for index, result in enumerate(results, start=offset + 1):
        _render_result(client, result, index)

    _render_pagination(response)
