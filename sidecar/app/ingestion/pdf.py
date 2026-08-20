"""PDF extraction with PyMuPDF.

Produces three things per document: page text, embedded pictures, and rendered
page bitmaps for the pages that carry no extractable text (i.e. scans, which
need OCR in phase 3).

Rendering *only* the text-less pages is deliberate. Rendering everything would
add a few megabytes per document and a captioning/OCR job per page for content
already available as text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from app.ingestion.titles import clean_title
from app.logging_conf import get_logger
from app.storage import files as storage

log = get_logger(__name__)

# A page with less than this much text is treated as a scan rather than text.
# Real text pages hold hundreds of characters; a scanned page usually yields a
# handful of stray marks, and an empty one yields none.
SCANNED_PAGE_TEXT_THRESHOLD = 80

# 200 DPI: enough for OCR accuracy, well short of the size explosion at 300+.
RENDER_DPI = 200

# Ignore tiny images -- bullets, rules, logos in headers. Nothing under this is
# worth a captioning pass.
MIN_IMAGE_PIXELS = 128


@dataclass
class ExtractedImage:
    path: Path
    # None for DOCX, which has no stored pagination.
    page_number: int | None
    ordinal: int
    width: int
    height: int
    is_page_render: bool = False


@dataclass
class PdfExtraction:
    pages: list[tuple[int, str]] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    page_count: int = 0
    title: str | None = None
    scanned_pages: list[int] = field(default_factory=list)


def extract(path: Path, sha256: str) -> PdfExtraction:
    """Pull text and images out of a PDF into the derived-files directory."""
    result = PdfExtraction()

    with fitz.open(path) as document:
        if document.needs_pass:
            raise ValueError("This PDF is password protected.")

        result.page_count = document.page_count
        result.title = clean_title(document.metadata.get("title"), path.name)

        image_ordinal = 0
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document[page_index]

            text = page.get_text("text")
            result.pages.append((page_number, text))

            is_scanned = len(text.strip()) < SCANNED_PAGE_TEXT_THRESHOLD
            if is_scanned:
                result.scanned_pages.append(page_number)
                render = _render_page(page, sha256, page_number)
                if render is not None:
                    result.images.append(render)
                # The render is a composite of the whole page, so it already
                # contains every picture on it. Extracting them separately would
                # OCR and caption the same content twice -- observed as two
                # near-identical chunks off one scanned invoice, and double the
                # captioning cost, which at ~390s per image on CPU is not a
                # rounding error. The render wins over the raw embedded image
                # because it includes any vector overlay drawn on top.
                continue

            image_ordinal = _extract_page_images(
                document, page, sha256, page_number, image_ordinal, result
            )

    log.info(
        "pdf %s: %d pages, %d scanned, %d images",
        path.name,
        result.page_count,
        len(result.scanned_pages),
        len(result.images),
    )
    return result


def _render_page(page: fitz.Page, sha256: str, page_number: int) -> ExtractedImage | None:
    """Rasterise a page that has no extractable text, for OCR later."""
    try:
        pixmap = page.get_pixmap(dpi=RENDER_DPI)
    except (RuntimeError, ValueError) as exc:
        log.warning("could not render page %d: %s", page_number, exc)
        return None

    destination = storage.derived_path(sha256, "pages", f"page-{page_number:04d}.png")
    pixmap.save(str(destination))
    return ExtractedImage(
        path=destination,
        page_number=page_number,
        ordinal=page_number,
        width=pixmap.width,
        height=pixmap.height,
        is_page_render=True,
    )


def _extract_page_images(
    document: fitz.Document,
    page: fitz.Page,
    sha256: str,
    page_number: int,
    ordinal: int,
    result: PdfExtraction,
) -> int:
    """Save the pictures embedded in one page. Returns the next ordinal."""
    seen: set[int] = set()

    for image_info in page.get_images(full=True):
        xref = image_info[0]
        # The same logo can be referenced by every page; store it once.
        if xref in seen:
            continue
        seen.add(xref)

        try:
            raw = document.extract_image(xref)
        except (RuntimeError, ValueError) as exc:
            log.debug("skipping image xref %d on page %d: %s", xref, page_number, exc)
            continue

        width, height = raw.get("width", 0), raw.get("height", 0)
        if width < MIN_IMAGE_PIXELS or height < MIN_IMAGE_PIXELS:
            continue

        extension = raw.get("ext", "png")
        destination = storage.derived_path(
            sha256, "images", f"img-{page_number:04d}-{ordinal:04d}.{extension}"
        )
        destination.write_bytes(raw["image"])

        result.images.append(
            ExtractedImage(
                path=destination,
                page_number=page_number,
                ordinal=ordinal,
                width=width,
                height=height,
            )
        )
        ordinal += 1

    return ordinal
