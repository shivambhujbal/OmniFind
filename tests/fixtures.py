"""Generated test documents.

Built at test time rather than committed as binaries: a few dozen lines of
generator code is easier to review and adjust than an opaque PDF, and it lets a
test ask for exactly the shape it needs (a scanned page, an embedded picture, a
document long enough to chunk).
"""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw

# Two topically DISTINCT bodies of text. They must not overlap: a retrieval
# test where both documents are about the same subject cannot tell a working
# ranker from a broken one, and will fail for the wrong reason.
SURVEY_TEXT = (
    "Transect counts were repeated at low tide on three consecutive mornings. "
    "Each observer walked a fixed two hundred metre line and recorded every "
    "individual within five metres of the shore, noting substrate type and "
    "disturbance from dog walkers. Tally sheets were reconciled the same evening "
    "and discrepancies above ten percent triggered a repeat count. "
)

LOREM = (
    "The migration of arctic terns spans nearly forty thousand kilometres each year, "
    "the longest recorded of any animal. Birds breeding in Greenland cross the Atlantic, "
    "follow the west coast of Africa, and continue to the Weddell Sea before turning north "
    "again in the southern autumn. Satellite tagging in the 2010s revised earlier estimates "
    "sharply upward and revealed a mid-Atlantic staging area where the birds pause to feed "
    "for roughly a month. "
)


def make_text_pdf(path: Path, pages: int = 3) -> Path:
    """A normal, text-bearing PDF."""
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 90), f"Chapter {index + 1}: Arctic Tern Migration", fontsize=16)
        page.insert_textbox(
            fitz.Rect(72, 120, 520, 720), LOREM * 3, fontsize=11, align=fitz.TEXT_ALIGN_LEFT
        )
    document.save(str(path))
    document.close()
    return path


def make_scanned_pdf(path: Path, pages: int = 1) -> Path:
    """A PDF whose pages are images with no text layer, i.e. a scan.

    Exercises the path where extraction yields no chunks and OCR has to supply
    the content in phase 3.
    """
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw.text((100, 100), "SCANNED INVOICE 2024-118", fill="black")
        draw.text((100, 200), "Total due: 4,820.00", fill="black")
        draw.rectangle([80, 80, 1120, 400], outline="black", width=3)

        temp = path.with_suffix(".page.png")
        image.save(temp)
        page.insert_image(page.rect, filename=str(temp))
        temp.unlink()
    document.save(str(path))
    document.close()
    return path


def make_pdf_with_image(path: Path) -> Path:
    """A text PDF that also carries an embedded picture."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 90), "Field Report: Coastal Survey", fontsize=16)
    page.insert_textbox(fitz.Rect(72, 120, 520, 400), LOREM, fontsize=11)

    image = Image.new("RGB", (600, 400), "#2d6a9f")
    draw = ImageDraw.Draw(image)
    draw.ellipse([150, 100, 450, 300], fill="#f2c14e")
    temp = path.with_suffix(".embed.png")
    image.save(temp)
    page.insert_image(fitz.Rect(72, 420, 372, 620), filename=str(temp))
    temp.unlink()

    document.save(str(path))
    document.close()
    return path


def make_docx(path: Path, paragraphs: int = 8, with_table: bool = True) -> Path:
    import docx

    document = docx.Document()
    document.core_properties.title = "Coastal Survey Notes"
    document.add_heading("Coastal Survey Notes", level=1)
    for index in range(paragraphs):
        document.add_paragraph(f"Observation {index + 1}. {SURVEY_TEXT}")

    if with_table:
        table = document.add_table(rows=3, cols=3)
        headers = ["Site", "Count", "Date"]
        for column, header in enumerate(headers):
            table.cell(0, column).text = header
        table.cell(1, 0).text = "Sandy Point"
        table.cell(1, 1).text = "412"
        table.cell(1, 2).text = "2024-06-11"
        table.cell(2, 0).text = "North Spit"
        table.cell(2, 1).text = "87"
        table.cell(2, 2).text = "2024-06-12"

    document.save(str(path))
    return path


def make_image(path: Path, size: tuple[int, int] = (900, 600)) -> Path:
    image = Image.new("RGB", size, "#1b3a4b")
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 60, size[0] - 60, size[1] - 60], outline="#e8eddf", width=6)
    draw.text((100, 100), "HARBOUR NOTICE", fill="#e8eddf")
    draw.ellipse([300, 250, 600, 450], fill="#c84630")
    image.save(path)
    return path


def make_unsupported(path: Path) -> Path:
    """A file type the app does not handle, for the reject path."""
    path.write_bytes(b"\x00\x01\x02not a document at all\xff")
    return path


def make_corrupt_pdf(path: Path) -> Path:
    """Correct magic bytes, garbage contents."""
    path.write_bytes(b"%PDF-1.7\n" + b"\xde\xad\xbe\xef" * 200)
    return path
