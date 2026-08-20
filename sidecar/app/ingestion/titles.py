"""Deciding whether a document's metadata title is worth showing.

PDF and DOCX metadata titles are frequently junk left behind by whatever
produced the file. Observed in a real library:

    (anonymous)                    several PDFs from one exporter
    anim-20260807-anim             a build artefact
    US00000011321312B220220503     a patent number, not a title
    Microsoft Word - report.docx   the source filename, twice over

Showing those instead of the filename makes the results list actively worse:
a search result headed "(anonymous)" tells the reader nothing, and three of
them tell them less than nothing.

The filename is the honest fallback -- the user chose it, or at least
recognises it.
"""

from __future__ import annotations

import re

MAX_TITLE_CHARS = 512

# Exact matches (lowercased) that are never a real title.
_JUNK = {
    "untitled",
    "unknown",
    "document",
    "documents",
    "(anonymous)",
    "anonymous",
    "no title",
    "title",
    "presentation",
    "powerpoint presentation",
    "slide 1",
    "chart1",
    "sheet1",
    "book1",
    "new document",
}

# Producers that prefix the source filename, e.g. "Microsoft Word - report.doc".
_PRODUCER_PREFIX = re.compile(r"^(microsoft\s+(word|powerpoint|excel)|adobe\s+\w+)\s*-\s*", re.I)

_DOC_EXTENSIONS = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".indd", ".pages", ".rtf", ".odt")


def _looks_like_an_identifier(title: str) -> bool:
    """A bare code, not something a person would call the document.

    ``US00000011321312B220220503`` and ``anim-20260807-anim`` are both
    single tokens dominated by digits. A genuine one-word title
    ("Bioluminescence") is not.
    """
    if " " in title:
        return False
    digits = sum(character.isdigit() for character in title)
    return digits / len(title) > 0.4 or len(title) > 24


def clean_title(raw: str | None, filename: str | None = None) -> str | None:
    """Return a title worth displaying, or None to fall back to the filename."""
    if not raw:
        return None

    title = _PRODUCER_PREFIX.sub("", raw.strip()).strip()
    if not title:
        return None

    lowered = title.lower()
    if lowered in _JUNK:
        return None
    if lowered.endswith(_DOC_EXTENSIONS):
        return None
    if len(title) < 3:
        return None
    if _looks_like_an_identifier(title):
        return None

    if filename:
        stem = filename.rsplit(".", 1)[0].lower()
        # The "title" is just the filename again, possibly with separators
        # swapped -- no information beyond what is already shown.
        normalised = re.sub(r"[\s_-]+", "", lowered)
        if normalised == re.sub(r"[\s_-]+", "", stem):
            return None

    return title[:MAX_TITLE_CHARS]
