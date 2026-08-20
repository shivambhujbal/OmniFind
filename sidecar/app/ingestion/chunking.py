"""Split extracted text into passages for embedding.

Chunk boundaries matter for retrieval quality: a chunk that starts mid-sentence
embeds badly, and a chunk that spans two unrelated sections embeds to the
average of both and matches neither. So the splitter prefers paragraph breaks,
falls back to sentence ends, and only cuts mid-sentence when a single sentence
exceeds the target size.

Overlap exists so a passage straddling a boundary still appears whole in one of
the two chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

# Paragraph break: a blank line, possibly with trailing spaces.
_PARAGRAPH = re.compile(r"\n\s*\n")
# Sentence end: terminator + closing quote/bracket + whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")

# Below this a chunk is almost always a page number, header or stray artefact.
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class TextChunk:
    text: str
    ordinal: int
    page_number: int | None


def normalise(text: str) -> str:
    """Tidy extracted text without changing its meaning.

    PDF extraction in particular produces ragged spacing and hyphen-broken
    words; both hurt embedding quality more than they look like they should.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphens
    # Re-join words broken across a line by hyphenation.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Break an over-long paragraph at sentence ends, hard-cutting if needed."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            pieces.append(current)
        # A single sentence longer than the limit (tables, run-on OCR output)
        # has no good boundary, so cut it on size.
        while len(sentence) > limit:
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    page_number: int | None = None,
    start_ordinal: int = 0,
    size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    """Split one page or document into overlapping chunks."""
    size = size or settings.chunk_size_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap_chars
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = normalise(text)
    if len(text) < MIN_CHUNK_CHARS:
        return []

    # Build units that are each at most `size`, preferring paragraph breaks.
    units: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, size))

    # Pack units into chunks, carrying `overlap` characters across boundaries.
    chunks: list[TextChunk] = []
    current = ""
    ordinal = start_ordinal

    def flush() -> str:
        """Emit the current chunk and return the text to carry forward."""
        nonlocal ordinal
        if len(current) >= MIN_CHUNK_CHARS:
            chunks.append(TextChunk(current.strip(), ordinal, page_number))
            ordinal += 1
        if overlap <= 0:
            return ""
        tail = current[-overlap:]
        # Start the carry-over at a sentence boundary when there is one nearby,
        # so the next chunk does not open mid-word.
        match = _SENTENCE_END.search(tail)
        return tail[match.end() :] if match else tail

    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            current = flush()
            current = f"{current}\n\n{unit}".strip() if current else unit
        else:
            current = f"{current}\n\n{unit}".strip() if current else unit

    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(TextChunk(current.strip(), ordinal, page_number))

    return chunks


def chunk_pages(pages: list[tuple[int, str]]) -> list[TextChunk]:
    """Chunk a document page by page, keeping ordinals continuous.

    Chunks never span a page break: a result has to point at one page for the
    "open at page N" action to work, and page boundaries usually coincide with
    a topic boundary anyway.
    """
    chunks: list[TextChunk] = []
    ordinal = 0
    for page_number, text in pages:
        page_chunks = chunk_text(text, page_number=page_number, start_ordinal=ordinal)
        chunks.extend(page_chunks)
        ordinal += len(page_chunks)
    return chunks
