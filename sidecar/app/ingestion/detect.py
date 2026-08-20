"""File type detection.

Extension first, magic bytes as the authority. A ``.pdf`` that is really a JPEG
should be treated as an image rather than blowing up the PDF parser, and an
extensionless file should still be usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.db.models import FileKind

# Extensions we accept, mapped to (kind, mime type).
SUPPORTED: dict[str, tuple[FileKind, str]] = {
    ".pdf": (FileKind.PDF, "application/pdf"),
    ".docx": (
        FileKind.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".png": (FileKind.IMAGE, "image/png"),
    ".jpg": (FileKind.IMAGE, "image/jpeg"),
    ".jpeg": (FileKind.IMAGE, "image/jpeg"),
    ".webp": (FileKind.IMAGE, "image/webp"),
    ".bmp": (FileKind.IMAGE, "image/bmp"),
    ".tif": (FileKind.IMAGE, "image/tiff"),
    ".tiff": (FileKind.IMAGE, "image/tiff"),
    ".gif": (FileKind.IMAGE, "image/gif"),
}

SUPPORTED_EXTENSIONS = sorted(SUPPORTED)

# (offset, signature) -> (kind, mime). Enough to separate the formats we accept.
_MAGIC: list[tuple[int, bytes, FileKind, str]] = [
    (0, b"%PDF-", FileKind.PDF, "application/pdf"),
    (0, b"\x89PNG\r\n\x1a\n", FileKind.IMAGE, "image/png"),
    (0, b"\xff\xd8\xff", FileKind.IMAGE, "image/jpeg"),
    (0, b"GIF87a", FileKind.IMAGE, "image/gif"),
    (0, b"GIF89a", FileKind.IMAGE, "image/gif"),
    (0, b"BM", FileKind.IMAGE, "image/bmp"),
    (0, b"II*\x00", FileKind.IMAGE, "image/tiff"),
    (0, b"MM\x00*", FileKind.IMAGE, "image/tiff"),
]


class UnsupportedFileError(ValueError):
    """Raised for a file this app cannot ingest.

    Carries a message meant for the user, not a stack trace: unsupported input
    is an ordinary outcome, not a bug (phase 2 gate).
    """


@dataclass(frozen=True)
class DetectedType:
    kind: FileKind
    mime_type: str


def _sniff(path: Path) -> DetectedType | None:
    try:
        header = path.open("rb").read(16)
    except OSError:
        return None

    for offset, signature, kind, mime in _MAGIC:
        if header[offset : offset + len(signature)] == signature:
            return DetectedType(kind, mime)

    # DOCX and WEBP are both containers whose first bytes are shared with other
    # formats, so they need a second look.
    if header[:4] == b"PK\x03\x04":
        return _sniff_zip_container(path)
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return DetectedType(FileKind.IMAGE, "image/webp")
    return None


def _sniff_zip_container(path: Path) -> DetectedType | None:
    """A ZIP could be .docx, .xlsx, .pptx or an ordinary archive."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return None

    if "word/document.xml" in names:
        return DetectedType(
            FileKind.DOCX,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    return None


def detect(path: Path, original_name: str | None = None) -> DetectedType:
    """Classify a file, or raise ``UnsupportedFileError`` with a usable message.

    Magic bytes win over the extension when they disagree -- the extension is a
    claim, the content is a fact.
    """
    name = original_name or path.name
    extension = Path(name).suffix.lower()

    sniffed = _sniff(path)
    if sniffed is not None:
        return sniffed

    if extension in SUPPORTED:
        # No signature matched but the extension is one we handle: let the
        # format-specific parser make the final call and report a real error.
        kind, mime = SUPPORTED[extension]
        return DetectedType(kind, mime)

    supported = ", ".join(SUPPORTED_EXTENSIONS)
    detail = f"'{extension}' files" if extension else "files without an extension"
    raise UnsupportedFileError(f"Cannot read {detail}. This app handles: {supported}.")


def is_supported(path: Path, original_name: str | None = None) -> bool:
    try:
        detect(path, original_name)
    except UnsupportedFileError:
        return False
    return True
