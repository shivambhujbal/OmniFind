"""Content-addressed local file store.

Originals go under ``settings.files_dir`` keyed by SHA-256; anything produced
from them (rendered pages, extracted pictures, thumbnails) goes under
``settings.derived_dir`` in a directory named after the same hash.

Content addressing rather than filenames because re-adding the same document
under a new name must not re-run a multi-minute ML pipeline, and because two
files with the same name are otherwise a collision waiting to happen.

Every path returned by this module is **relative** to the corresponding root, so
the whole data directory can be moved or shipped without rewriting the database.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class StoredFile:
    sha256: str
    relative_path: str
    absolute_path: Path
    size_bytes: int
    already_present: bool


def hash_stream(stream: BinaryIO) -> tuple[str, int]:
    """SHA-256 and byte count, without loading the file into memory."""
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(_HASH_CHUNK), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def hash_path(path: Path) -> tuple[str, int]:
    with path.open("rb") as handle:
        return hash_stream(handle)


def _shard(sha256: str, suffix: str) -> Path:
    """``ab/abcdef...123.pdf`` -- two-character shard keeps directories small.

    Windows Explorer and most filesystems degrade with tens of thousands of
    entries in one directory; 256 buckets is enough for a personal library.
    """
    return Path(sha256[:2]) / f"{sha256}{suffix.lower()}"


def store_upload(source: Path, original_name: str) -> StoredFile:
    """Take ownership of a temp upload, consuming the source.

    Only for files this app created (a spooled HTTP upload). Never call it on a
    path the user owns -- see ``store_existing_file``.
    """
    return _store(source, original_name, consume_source=True)


def store_existing_file(source: Path) -> StoredFile:
    """Copy a file the USER owns into the store, leaving the original alone.

    The distinction from ``store_upload`` is not stylistic. That function moves
    the source and, on a duplicate, deletes it -- correct for a temp file this
    app spooled, catastrophic for a document sitting in someone's Documents
    folder. Indexing a folder must never modify what it indexes.
    """
    return _store(source, source.name, consume_source=False)


def _store(source: Path, original_name: str, *, consume_source: bool) -> StoredFile:
    """Put bytes into the content-addressed store.

    Idempotent: storing identical bytes twice keeps the first copy and reports
    ``already_present``.
    """
    sha256, size = hash_path(source)
    suffix = Path(original_name).suffix or source.suffix
    relative = _shard(sha256, suffix)
    destination = settings.files_dir / relative

    if destination.exists():
        log.info("file already stored: %s (%s)", original_name, sha256[:12])
        if consume_source:
            source.unlink(missing_ok=True)
        return StoredFile(sha256, relative.as_posix(), destination, size, already_present=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if consume_source:
        # Same volume in the normal case, so this is a rename; shutil falls back
        # to a copy when the temp dir is on another drive.
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(str(source), str(destination))
    log.info("stored %s -> %s (%.1f KB)", original_name, relative.as_posix(), size / 1024)
    return StoredFile(sha256, relative.as_posix(), destination, size, already_present=False)


def absolute_file_path(relative_path: str) -> Path:
    return settings.files_dir / relative_path


def derived_root(sha256: str) -> Path:
    """Directory holding everything extracted from one file."""
    path = settings.derived_dir / sha256[:2] / sha256
    path.mkdir(parents=True, exist_ok=True)
    return path


def derived_path(sha256: str, *parts: str) -> Path:
    """Absolute path for a derived artefact, creating parent directories."""
    path = derived_root(sha256).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def relative_derived(path: Path) -> str:
    """Store-relative form of a derived path, for the database."""
    return path.relative_to(settings.derived_dir).as_posix()


def absolute_derived_path(relative_path: str) -> Path:
    return settings.derived_dir / relative_path


def delete_file_data(sha256: str, relative_path: str) -> None:
    """Remove the original and every derived artefact for one file."""
    original = settings.files_dir / relative_path
    original.unlink(missing_ok=True)
    shutil.rmtree(settings.derived_dir / sha256[:2] / sha256, ignore_errors=True)
    log.info("deleted stored data for %s", sha256[:12])


def disk_usage_mb() -> dict[str, float]:
    """Per-area sizes, for the status view."""

    def size_of(root: Path) -> float:
        if not root.exists():
            return 0.0
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / (1024 * 1024)

    return {
        "originals": round(size_of(settings.files_dir), 1),
        "derived": round(size_of(settings.derived_dir), 1),
        "vectors": round(size_of(settings.qdrant_path), 1),
        "models": round(size_of(settings.models_dir), 1),
    }
