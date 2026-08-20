"""Index a folder the user already has, in place.

This is the app's natural workflow -- the user points it at their documents
rather than uploading them one by one through a browser. The originals are
copied into the content-addressed store and **never modified or moved**.

Deliberately not recursive by default: pointing at a home directory and
descending through it is rarely what someone means, and it can pull in tens of
thousands of files before they realise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion import detect, pipeline
from app.logging_conf import get_logger

log = get_logger(__name__)

# Folders that are never worth walking: caches, version control, and this app's
# own data directory (which would re-ingest the store from itself).
SKIP_DIRECTORIES = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "$RECYCLE.BIN",
    "System Volume Information",
}


class ScanError(ValueError):
    """The folder cannot be scanned. Message is meant for the user."""


@dataclass
class ScanResult:
    folder: str
    queued: list[int] = field(default_factory=list)
    already_present: int = 0
    unsupported: int = 0
    too_large: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def considered(self) -> int:
        return (len(self.queued) + self.already_present + self.unsupported + self.too_large) + len(
            self.failed
        )


def iter_candidate_files(folder: Path, recursive: bool) -> list[Path]:
    """Supported files in a folder, sorted for a predictable order."""
    walker = folder.rglob("*") if recursive else folder.glob("*")
    candidates = []
    for path in walker:
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in detect.SUPPORTED:
            candidates.append(path)
    return sorted(candidates)


def validate_folder(raw: str) -> Path:
    """Turn user input into a folder path, or explain why it cannot be used."""
    text = raw.strip().strip('"')
    if not text:
        raise ScanError("Enter a folder path.")

    folder = Path(text).expanduser()
    if not folder.exists():
        raise ScanError(f"No such folder: {folder}")
    if not folder.is_dir():
        raise ScanError(f"That is a file, not a folder: {folder}")

    # Indexing the app's own store would re-ingest every file from its
    # content-addressed copy, under a hashed name.
    try:
        resolved = folder.resolve()
        data_dir = settings.data_dir.resolve()
        if resolved == data_dir or data_dir in resolved.parents:
            raise ScanError("That folder is inside the app's own data directory.")
    except OSError:
        # An unresolvable path (a dead junction, a disconnected drive) is not
        # worth failing the scan over; the walk below will simply find nothing.
        pass

    return folder


def scan_folder(session: Session, raw_folder: str, recursive: bool = False) -> ScanResult:
    """Register every supported file in a folder and queue it for extraction.

    Registration happens synchronously so the caller learns immediately how
    many files were accepted; the slow work (extraction, OCR, embedding) runs
    on the background queue.
    """
    from app.tasks import ingest

    folder = validate_folder(raw_folder)
    result = ScanResult(folder=str(folder))
    limit_bytes = settings.max_upload_mb * 1024 * 1024

    candidates = iter_candidate_files(folder, recursive)
    log.info("scanning %s: %d candidate file(s)", folder, len(candidates))

    for path in candidates:
        try:
            if path.stat().st_size > limit_bytes:
                result.too_large += 1
                continue

            file_row, is_new = pipeline.register_existing_file(session, path)
            session.commit()
        except detect.UnsupportedFileError:
            result.unsupported += 1
            continue
        except (OSError, ValueError) as exc:
            session.rollback()
            log.warning("could not register %s: %s", path, exc)
            result.failed.append(f"{path.name}: {exc}")
            continue

        if is_new:
            ingest.submit_extraction(file_row.id, path.name)
            result.queued.append(file_row.id)
        else:
            result.already_present += 1

    log.info(
        "scan of %s: %d queued, %d already present, %d unsupported, %d too large, %d failed",
        folder,
        len(result.queued),
        result.already_present,
        result.unsupported,
        result.too_large,
        len(result.failed),
    )
    return result


def count_supported(raw_folder: str, recursive: bool = False) -> int:
    """How many files a scan would consider. For a preview before committing."""
    return len(iter_candidate_files(validate_folder(raw_folder), recursive))
