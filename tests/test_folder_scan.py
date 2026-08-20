"""Indexing a folder the user owns.

The defining requirement: reading someone's Documents folder must never modify
it. The upload path deliberately *moves* its source (a temp file this app
spooled) and deletes it on a duplicate -- doing that to a real folder would
destroy the user's files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db.models import File
from app.ingestion import scan
from app.storage import files as storage
from tests import fixtures


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    source = tmp_path / "documents"
    source.mkdir()
    fixtures.make_text_pdf(source / "terns.pdf", pages=1)
    fixtures.make_docx(source / "notes.docx")
    fixtures.make_image(source / "photo.png")
    fixtures.make_unsupported(source / "archive.xyz")
    (source / "subfolder").mkdir()
    fixtures.make_text_pdf(source / "subfolder" / "nested.pdf", pages=1)
    return source


# --- the originals are untouched ---------------------------------------


def test_scanning_does_not_move_or_delete_the_originals(db_session: Session, folder: Path) -> None:
    """The whole point. A folder scan reads; it does not take ownership."""
    before = {p.name: p.read_bytes() for p in folder.glob("*") if p.is_file()}

    scan.scan_folder(db_session, str(folder))

    after = {p.name: p.read_bytes() for p in folder.glob("*") if p.is_file()}
    assert after == before, "the source folder was modified"


def test_rescanning_does_not_delete_duplicates(db_session: Session, folder: Path) -> None:
    """The upload path unlinks the source when the bytes are already stored.

    On a user's folder that would silently delete their file the second time
    they pressed the button.
    """
    scan.scan_folder(db_session, str(folder))
    scan.scan_folder(db_session, str(folder))

    assert (folder / "terns.pdf").exists()
    assert (folder / "notes.docx").exists()


def test_store_existing_file_copies_rather_than_moves(tmp_path: Path) -> None:
    source = fixtures.make_text_pdf(tmp_path / "original.pdf", pages=1)
    original_bytes = source.read_bytes()

    stored = storage.store_existing_file(source)

    assert source.exists(), "the source file was consumed"
    assert source.read_bytes() == original_bytes
    assert stored.absolute_path.exists()
    assert stored.absolute_path.read_bytes() == original_bytes


# --- what gets picked up -----------------------------------------------


def test_supported_files_are_registered_and_the_rest_reported(
    db_session: Session, folder: Path
) -> None:
    result = scan.scan_folder(db_session, str(folder))

    assert len(result.queued) == 3, "pdf, docx and image should all be added"
    assert result.unsupported == 0, "the .xyz file is filtered before registration"
    assert not result.failed

    names = {f.original_name for f in db_session.query(File).all()}
    assert names == {"terns.pdf", "notes.docx", "photo.png"}


def test_subfolders_are_skipped_unless_asked_for(db_session: Session, folder: Path) -> None:
    """Pointing at a home directory and descending is rarely what people mean."""
    shallow = scan.scan_folder(db_session, str(folder))
    assert len(shallow.queued) == 3

    deep = scan.scan_folder(db_session, str(folder), recursive=True)
    assert len(deep.queued) == 1, "only the nested file is new"
    assert deep.already_present == 3


def test_rescanning_reports_nothing_new(db_session: Session, folder: Path) -> None:
    """ "3 added" the first time, "0 added, 3 already present" the second."""
    scan.scan_folder(db_session, str(folder))
    again = scan.scan_folder(db_session, str(folder))

    assert again.queued == []
    assert again.already_present == 3


def test_files_over_the_size_limit_are_counted_not_ingested(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0)
    source = tmp_path / "docs"
    source.mkdir()
    fixtures.make_text_pdf(source / "big.pdf", pages=1)

    result = scan.scan_folder(db_session, str(source))

    assert result.too_large == 1
    assert result.queued == []


def test_skip_directories_are_not_walked(db_session: Session, tmp_path: Path) -> None:
    """A .git or node_modules folder is never worth indexing."""
    source = tmp_path / "project"
    (source / ".git").mkdir(parents=True)
    (source / "node_modules").mkdir()
    fixtures.make_text_pdf(source / ".git" / "hidden.pdf", pages=1)
    fixtures.make_text_pdf(source / "node_modules" / "vendored.pdf", pages=1)
    fixtures.make_text_pdf(source / "real.pdf", pages=1)

    result = scan.scan_folder(db_session, str(source), recursive=True)

    assert len(result.queued) == 1
    assert db_session.query(File).one().original_name == "real.pdf"


# --- bad input ---------------------------------------------------------


def test_a_missing_folder_is_reported_clearly(db_session: Session, tmp_path: Path) -> None:
    with pytest.raises(scan.ScanError, match="No such folder"):
        scan.scan_folder(db_session, str(tmp_path / "nope"))


def test_a_file_path_is_rejected(db_session: Session, tmp_path: Path) -> None:
    path = fixtures.make_text_pdf(tmp_path / "single.pdf", pages=1)
    with pytest.raises(scan.ScanError, match="not a folder"):
        scan.scan_folder(db_session, str(path))


def test_an_empty_path_is_rejected(db_session: Session) -> None:
    with pytest.raises(scan.ScanError, match="Enter a folder path"):
        scan.scan_folder(db_session, "   ")


def test_quoted_paths_are_accepted(db_session: Session, folder: Path) -> None:
    """Windows' "Copy as path" wraps the path in quotes."""
    result = scan.scan_folder(db_session, f'"{folder}"')
    assert len(result.queued) == 3


def test_the_apps_own_data_directory_is_refused(db_session: Session) -> None:
    """Indexing the store would re-ingest every file under its hashed name."""
    from app.config import settings

    with pytest.raises(scan.ScanError, match="data directory"):
        scan.scan_folder(db_session, str(settings.files_dir))
