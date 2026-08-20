"""Test bootstrap.

`FS_DATA_DIR` is redirected to a throwaway directory **before** anything under
`app` is imported. `Settings` is built at `app.config` import time and cached,
so an env var set later would have no effect, and the suite would otherwise
create Qdrant storage and a SQLite file in the user's real app data directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

_TEST_DATA_DIR = Path(tempfile.gettempdir()) / "filesearch-tests"
os.environ["FS_DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ.setdefault("FS_ML_DEVICE", "cpu")
# Two big models at once exceed the commit limit on a 16GB box, and a test
# run that dies with an access violation is worse than a slow one.
os.environ.setdefault("FS_ML_LOW_MEMORY", "true")

# Point the database, file store and vectors at a scratch directory, but keep
# using the REAL models directory. Redirecting that too would hide the ~9GB of
# vendored weights, and every inference test would silently skip -- passing
# while testing nothing.
if "FS_MODELS_DIR_OVERRIDE" not in os.environ:
    _real_data_dir = (
        Path(os.environ["LOCALAPPDATA"]) / "FileSearch"
        if os.environ.get("LOCALAPPDATA")
        else Path.home() / ".filesearch"
    )
    os.environ["FS_MODELS_DIR_OVERRIDE"] = str(_real_data_dir / "models")


@pytest.fixture(scope="session", autouse=True)
def _clean_test_data_dir() -> Iterator[None]:
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)
    _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    yield
    from app.db.session import reset_engine

    # Release the SQLite file handles before removing the tree; Windows refuses
    # to delete a file that is still open.
    reset_engine()
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def _migrated_db(_clean_test_data_dir: None) -> None:
    """Create the schema once, the same way the app does at startup."""
    from app.db import bootstrap

    bootstrap.upgrade_to_head()


@pytest.fixture
def db_session(_migrated_db: None) -> Iterator[Session]:
    """A session against the real schema, rolled back to empty after each test.

    Tables are truncated rather than the file recreated: re-running migrations
    per test is slow, and a shared file keeps the storage paths stable.
    """
    from app.db.models import Asset, Chunk, File
    from app.db.session import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        for model in (Chunk, Asset, File):
            session.query(model).delete()
        session.commit()
        session.close()
