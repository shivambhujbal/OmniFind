"""Migrations must apply to a database that already has data.

Every other test builds a fresh database, so the whole class of
"works on empty, fails on populated" bugs is invisible to them. One got
through: adding a NOT NULL column without a server default is accepted by an
empty `chunks` table and rejected by one with rows --

    sqlite3.OperationalError: Cannot add a NOT NULL column with default value NULL

-- which meant the sidecar failed to start against a real library while the
whole suite stayed green.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = REPO_ROOT / "sidecar"


def _alembic(data_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SIDECAR)
    env["FS_DATA_DIR"] = str(data_dir)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=SIDECAR,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def migration_dir(tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield data_dir
    shutil.rmtree(data_dir, ignore_errors=True)


def _revisions(data_dir: Path) -> list[str]:
    result = _alembic(data_dir, "history")
    assert result.returncode == 0, result.stderr
    # "abc123 -> def456, message" -- take the target of each step, oldest last.
    revisions = []
    for line in result.stdout.splitlines():
        if "->" in line:
            target = line.split("->", 1)[1].split(",", 1)[0].strip()
            revisions.append(target.replace(" (head)", "").strip())
    return list(reversed(revisions))


def test_every_migration_applies_to_a_populated_database(migration_dir: Path) -> None:
    """Step through revisions one at a time, with rows present throughout.

    Upgrading straight to head on an empty database — which is what the rest of
    the suite does — cannot catch a column default problem, because there are no
    existing rows for the new constraint to be violated by.
    """
    revisions = _revisions(migration_dir)
    assert len(revisions) >= 2, "expected more than one revision to step through"

    database = migration_dir / "filesearch.db"

    for index, revision in enumerate(revisions):
        result = _alembic(migration_dir, "upgrade", revision)
        assert (
            result.returncode == 0
        ), f"migration to {revision} failed against a populated database:\n{result.stderr}"

        # Seed rows once the first revision has created the tables, so every
        # LATER migration runs against a table that already has content.
        if index == 0:
            _seed(database)

        # And the rows survive each step -- a batch_alter_table on SQLite
        # rebuilds the table, so a botched migration can silently drop them.
        assert _chunk_count(database) == 2, f"rows lost migrating to {revision}"


def _seed(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute(
        "insert into files (id, sha256, original_name, stored_path, kind, mime_type,"
        " size_bytes, status, created_at, updated_at)"
        " values (1, 'a', 'x.pdf', 'aa/x.pdf', 'PDF', 'application/pdf', 1, 'READY',"
        " '2026-01-01', '2026-01-01')"
    )
    for chunk_id, text in ((1, "real prose about arctic terns"), (2, "2016/0012044 2016/0012045")):
        connection.execute(
            "insert into chunks (id, file_id, kind, ordinal, text, char_count,"
            " text_embedded, created_at)"
            " values (?, 1, 'TEXT', ?, ?, ?, 0, '2026-01-01')",
            (chunk_id, chunk_id, text, len(text)),
        )
    connection.commit()
    connection.close()


def _chunk_count(database: Path) -> int:
    if not database.exists():
        return 0
    connection = sqlite3.connect(database)
    try:
        return int(connection.execute("select count(*) from chunks").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        connection.close()


def test_existing_rows_get_a_usable_value_for_new_columns(migration_dir: Path) -> None:
    """A backfilled column must not leave old rows in a broken state."""
    revisions = _revisions(migration_dir)
    database = migration_dir / "filesearch.db"

    assert _alembic(migration_dir, "upgrade", revisions[0]).returncode == 0
    _seed(database)
    result = _alembic(migration_dir, "upgrade", "head")
    assert result.returncode == 0, result.stderr

    connection = sqlite3.connect(database)
    values = [row[0] for row in connection.execute("select searchable from chunks")]
    connection.close()

    assert values == [1, 1], "pre-existing chunks should default to searchable"
