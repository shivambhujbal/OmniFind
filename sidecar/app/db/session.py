"""SQLite engine and session management.

One process, one user, one database file. The pragmas below matter more than
usual here because the ingestion worker writes from a background thread while
the HTTP handlers read on the request thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _configure_sqlite(dbapi_connection: object, _record: object) -> None:
    """Pragmas applied to every new connection."""
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    # WAL lets the ingestion worker write while the API reads, instead of the
    # readers getting "database is locked" during a long ingest.
    cursor.execute("PRAGMA journal_mode=WAL")
    # SQLite ignores foreign keys unless asked, which would silently defeat the
    # ON DELETE CASCADE on chunks/assets.
    cursor.execute("PRAGMA foreign_keys=ON")
    # Wait rather than fail if a write lock is briefly held.
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings.ensure_dirs()
        _engine = create_engine(
            settings.db_url,
            # The background worker runs in a different thread from the request
            # handlers; each still gets its own Session from the pool.
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            future=True,
        )
        event.listen(_engine, "connect", _configure_sqlite)
        log.info("database: %s", settings.db_path)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background work and scripts."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency. Commits are the caller's responsibility."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine. Used by tests that relocate the data directory."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
