"""Bring the database schema up to date at startup.

A desktop app has no operator to run ``alembic upgrade head`` after an update,
so the sidecar does it itself on launch. Alembic (rather than
``create_all``) because an installed app carries the user's real data across
versions -- adding a column in v0.3 must not mean losing their library.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# <root>/sidecar/app/db/bootstrap.py -> <root>/sidecar
SIDECAR_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config() -> Config:
    config = Config(str(SIDECAR_ROOT / "alembic.ini"))
    # Absolute, because the working directory differs between dev (repo root)
    # and the packaged app (wherever the exe was launched from).
    config.set_main_option("script_location", str(SIDECAR_ROOT / "app" / "db" / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.db_url)
    # Keep our logging handlers; see the comment in migrations/env.py.
    config.attributes["configure_logger"] = False
    return config


def upgrade_to_head() -> None:
    settings.ensure_dirs()
    log.info("applying database migrations")
    command.upgrade(_alembic_config(), "head")
    log.info("database schema is current")


def current_revision() -> str | None:
    from alembic.runtime.migration import MigrationContext

    from app.db.session import get_engine

    with get_engine().connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()
