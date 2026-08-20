"""Alembic environment.

The database URL comes from ``app.config`` rather than ``alembic.ini`` so that
migrations always target the same file the app uses, including when the data
directory has been relocated with ``FS_DATA_DIR``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.db_url)

# `fileConfig` calls `logging.config.fileConfig`, which by default *removes*
# every existing handler on the root logger. When migrations run in-process at
# sidecar startup that silently destroys the app's stderr and file handlers, and
# the app logs nothing for the rest of its life. So only configure logging when
# the alembic CLI is driving; `bootstrap.upgrade_to_head` sets this attribute to
# opt out.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place; batch mode rewrites the
            # table instead, which is the only way ALTER works here at all.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
