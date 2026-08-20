"""Regression: startup migrations must not silence application logging.

Alembic's ``fileConfig`` removes existing root handlers by default. When
migrations run in-process at sidecar startup, that wiped the stderr and rotating
file handlers and the app logged nothing for the rest of the session -- on a
desktop app whose only diagnostics are that log file.
"""

from __future__ import annotations

import logging


def test_handlers_survive_upgrade_to_head(_clean_test_data_dir: None) -> None:
    from app.db import bootstrap
    from app.logging_conf import configure_logging

    configure_logging()
    before = list(logging.getLogger().handlers)
    assert before, "precondition: logging is configured"

    bootstrap.upgrade_to_head()

    after = logging.getLogger().handlers
    assert after, "alembic removed every root handler"
    for handler in before:
        assert handler in after, f"alembic removed {handler!r}"
