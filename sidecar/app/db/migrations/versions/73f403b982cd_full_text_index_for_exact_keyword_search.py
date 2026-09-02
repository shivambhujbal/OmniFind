"""full-text index for exact keyword search

Dense embeddings cannot find an exact identifier: searching a reference number
scored the chunk containing it *below* unrelated documents, because the model
tokenises a long digit string into meaningless fragments. FTS5 indexes it as a
single token.

An external-content table is used so the text is not stored twice -- FTS5 reads
it from `chunks` and only keeps the index. Triggers keep the two in step,
including deletes, which matters because removing a file must remove its
passages from every index.

Revision ID: 73f403b982cd
Revises: 6498ef634333
Create Date: 2026-08-21 03:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "73f403b982cd"
down_revision: str | None = "6498ef634333"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # unicode61 keeps digit runs intact as single tokens, which is the whole
    # point: "42595078122" must be findable as itself.
    op.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )

    # Backfill everything already ingested, so an existing library becomes
    # searchable by keyword without a re-import.
    op.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")

    op.execute(
        """
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END
        """
    )
    # External-content tables need the old value handed back on delete, or the
    # index keeps rows that no longer exist and search returns dangling ids.
    op.execute(
        """
        CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
    op.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
    op.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
    op.execute("DROP TABLE IF EXISTS chunks_fts")
