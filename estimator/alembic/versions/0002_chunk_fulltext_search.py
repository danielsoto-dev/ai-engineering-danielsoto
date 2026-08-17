"""Add a Spanish full-text search vector to chunks, with a GIN index.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-17

The column is GENERATED ALWAYS, so Postgres keeps it in sync with ``content``
on every insert and update — no trigger to maintain. The ``spanish`` text
search configuration applies Spanish stemming and stop words, which matters
because the budget corpus is written in Spanish.
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
