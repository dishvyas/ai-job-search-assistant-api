"""add job descriptions with pgvector embeddings

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-17

Creates the job_descriptions table used by the RAG pipeline.
The embedding column uses pgvector's vector type so PostgreSQL can
perform efficient approximate nearest-neighbour (ANN) similarity searches
without loading all rows into application memory.

Notes
-----
- The pgvector extension must be present in the target PostgreSQL database.
  CREATE EXTENSION IF NOT EXISTS vector is idempotent and safe to run on
  every `alembic upgrade head`.
- vector(1536) matches the output dimension of text-embedding-3-small.
  If you switch embedding models you must also update this dimension and
  rebuild any vector indexes.
- The metadata column is JSON to allow arbitrary structured filters
  (role_type, seniority, tech_stack) without requiring schema migrations
  every time a new filter dimension is introduced.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable pgvector — idempotent, so running this on an existing database with
    # the extension already installed is safe. Required before any vector column
    # can be created.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "job_descriptions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("company", sa.Text(), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        # Full text is stored for display in API responses and for eval scoring.
        sa.Column("raw_text", sa.Text(), nullable=False),
        # Nullable JSON — callers can omit metadata if no structured filters are needed.
        sa.Column("metadata", sa.JSON(), nullable=True),
        # vector(1536) = text-embedding-3-small output dimension.
        # Nullable so a row can be inserted before embedding generation completes.
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_job_descriptions_id"),
        "job_descriptions",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_descriptions_id"), table_name="job_descriptions")
    op.drop_table("job_descriptions")
    # Do NOT drop the vector extension — other tables or external tools may use it.
