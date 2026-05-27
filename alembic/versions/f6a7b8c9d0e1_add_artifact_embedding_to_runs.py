"""add artifact embedding to tailoring runs

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-27

Adds an optional embedding column for completed tailored artifacts so the app can
retrieve prior generated outputs as reference context for future tailoring.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "application_tailoring_runs",
        sa.Column("artifact_embedding", Vector(1536), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("application_tailoring_runs", "artifact_embedding")
