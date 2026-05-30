"""add fallback reason to tailoring runs

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-05-30

Adds a nullable fallback_reason field so completed degraded-success runs can
explain why mock fallback was used without overloading failed-run error_message.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: str | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "application_tailoring_runs",
        sa.Column("fallback_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("application_tailoring_runs", "fallback_reason")
