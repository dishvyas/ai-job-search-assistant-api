"""add agent decision metadata to tailoring runs

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-27

Adds compact agent decision summary fields to completed runs so callers can
display route/review outcomes without exposing raw prompts or retrieved content.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "application_tailoring_runs",
        sa.Column("route_decision", sa.Text(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("review_notes", sa.Text(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("revision_needed", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("retrieved_context_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("artifact_context_count", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("application_tailoring_runs", "artifact_context_count")
    op.drop_column("application_tailoring_runs", "retrieved_context_count")
    op.drop_column("application_tailoring_runs", "revision_needed")
    op.drop_column("application_tailoring_runs", "review_notes")
    op.drop_column("application_tailoring_runs", "route_decision")
