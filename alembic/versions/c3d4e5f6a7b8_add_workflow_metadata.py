"""add workflow metadata

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-17

Adds lightweight instrumentation columns to application_tailoring_runs:
  - started_at / completed_at  — task execution window
  - latency_ms                 — total background task duration
  - estimated_input_tokens     — word-count approximation of prompt tokens
  - estimated_output_tokens    — word-count approximation of output tokens
  - estimated_cost_usd         — rough provider cost estimate
  - generation_attempts        — how many LLM calls were made (1 normal, 2 if fallback)

All new columns are either nullable (timing/token fields) or have a server
default of 0 (generation_attempts), so the migration is safe on databases
that already have rows from earlier milestones.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "application_tailoring_runs",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("estimated_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("estimated_output_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
    )
    op.add_column(
        "application_tailoring_runs",
        sa.Column(
            "generation_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",  # existing rows default to 0
        ),
    )


def downgrade() -> None:
    op.drop_column("application_tailoring_runs", "generation_attempts")
    op.drop_column("application_tailoring_runs", "estimated_cost_usd")
    op.drop_column("application_tailoring_runs", "estimated_output_tokens")
    op.drop_column("application_tailoring_runs", "estimated_input_tokens")
    op.drop_column("application_tailoring_runs", "latency_ms")
    op.drop_column("application_tailoring_runs", "completed_at")
    op.drop_column("application_tailoring_runs", "started_at")
