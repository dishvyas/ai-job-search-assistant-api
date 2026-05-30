"""add agent trace cost metadata

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-05-30

Adds optional approximate per-stage token and cost metadata to agent trace
rows so multi-step agent workflows can expose which stages were expensive
without storing raw prompts or full model outputs.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_trace_steps",
        sa.Column("estimated_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "agent_trace_steps",
        sa.Column("estimated_output_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "agent_trace_steps",
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_trace_steps", "estimated_cost_usd")
    op.drop_column("agent_trace_steps", "estimated_output_tokens")
    op.drop_column("agent_trace_steps", "estimated_input_tokens")
