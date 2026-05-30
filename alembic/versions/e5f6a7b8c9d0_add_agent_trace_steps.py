"""add agent trace steps

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-27

Adds a lightweight per-stage trace table for the agentic workflow so each
completed run can expose what happened at each node without storing raw prompts
or the full resume / job description text.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_trace_steps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("step_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column("output_summary", sa.Text(), nullable=True),
        sa.Column("provider_used", sa.Text(), nullable=True),
        sa.Column("fallback_used", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["run_id"], ["application_tailoring_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_trace_steps_id"), "agent_trace_steps", ["id"], unique=False)
    op.create_index(
        op.f("ix_agent_trace_steps_run_id"),
        "agent_trace_steps",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_trace_steps_run_id"), table_name="agent_trace_steps")
    op.drop_index(op.f("ix_agent_trace_steps_id"), table_name="agent_trace_steps")
    op.drop_table("agent_trace_steps")
