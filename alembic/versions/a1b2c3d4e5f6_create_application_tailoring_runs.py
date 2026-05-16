"""create application tailoring runs

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-05-16

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "application_tailoring_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("master_resume", sa.Text(), nullable=False),
        sa.Column("job_description", sa.Text(), nullable=False),
        sa.Column("company_info", sa.Text(), nullable=True),
        sa.Column("user_preferences", sa.Text(), nullable=True),
        sa.Column("tailored_summary", sa.Text(), nullable=False),
        sa.Column("tailored_bullets", sa.JSON(), nullable=False),
        sa.Column("cover_letter_draft", sa.Text(), nullable=False),
        sa.Column("application_question_answers", sa.JSON(), nullable=False),
        sa.Column("recruiter_message_draft", sa.Text(), nullable=False),
        sa.Column("fit_gap_analysis", sa.Text(), nullable=False),
        sa.Column("interview_talking_points", sa.JSON(), nullable=False),
        sa.Column("provider_used", sa.Text(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_application_tailoring_runs_id"),
        "application_tailoring_runs",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_application_tailoring_runs_id"),
        table_name="application_tailoring_runs",
    )
    op.drop_table("application_tailoring_runs")
