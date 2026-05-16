"""add background job status

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-16

Changes
-------
1. Add `status` column (TEXT NOT NULL, server_default='completed' so that
   all M5 rows — which already have full output — are treated as completed).
2. Add `error_message` column (TEXT, nullable).
3. Make all AI output columns and `provider_used` nullable via batch mode.
   SQLite does not support ALTER COLUMN natively; Alembic's batch mode
   recreates the table transparently. PostgreSQL handles the ALTER directly.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Use batch mode for full SQLite compatibility (ALTER COLUMN support).
    with op.batch_alter_table("application_tailoring_runs") as batch_op:
        # New workflow columns
        batch_op.add_column(
            sa.Column(
                "status",
                sa.Text(),
                nullable=False,
                server_default="completed",  # existing M5 rows are already complete
            )
        )
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))

        # Make AI output columns nullable (they start NULL for new pending rows)
        batch_op.alter_column("tailored_summary", existing_type=sa.Text(), nullable=True)
        batch_op.alter_column("tailored_bullets", existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column("cover_letter_draft", existing_type=sa.Text(), nullable=True)
        batch_op.alter_column(
            "application_question_answers", existing_type=sa.JSON(), nullable=True
        )
        batch_op.alter_column("recruiter_message_draft", existing_type=sa.Text(), nullable=True)
        batch_op.alter_column("fit_gap_analysis", existing_type=sa.Text(), nullable=True)
        batch_op.alter_column("interview_talking_points", existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column("provider_used", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("application_tailoring_runs") as batch_op:
        batch_op.alter_column("provider_used", existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("interview_talking_points", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("fit_gap_analysis", existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("recruiter_message_draft", existing_type=sa.Text(), nullable=False)
        batch_op.alter_column(
            "application_question_answers", existing_type=sa.JSON(), nullable=False
        )
        batch_op.alter_column("cover_letter_draft", existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("tailored_bullets", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("tailored_summary", existing_type=sa.Text(), nullable=False)
        batch_op.drop_column("error_message")
        batch_op.drop_column("status")
