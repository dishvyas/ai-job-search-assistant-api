from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.run_status import RunStatus


class ApplicationTailoringRun(Base):
    """
    Persisted record of a single application tailoring request + AI output.

    Lifecycle
    ---------
    Rows are created immediately (status=pending) when a request is accepted.
    A background task then processes the request and updates the row through
    the full lifecycle: pending → processing → completed | failed.

    Output fields (tailored_summary, tailored_bullets, etc.) are nullable
    because they are not populated until the background task completes.
    """

    __tablename__ = "application_tailoring_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # --- Inputs (set at creation time) ---
    master_resume: Mapped[str] = mapped_column(Text, nullable=False)
    job_description: Mapped[str] = mapped_column(Text, nullable=False)
    company_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_preferences: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Workflow status ---
    status: Mapped[str] = mapped_column(Text, nullable=False, default=RunStatus.PENDING.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- AI outputs (nullable until status=completed) ---
    tailored_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tailored_bullets: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cover_letter_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    application_question_answers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recruiter_message_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    fit_gap_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    interview_talking_points: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # --- Metadata (set after processing) ---
    provider_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
