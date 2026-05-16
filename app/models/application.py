from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ApplicationTailoringRun(Base):
    """
    Persisted record of a single application tailoring request + AI output.

    List fields (bullets, answers, talking points) are stored as JSON.
    provider_used tracks which LLM was called: "mock", "gemini", or "fallback-mock".
    """

    __tablename__ = "application_tailoring_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Inputs
    master_resume: Mapped[str] = mapped_column(Text, nullable=False)
    job_description: Mapped[str] = mapped_column(Text, nullable=False)
    company_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_preferences: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Outputs
    tailored_summary: Mapped[str] = mapped_column(Text, nullable=False)
    tailored_bullets: Mapped[list] = mapped_column(JSON, nullable=False)
    cover_letter_draft: Mapped[str] = mapped_column(Text, nullable=False)
    application_question_answers: Mapped[list] = mapped_column(JSON, nullable=False)
    recruiter_message_draft: Mapped[str] = mapped_column(Text, nullable=False)
    fit_gap_analysis: Mapped[str] = mapped_column(Text, nullable=False)
    interview_talking_points: Mapped[list] = mapped_column(JSON, nullable=False)

    # Metadata
    provider_used: Mapped[str] = mapped_column(Text, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
