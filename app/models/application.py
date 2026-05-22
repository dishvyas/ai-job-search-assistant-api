# Single ORM model for the entire job application tailoring workflow.
# One table per workflow run keeps queries simple and avoids joins; a separate
# "outputs" table would add complexity with no benefit at this scale.
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, Text
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
    # Optional context fields — not all callers have company info or preferences.
    company_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_preferences: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Workflow status ---
    status: Mapped[str] = mapped_column(Text, nullable=False, default=RunStatus.PENDING.value)
    # error_message is only set on failed runs; null on all other statuses.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- AI outputs (nullable until status=completed) ---
    # JSON columns avoid a separate bullets/points table — list-valued fields that
    # are always read and written together don't benefit from normalisation here.
    tailored_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tailored_bullets: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cover_letter_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    application_question_answers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recruiter_message_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    fit_gap_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    interview_talking_points: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # --- Metadata (set after processing) ---
    # provider_used is nullable because it is only known after the task runs.
    provider_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # lambda default instead of server_default so the value is timezone-aware Python
    # datetime; server_default CURRENT_TIMESTAMP is naive in SQLite.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # --- Workflow instrumentation (populated by background task) ---
    # All timing and token fields are nullable — a failed run before the LLM call
    # may not have all of these, but started_at/completed_at/latency_ms are still
    # written on failure so debugging can answer "how long did it run before failing?".
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # default=0 so the column is queryable even before a task runs.
    generation_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
