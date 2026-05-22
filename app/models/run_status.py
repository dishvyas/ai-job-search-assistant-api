# StrEnum so that RunStatus.PENDING == "pending" is True without calling .value.
# This means status can be stored and compared as a plain string in SQLAlchemy
# without a custom TypeDecorator or explicit .value calls everywhere.
from enum import StrEnum


class RunStatus(StrEnum):
    """
    Lifecycle states for an ApplicationTailoringRun.

    pending    — row created, background task not yet started
    processing — background task is actively generating output
    completed  — generation finished; all output fields are populated
    failed     — generation failed; error_message describes what went wrong
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
