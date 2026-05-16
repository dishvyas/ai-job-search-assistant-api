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
