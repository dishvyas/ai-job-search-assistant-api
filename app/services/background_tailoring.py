import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.cost_estimation import estimate_generation_cost
from app.llm.token_estimation import estimate_input_tokens, estimate_output_tokens
from app.models.run_status import RunStatus
from app.prompts.tailoring import build_tailoring_prompt
from app.repositories.application_runs import (
    get_application_tailoring_run,
    save_completed_run,
    update_run_status,
)
from app.schemas.application import ApplicationTailorRequest
from app.services.agentic_tailoring import run_agentic_workflow
from app.services.application_tailoring import _get_llm_output

_SUPPORTED_MODES = {"single_step", "agentic"}


def process_tailoring_job(run_id: int, db: Session) -> None:
    """
    Background task: generate tailoring output and track workflow metadata.

    Workflow mode is read from settings.workflow_mode:
      single_step — one LLM call (fast, cheap, default)
      agentic     — four-stage LangGraph workflow (more reasoning steps)

    An unsupported workflow_mode raises ValueError immediately, which is
    caught by the exception handler and stored as a failed run.
    """
    run = get_application_tailoring_run(db, run_id)
    if run is None:
        return

    update_run_status(db, run, RunStatus.PROCESSING.value)
    started_at = datetime.now(UTC)
    generation_attempts = 0

    try:
        request = ApplicationTailorRequest(
            master_resume=run.master_resume,
            job_description=run.job_description,
            company_info=run.company_info,
            user_preferences=run.user_preferences,
        )

        mode = settings.workflow_mode.lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported workflow_mode: {mode!r}. Must be one of: {sorted(_SUPPORTED_MODES)}"
            )

        # Build the single-step prompt; used for input token estimation in both modes.
        prompt = build_tailoring_prompt(request)

        if mode == "single_step":
            generation_attempts = 1
            llm_output, provider_used, used_fallback = _get_llm_output(prompt)
            if used_fallback:
                generation_attempts = 2

        else:  # agentic
            # 4 stages × 1 call each; double if any stage fell back (approximate).
            llm_output, provider_used, used_fallback = run_agentic_workflow(request)
            generation_attempts = 4 + (4 if used_fallback else 0)

        completed_at = datetime.now(UTC)
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Token/cost estimation.
        # For agentic mode this approximates the final stage only; intermediate
        # stage tokens are not counted individually (acceptable approximation).
        output_text = json.dumps(llm_output.model_dump())
        input_tokens = estimate_input_tokens(prompt)
        output_tokens = estimate_output_tokens(output_text)
        cost_usd = estimate_generation_cost(input_tokens, output_tokens, provider_used)

        save_completed_run(
            db=db,
            run=run,
            llm_output=llm_output,
            provider_used=provider_used,
            fallback_used=used_fallback,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=cost_usd,
            generation_attempts=generation_attempts,
        )

    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(UTC)
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        update_run_status(
            db,
            run,
            RunStatus.FAILED.value,
            error_message=str(exc),
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            generation_attempts=generation_attempts,
        )
