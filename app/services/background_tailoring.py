import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

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
from app.services.application_tailoring import _get_llm_output


def process_tailoring_job(run_id: int, db: Session) -> None:
    """
    Background task: generate tailoring output and track workflow metadata.

    Flow
    ----
    1. Load the run row.
    2. Transition to processing; record started_at.
    3. Reconstruct the original request and build the LLM prompt.
    4. Call _get_llm_output (includes provider fallback logic).
    5. Estimate tokens and cost from prompt + raw output.
    6. Persist output, metadata, and transition to completed.

    On any exception the run is marked failed and timing metadata is still
    saved so partial observability data is available for debugging.

    generation_attempts
    -------------------
    Set to 1 before calling _get_llm_output. Incremented to 2 if fallback
    occurred (primary provider failed, mock was called as backup).
    """
    run = get_application_tailoring_run(db, run_id)
    if run is None:
        # Should never happen — run was just created by the route.
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
        prompt = build_tailoring_prompt(request)

        generation_attempts = 1
        llm_output, provider_used, used_fallback = _get_llm_output(prompt)
        if used_fallback:
            generation_attempts = 2

        completed_at = datetime.now(UTC)
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Estimate tokens from the prompt (input) and serialised output (output).
        # Using json.dumps(model_dump()) gives a string comparable in size to the
        # raw LLM response, which is what we'd ideally measure.
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
