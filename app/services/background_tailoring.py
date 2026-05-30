# Orchestrates the full background job lifecycle: status transitions, LLM dispatch,
# and metadata recording. This is the single function registered as a FastAPI
# BackgroundTask — it runs in-process after the HTTP response is sent, so there is
# no Redis, Celery, or external broker required at this stage.
import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.cost_estimation import estimate_generation_cost
from app.llm.token_estimation import estimate_input_tokens, estimate_output_tokens
from app.models.run_status import RunStatus
from app.prompts.tailoring import build_tailoring_prompt
from app.rag.artifacts import retrieve_similar_artifacts, store_artifact_embedding_for_run
from app.repositories.application_runs import (
    get_application_tailoring_run,
    save_completed_run,
    update_run_status,
)
from app.schemas.application import ApplicationTailorRequest
from app.services.agentic_tailoring import run_agentic_workflow
from app.services.application_tailoring import _get_llm_output

# Explicit allowlist — an unsupported mode fails fast with a clear error rather than
# silently doing nothing or falling through to unexpected behaviour.
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
    # Guard against the (unlikely) race where the row was deleted after the task was enqueued.
    if run is None:
        return

    # Mark processing before the LLM call so the GET endpoint reflects real-time state.
    update_run_status(db, run, RunStatus.PROCESSING.value)
    started_at = datetime.now(UTC)
    # Initialised to 0 so that a failure before any LLM call is recorded as 0 attempts.
    generation_attempts = 0

    try:
        # Reconstruct a validated Pydantic request from the persisted ORM row so that
        # service functions receive the same type as the original HTTP handler.
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

        # Optionally retrieve RAG context to enrich the tailoring prompt.
        # When rag_enabled=False (the default), this block is skipped entirely and
        # the existing single-step/agentic behaviour is completely unchanged.
        rag_context = None
        if settings.rag_enabled:
            try:
                from app.rag.retrieve import retrieve_relevant_jobs

                retrieved = retrieve_relevant_jobs(db, query=request.job_description)
                # retrieve_relevant_jobs returns (jd, score) tuples; we only need the jd.
                rag_context = [jd for jd, _score in retrieved]
            except Exception:  # noqa: BLE001
                rag_context = None

        artifact_context = None
        if settings.rag_enabled and settings.artifact_retrieval_enabled:
            artifact_context = retrieve_similar_artifacts(db, query=request.job_description)

        # Build the single-step prompt; used for input token estimation in both modes.
        # In agentic mode the prompt approximates only the final stage — acceptable
        # because intermediate stage prompts are similar in size to the full prompt.
        prompt_kwargs = {"rag_context": rag_context}
        if artifact_context:
            prompt_kwargs["artifact_context"] = artifact_context
        prompt = build_tailoring_prompt(request, **prompt_kwargs)

        if mode == "single_step":
            generation_attempts = 1
            llm_output, provider_used, used_fallback = _get_llm_output(prompt)
            agent_metadata = None
            # Each fallback = one additional attempt (primary failed, mock succeeded).
            if used_fallback:
                generation_attempts = 2

        else:  # agentic
            # Pass the db session so the retrieve_context node can perform RAG tool use.
            # If rag_enabled=False, the node skips retrieval and db is unused.
            # Generation attempt accounting: 4 analysis/composition stages + optional
            # revision (absorbed into approximation); doubled if any stage fell back.
            # This intentionally approximates rather than tracking every possible path.
            llm_output, provider_used, used_fallback, agent_metadata = run_agentic_workflow(
                request,
                db=db,
                run_id=run.id,
                artifact_context=artifact_context,
            )
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
            agent_metadata=agent_metadata,
        )
        try:
            store_artifact_embedding_for_run(db, run)
        except Exception:  # noqa: BLE001
            db.rollback()

    except Exception as exc:  # noqa: BLE001
        # Broad catch is intentional — any unhandled exception (config error, DB error,
        # LLM parse failure) must be recorded as a failed run, never silently swallowed.
        completed_at = datetime.now(UTC)
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        # Write timing even on failure so debugging can answer "how long before it failed?".
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
