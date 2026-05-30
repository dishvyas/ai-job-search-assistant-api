from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_trace import AgentTraceStep


def create_agent_trace_step(
    db: Session,
    run_id: int,
    step_name: str,
    status: str,
    input_summary: str | None = None,
    output_summary: str | None = None,
    provider_used: str | None = None,
    fallback_used: bool = False,
    latency_ms: int | None = None,
    estimated_input_tokens: int | None = None,
    estimated_output_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    error_message: str | None = None,
) -> AgentTraceStep:
    """Persist and return one agent workflow trace step."""
    step = AgentTraceStep(
        run_id=run_id,
        step_name=step_name,
        status=status,
        input_summary=input_summary,
        output_summary=output_summary,
        provider_used=provider_used,
        fallback_used=fallback_used,
        latency_ms=latency_ms,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        error_message=error_message,
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    return step


def get_agent_trace_steps(db: Session, run_id: int) -> list[AgentTraceStep]:
    """Return all trace steps for a run in creation order."""
    stmt = select(AgentTraceStep).where(AgentTraceStep.run_id == run_id).order_by(AgentTraceStep.id)
    return list(db.scalars(stmt))
