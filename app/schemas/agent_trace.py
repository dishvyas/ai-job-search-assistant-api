from datetime import datetime

from pydantic import BaseModel


class AgentTraceStepResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    run_id: int
    step_name: str
    status: str
    input_summary: str | None = None
    output_summary: str | None = None
    provider_used: str | None = None
    fallback_used: bool
    latency_ms: int | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error_message: str | None = None
    created_at: datetime


class AgentTraceResponse(BaseModel):
    run_id: int
    steps: list[AgentTraceStepResponse]
