"""
Four-stage agentic tailoring workflow implemented with LangGraph.

Stages
------
1. analyze_resume        — extract skills, experience, strengths from the resume
2. analyze_jd            — extract requirements, responsibilities, role focus from the JD
3. analyze_fit_gap       — identify fit points, gaps, and positioning strategy
4. compose_final_output  — produce the full TailoringLLMOutput using all prior analyses

Each node makes one LLM call (with provider fallback) and parses the response
into the stage's Pydantic schema. Failures at any node propagate as exceptions
and are caught by the background task's error handler.
"""

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.core.config import settings
from app.llm.exceptions import LLMOutputParsingError, LLMProviderError
from app.llm.factory import get_llm_provider
from app.llm.mock import MockLLMProvider
from app.prompts.agentic_tailoring import (
    build_final_tailoring_prompt,
    build_fit_gap_prompt,
    build_jd_analysis_prompt,
    build_resume_analysis_prompt,
)
from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest
from app.schemas.llm_output import TailoringLLMOutput

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class AgenticTailoringState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    request: ApplicationTailorRequest
    resume_analysis: ResumeAnalysis | None
    jd_analysis: JobDescriptionAnalysis | None
    fit_gap: FitGapAnalysis | None
    final_output: TailoringLLMOutput | None
    provider_used: str  # reflects the most-recent provider; "fallback-mock" if any stage fell back
    fallback_used: bool  # True if any stage used the fallback mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_and_parse(prompt: str, model_class: type) -> tuple[object, str, bool]:
    """
    Call the configured LLM provider and parse the response into model_class.

    Falls back to MockLLMProvider if the configured provider raises LLMProviderError
    or if the response can't be parsed into the expected schema.

    Returns (parsed_instance, provider_used, fallback_used).
    """
    configured = settings.llm_provider.lower()

    try:
        raw = get_llm_provider().generate_text(prompt)
        return _parse(raw, model_class), configured, False
    except (LLMProviderError, LLMOutputParsingError):
        pass  # fall through to mock

    raw = MockLLMProvider().generate_text(prompt)
    return _parse(raw, model_class), "fallback-mock", True


def _parse(raw: str, model_class: type) -> object:
    """Parse a JSON string into the given Pydantic model.

    Raises LLMOutputParsingError on failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMOutputParsingError(
            f"Agent stage output is not valid JSON: {e}. Preview: {raw[:200]!r}"
        ) from e
    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        raise LLMOutputParsingError(
            f"Agent stage output does not match {model_class.__name__}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _analyze_resume(state: AgenticTailoringState) -> dict:
    prompt = build_resume_analysis_prompt(state["request"])
    result, provider, fallback = _call_and_parse(prompt, ResumeAnalysis)
    return {
        "resume_analysis": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _analyze_jd(state: AgenticTailoringState) -> dict:
    prompt = build_jd_analysis_prompt(state["request"])
    result, provider, fallback = _call_and_parse(prompt, JobDescriptionAnalysis)
    return {
        "jd_analysis": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _analyze_fit_gap(state: AgenticTailoringState) -> dict:
    prompt = build_fit_gap_prompt(state["request"], state["resume_analysis"], state["jd_analysis"])
    result, provider, fallback = _call_and_parse(prompt, FitGapAnalysis)
    return {
        "fit_gap": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _compose_final(state: AgenticTailoringState) -> dict:
    prompt = build_final_tailoring_prompt(
        state["request"],
        state["resume_analysis"],
        state["jd_analysis"],
        state["fit_gap"],
    )
    result, provider, fallback = _call_and_parse(prompt, TailoringLLMOutput)
    return {
        "final_output": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> object:
    graph: StateGraph = StateGraph(AgenticTailoringState)

    graph.add_node("analyze_resume", _analyze_resume)
    graph.add_node("analyze_jd", _analyze_jd)
    graph.add_node("analyze_fit_gap", _analyze_fit_gap)
    graph.add_node("compose_final", _compose_final)

    graph.add_edge(START, "analyze_resume")
    graph.add_edge("analyze_resume", "analyze_jd")
    graph.add_edge("analyze_jd", "analyze_fit_gap")
    graph.add_edge("analyze_fit_gap", "compose_final")
    graph.add_edge("compose_final", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_agentic_workflow(
    request: ApplicationTailorRequest,
) -> tuple[TailoringLLMOutput, str, bool]:
    """
    Execute the four-stage agentic workflow and return the final tailoring output.

    Returns the same (output, provider_used, fallback_used) tuple as _get_llm_output
    so the background task can treat both modes uniformly.
    """
    graph = _build_graph()

    initial_state: AgenticTailoringState = {
        "request": request,
        "resume_analysis": None,
        "jd_analysis": None,
        "fit_gap": None,
        "final_output": None,
        "provider_used": settings.llm_provider.lower(),
        "fallback_used": False,
    }

    final_state = graph.invoke(initial_state)

    return (
        final_state["final_output"],
        final_state["provider_used"],
        final_state["fallback_used"],
    )
