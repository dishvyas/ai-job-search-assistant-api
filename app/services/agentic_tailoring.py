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

# LangGraph is used here as a lightweight sequential orchestrator — not for its
# conditional branching or parallel execution features, but because it provides a
# clean typed-state model where each node receives the full prior context and
# returns only the fields it modifies. This makes the data flow auditable and
# testable at the node level rather than inside one large function.
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
    # Intermediate analyses start as None and are populated stage-by-stage.
    # None values make it explicit that a field hasn't been filled yet vs being empty.
    resume_analysis: ResumeAnalysis | None
    jd_analysis: JobDescriptionAnalysis | None
    fit_gap: FitGapAnalysis | None
    final_output: TailoringLLMOutput | None
    provider_used: str  # reflects the most-recent provider; "fallback-mock" if any stage fell back
    fallback_used: bool  # True if any stage used the fallback mock; OR-ed across all stages


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
        pass  # fall through to mock — same fallback logic as the single-step path

    raw = MockLLMProvider().generate_text(prompt)
    # If mock raises here it is a code bug, not a runtime condition — let it propagate.
    return _parse(raw, model_class), "fallback-mock", True


def _parse(raw: str, model_class: type) -> object:
    """Parse a JSON string into the given Pydantic model.

    Raises LLMOutputParsingError on failure.
    """
    # Two separate try/except blocks so the error message distinguishes between
    # "not valid JSON at all" and "valid JSON but wrong shape".
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

# Each node function follows the same pattern: build a prompt from the current state,
# call the LLM (with fallback), parse into the stage's schema, and return only the
# state keys that this node owns. LangGraph merges the returned dict into the state.


def _analyze_resume(state: AgenticTailoringState) -> dict:
    prompt = build_resume_analysis_prompt(state["request"])
    result, provider, fallback = _call_and_parse(prompt, ResumeAnalysis)
    return {
        "resume_analysis": result,
        "provider_used": provider,
        # OR with prior stages so fallback_used stays True once any stage has fallen back.
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
    # Receives the outputs of both prior stages — cross-referencing them is the
    # core value of this stage versus asking one prompt to do everything.
    prompt = build_fit_gap_prompt(state["request"], state["resume_analysis"], state["jd_analysis"])
    result, provider, fallback = _call_and_parse(prompt, FitGapAnalysis)
    return {
        "fit_gap": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _compose_final(state: AgenticTailoringState) -> dict:
    # All three prior analyses feed this stage, so the final composition prompt
    # is richer and more targeted than a single-step prompt over raw text.
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
    # Graph is built fresh on each call rather than at module load time so that
    # test monkeypatching of settings and provider functions takes effect inside
    # each invocation without needing to reload the module.
    graph: StateGraph = StateGraph(AgenticTailoringState)

    graph.add_node("analyze_resume", _analyze_resume)
    graph.add_node("analyze_jd", _analyze_jd)
    graph.add_node("analyze_fit_gap", _analyze_fit_gap)
    graph.add_node("compose_final", _compose_final)

    # Strictly sequential edges — each stage depends on all prior stages' outputs.
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
        # Set provider_used to the configured value upfront; individual nodes may
        # overwrite it to "fallback-mock" if their call fails.
        "provider_used": settings.llm_provider.lower(),
        "fallback_used": False,
    }

    final_state = graph.invoke(initial_state)

    return (
        final_state["final_output"],
        final_state["provider_used"],
        final_state["fallback_used"],
    )
