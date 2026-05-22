"""
Tool-using agentic tailoring workflow (M10 upgrade of the M8 four-stage workflow).

Stages
------
1. retrieve_context     — fetch relevant job descriptions via RAG (tool use)
2. analyze_resume       — extract skills, experience, strengths
3. analyze_jd           — extract requirements, responsibilities, role focus
4. analyze_fit_gap      — identify fit points, gaps, positioning strategy
5. decide_route         — deterministic routing: proceed / needs_more_context / low_fit_warning
6. compose_final        — produce TailoringLLMOutput using all prior context
7. review_output        — deterministic quality gate: checks required sections are present
8. revise_output        — optional single LLM call to fix incomplete output (max once)

The addition of retrieve_context shows how agents use tools: the workflow queries an
external store (the RAG DB), incorporates the results into later prompts, and continues
without human intervention. The routing and review nodes demonstrate controlled
decision-making inside an agent graph without requiring extra LLM calls.
"""

# LangGraph provides typed-state orchestration: each node receives the full prior
# context and returns only the fields it mutates. This is cleaner than passing
# everything as function arguments and easier to trace/debug than a single large function.
import json
from typing import Any, TypedDict

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
    build_revision_prompt,
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
    # db is Session | None — held in state so the retrieve_context node can call
    # the RAG retrieval tool without needing to import the session from outside the graph.
    # TypedDict uses Any here because Session is not serialisable; we never persist state.
    db: Any

    # M8 intermediate analyses — populated stage-by-stage; None until that stage runs.
    resume_analysis: ResumeAnalysis | None
    jd_analysis: JobDescriptionAnalysis | None
    fit_gap: FitGapAnalysis | None
    final_output: TailoringLLMOutput | None

    # M10 additions
    # retrieved_context: text snippets from RAG retrieval; empty list when unavailable.
    # Stored as plain strings (not JobDescription objects) so state remains lightweight.
    retrieved_context: list[str]
    # route_decision: advisory label set by decide_route; does not hard-gate later stages.
    # Values: "proceed_to_tailoring" | "needs_more_context" | "low_fit_warning"
    route_decision: str
    # review_notes: human-readable outcome from the quality review node.
    review_notes: str | None
    # revision_needed: True if the review node found the output incomplete.
    revision_needed: bool

    # Tracking fields — reflect the most-recent provider and whether any stage fell back.
    provider_used: str
    fallback_used: bool


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

# Each node follows the same contract: read from state, do work, return only the
# fields it owns. LangGraph merges returned dicts into the state automatically.


def _retrieve_context(state: AgenticTailoringState) -> dict:
    """
    Tool-use node: retrieve relevant job descriptions from the RAG store.

    This is the core "tool use" pattern — the agent queries an external data source
    (the vector DB) and incorporates the results into its working context before
    performing any reasoning. If retrieval is disabled, unavailable, or fails,
    the workflow continues gracefully with an empty context list.
    """
    # Guard: skip if RAG is disabled or no DB session was provided.
    # This preserves identical behaviour for all callers that don't pass a db session,
    # including all pre-M10 tests.
    if not settings.rag_enabled or state.get("db") is None:
        return {"retrieved_context": []}

    try:
        # Lazy import keeps pgvector optional — it only needs to be installed when RAG
        # is actually enabled. Module-level import would force the dependency for everyone.
        from app.rag.retrieve import retrieve_relevant_jobs

        results = retrieve_relevant_jobs(state["db"], query=state["request"].job_description)
        # Store raw text snippets (not ORM objects) so state stays serialisable and lightweight.
        # Truncate each snippet to 300 chars — enough semantic signal without blowing token budget.
        snippets = [(jd.raw_text or "")[:300].strip() for jd, _ in results]
        return {"retrieved_context": snippets}
    except Exception:  # noqa: BLE001
        # Retrieval failure must not abort the whole workflow — context is optional enrichment.
        # Silently degrade to empty context rather than propagating the exception.
        return {"retrieved_context": []}


def _analyze_resume(state: AgenticTailoringState) -> dict:
    # Resume analysis doesn't use retrieved context — the LLM should extract facts
    # about the candidate from the resume alone, unbiased by external context.
    prompt = build_resume_analysis_prompt(state["request"])
    result, provider, fallback = _call_and_parse(prompt, ResumeAnalysis)
    return {
        "resume_analysis": result,
        "provider_used": provider,
        # OR with prior stages so fallback_used stays True once any stage has fallen back.
        "fallback_used": state["fallback_used"] or fallback,
    }


def _analyze_jd(state: AgenticTailoringState) -> dict:
    # JD analysis also doesn't use retrieved context — same reasoning as resume analysis.
    prompt = build_jd_analysis_prompt(state["request"])
    result, provider, fallback = _call_and_parse(prompt, JobDescriptionAnalysis)
    return {
        "jd_analysis": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _analyze_fit_gap(state: AgenticTailoringState) -> dict:
    # Receives outputs of both prior stages + retrieved context. The retrieved context
    # helps calibrate gap severity — e.g., if similar real roles also require Kubernetes,
    # that gap is more material than if the retrieved roles don't mention it.
    prompt = build_fit_gap_prompt(
        state["request"],
        state["resume_analysis"],
        state["jd_analysis"],
        retrieved_context=state["retrieved_context"] or None,
    )
    result, provider, fallback = _call_and_parse(prompt, FitGapAnalysis)
    return {
        "fit_gap": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _decide_route(state: AgenticTailoringState) -> dict:
    """
    Deterministic routing node — no LLM call needed.

    Route decision logic:
    - needs_more_context: no retrieved context AND no company_info — the LLM has
      less to work with for company-specific tailoring; surface this as a signal.
    - low_fit_warning: gap_points outnumber fit_points — the candidate may be
      underqualified; surface this so downstream callers can act on it.
    - proceed_to_tailoring: enough context present, fit is reasonable.

    The route is advisory metadata only — composition always proceeds regardless.
    Hard-gating (skipping composition for low-fit) would require product-level
    decisions that don't belong in a generic tailoring workflow.
    """
    fit_gap = state["fit_gap"]
    retrieved_context = state["retrieved_context"]
    request = state["request"]

    if not retrieved_context and not request.company_info:
        # No external context of any kind — flag it but continue with what we have.
        route = "needs_more_context"
    elif fit_gap and len(fit_gap.gap_points) > len(fit_gap.fit_points):
        # More gaps than fit points signals a weak match; proceed but warn the caller.
        route = "low_fit_warning"
    else:
        route = "proceed_to_tailoring"

    return {"route_decision": route}


def _compose_final(state: AgenticTailoringState) -> dict:
    # All three prior analyses + retrieved context feed the final composition, making
    # the prompt richer and more targeted than a single-step call over raw text.
    prompt = build_final_tailoring_prompt(
        state["request"],
        state["resume_analysis"],
        state["jd_analysis"],
        state["fit_gap"],
        retrieved_context=state["retrieved_context"] or None,
    )
    result, provider, fallback = _call_and_parse(prompt, TailoringLLMOutput)
    return {
        "final_output": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


def _review_output(state: AgenticTailoringState) -> dict:
    """
    Deterministic quality-gate node — no LLM call.

    Checks whether the final output contains all required, non-empty sections.
    Deterministic review is intentional: it's cheaper (no LLM token spend), faster,
    and fully auditable. A real production system would layer in semantic review
    (LLM-as-judge), but for a portfolio project the structural check demonstrates
    the critique pattern without the cost.
    """
    output = state["final_output"]
    issues: list[str] = []

    if not output or not output.tailored_summary:
        issues.append("tailored_summary is empty")
    if not output or not output.tailored_bullets:
        issues.append("tailored_bullets is empty")
    if not output or not output.interview_talking_points:
        issues.append("interview_talking_points is empty")

    if issues:
        notes = "Output incomplete — " + "; ".join(issues) + "."
        return {"revision_needed": True, "review_notes": notes}

    return {
        "revision_needed": False,
        "review_notes": "Review passed: all required sections are present.",
    }


def _revise_output(state: AgenticTailoringState) -> dict:
    """
    Optional correction node — runs at most once.

    Called only when review_output sets revision_needed=True. Uses a single LLM call
    to produce a corrected TailoringLLMOutput, guided by the specific review notes.
    After this node, the graph always proceeds to END — no second review is performed,
    which prevents any possibility of a correction loop.
    """
    current_json = json.dumps(state["final_output"].model_dump()) if state["final_output"] else "{}"
    prompt = build_revision_prompt(
        current_output_json=current_json,
        review_notes=state["review_notes"] or "",
    )
    result, provider, fallback = _call_and_parse(prompt, TailoringLLMOutput)
    return {
        "final_output": result,
        "provider_used": provider,
        "fallback_used": state["fallback_used"] or fallback,
    }


# ---------------------------------------------------------------------------
# Conditional edge routing function
# ---------------------------------------------------------------------------


def _route_after_review(state: AgenticTailoringState) -> str:
    # Single conditional branch: if the review found issues, call the revision node.
    # revise_output → END unconditionally, so there is no loop — at most one revision.
    return "revise_output" if state["revision_needed"] else END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> object:
    # Graph is built fresh on each call rather than at module load time so that
    # test monkeypatching of settings and provider functions takes effect inside
    # each invocation without needing to reload the module.
    graph: StateGraph = StateGraph(AgenticTailoringState)

    # Register all nodes.
    graph.add_node("retrieve_context", _retrieve_context)
    graph.add_node("analyze_resume", _analyze_resume)
    graph.add_node("analyze_jd", _analyze_jd)
    graph.add_node("analyze_fit_gap", _analyze_fit_gap)
    graph.add_node("decide_route", _decide_route)
    graph.add_node("compose_final", _compose_final)
    graph.add_node("review_output", _review_output)
    graph.add_node("revise_output", _revise_output)

    # Sequential edges through the main pipeline.
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_resume")
    graph.add_edge("analyze_resume", "analyze_jd")
    graph.add_edge("analyze_jd", "analyze_fit_gap")
    graph.add_edge("analyze_fit_gap", "decide_route")
    # decide_route always proceeds to composition — the route is advisory, not a hard gate.
    graph.add_edge("decide_route", "compose_final")
    graph.add_edge("compose_final", "review_output")

    # Conditional branch: review_output → revise_output (if incomplete) OR → END.
    graph.add_conditional_edges("review_output", _route_after_review)
    # revise_output always goes to END — no loop, at most one revision pass.
    graph.add_edge("revise_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_agentic_workflow(
    request: ApplicationTailorRequest,
    db: Any = None,
) -> tuple[TailoringLLMOutput, str, bool]:
    """
    Execute the tool-using agentic workflow and return the final tailoring output.

    Returns the same (output, provider_used, fallback_used) tuple as _get_llm_output
    so the background task can treat both modes uniformly.

    Parameters
    ----------
    request : ApplicationTailorRequest
        The tailoring inputs (resume, JD, optional company info).
    db : Session | None
        SQLAlchemy session for RAG retrieval. When None or when rag_enabled=False,
        the retrieve_context node runs as a no-op and context stays empty.
    """
    graph = _build_graph()

    initial_state: AgenticTailoringState = {
        "request": request,
        "db": db,
        "resume_analysis": None,
        "jd_analysis": None,
        "fit_gap": None,
        "final_output": None,
        "retrieved_context": [],
        # Default route; will be overwritten by decide_route.
        "route_decision": "proceed_to_tailoring",
        "review_notes": None,
        "revision_needed": False,
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
