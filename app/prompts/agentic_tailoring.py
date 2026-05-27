"""
Prompt builders for the agentic tailoring workflow.

Each builder embeds a ## Task: <stage> header that the MockLLMProvider
uses to detect which schema to return during testing.

The final tailoring prompt produces the same JSON shape as the single-step
tailoring prompt (TailoringLLMOutput), so the existing parsing pipeline is reused.

M10 additions:
- retrieved_context parameter on fit/gap and final prompts injects RAG snippets
- build_revision_prompt for the optional review/revision step
"""

# Each prompt builder is a pure function — no side effects, no LLM calls.
# This makes prompts testable in isolation and easy to iterate on without
# touching any service logic.
from typing import TYPE_CHECKING

from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest

if TYPE_CHECKING:
    from app.models.application import ApplicationTailoringRun

# JSON shape examples shown to the LLM in each prompt.
# Using a concrete example (not an abstract schema) because LLMs follow
# patterns more reliably than abstract type descriptions.
_RESUME_ANALYSIS_SCHEMA = """{
  "key_skills": ["string"],
  "relevant_experience": ["string"],
  "strengths": ["string"]
}"""

_JD_ANALYSIS_SCHEMA = """{
  "required_skills": ["string"],
  "responsibilities": ["string"],
  "role_focus": "string"
}"""

_FIT_GAP_SCHEMA = """{
  "fit_points": ["string"],
  "gap_points": ["string"],
  "positioning_strategy": "string"
}"""

_FINAL_TAILORING_SCHEMA = """{
  "tailored_summary": "string",
  "tailored_bullets": ["string"],
  "cover_letter_draft": "string",
  "application_question_answers": ["string"],
  "recruiter_message_draft": "string",
  "fit_gap_analysis": "string",
  "interview_talking_points": ["string"]
}"""


def build_resume_analysis_prompt(request: ApplicationTailorRequest) -> str:
    """Stage 1 — extract structured information from the candidate's resume."""
    # The "## Task: Analyze Resume" header serves a dual purpose: it guides the LLM
    # and it is the string the MockLLMProvider checks to return the correct JSON shape.
    return "\n".join(
        [
            "## Task: Analyze Resume",
            "",
            "Extract structured information from the resume below.",
            "Your response MUST be valid JSON only. No markdown, no code fences.",
            "",
            "Required JSON shape:",
            _RESUME_ANALYSIS_SCHEMA,
            "",
            "## Resume",
            request.master_resume,
            "",
            "Respond with the JSON object only. No explanation. No markdown.",
        ]
    )


def build_jd_analysis_prompt(request: ApplicationTailorRequest) -> str:
    """Stage 2 — extract structured information from the job description."""
    # "## Task: Analyze Job Description" header doubles as mock detection key.
    return "\n".join(
        [
            "## Task: Analyze Job Description",
            "",
            "Extract structured information from the job description below.",
            "Your response MUST be valid JSON only. No markdown, no code fences.",
            "",
            "Required JSON shape:",
            _JD_ANALYSIS_SCHEMA,
            "",
            "## Job Description",
            request.job_description,
            "",
            "Respond with the JSON object only. No explanation. No markdown.",
        ]
    )


def build_fit_gap_prompt(
    request: ApplicationTailorRequest,
    resume_analysis: ResumeAnalysis,
    jd_analysis: JobDescriptionAnalysis,
    retrieved_context: list[str] | None = None,
    artifact_context: list["ApplicationTailoringRun"] | None = None,
) -> str:
    """Stage 3 — identify fit points, gaps, and positioning strategy."""
    # Passing structured skill lists rather than raw resume/JD text keeps the prompt
    # focused — the LLM has already done the extraction work in stages 1 and 2.
    resume_skills = ", ".join(resume_analysis.key_skills)
    required_skills = ", ".join(jd_analysis.required_skills)

    # "## Task: Analyze Fit and Gap" header doubles as mock detection key.
    sections = [
        "## Task: Analyze Fit and Gap",
        "",
        "Given the resume analysis and job description analysis below, identify"
        " fit points, gap points, and a positioning strategy.",
        "Your response MUST be valid JSON only. No markdown, no code fences.",
        "",
        "Required JSON shape:",
        _FIT_GAP_SCHEMA,
        "",
        "## Resume Skills",
        resume_skills,
        "",
        "## Required Skills",
        required_skills,
        "",
        "## Role Focus",
        jd_analysis.role_focus,
    ]

    # Inject retrieved context as supporting signal — helps the LLM calibrate gap
    # severity against what similar real roles actually require. Capped at 3 snippets
    # to avoid blowing the prompt budget. User resume/JD always take precedence.
    if retrieved_context:
        sections += ["", "## Retrieved Context (from similar roles — for reference only)"]
        for snippet in retrieved_context[:3]:
            sections += ["", snippet[:300].strip()]
        sections += [
            "",
            "Use the retrieved context above as background reference only."
            " Base your analysis primarily on the resume and job description.",
        ]

    if artifact_context:
        sections += ["", "## Retrieved Past Tailored Artifacts"]
        for artifact in artifact_context[:3]:
            if artifact.tailored_summary:
                sections += ["", "Summary", artifact.tailored_summary[:250].strip()]
            if artifact.fit_gap_analysis:
                sections += ["", "Fit/Gap Preview", artifact.fit_gap_analysis[:250].strip()]
        sections += [
            "",
            "Use these artifact examples as positioning reference only.",
            "Do not copy claims or invent experience not present in the current resume.",
        ]

    sections += ["", "Respond with the JSON object only. No explanation. No markdown."]
    return "\n".join(sections)


def build_final_tailoring_prompt(
    request: ApplicationTailorRequest,
    resume_analysis: ResumeAnalysis,
    jd_analysis: JobDescriptionAnalysis,
    fit_gap: FitGapAnalysis,
    retrieved_context: list[str] | None = None,
    artifact_context: list["ApplicationTailoringRun"] | None = None,
) -> str:
    """Stage 4 — compose final application materials using all prior analyses."""
    # Slice to the top 2 fit points and 1 gap point to keep the prompt concise;
    # the full lists are available in prior state if the LLM needs more context.
    fit_summary = "; ".join(fit_gap.fit_points[:2])
    gap_summary = "; ".join(fit_gap.gap_points[:1])

    # No "## Task:" header here — the final stage returns TailoringLLMOutput, the
    # same shape as the single-step prompt, so the mock falls through to the default
    # _tailoring_response path automatically.
    sections = [
        "You are an expert career coach and resume writer.",
        "",
        "Using the structured analyses below, produce tailored job application materials.",
        "Your response MUST be valid JSON only. No markdown, no code fences.",
        "",
        "Required JSON shape:",
        _FINAL_TAILORING_SCHEMA,
        "",
        "## Key Skills (from resume)",
        ", ".join(resume_analysis.key_skills),
        "",
        "## Role Requirements",
        jd_analysis.role_focus,
        "",
        "## Fit Points",
        fit_summary,
        "",
        "## Gap Points",
        gap_summary,
        "",
        "## Positioning Strategy",
        fit_gap.positioning_strategy,
        "",
        "## Original Job Description",
        request.job_description,
    ]

    # Inject retrieved context last so the LLM's primary anchors are the structured
    # analyses above. Retrieved context provides vocabulary and framing cues from
    # real similar roles — useful for bullet phrasing and industry alignment.
    if retrieved_context:
        sections += ["", "## Retrieved Context (from similar roles — for reference only)"]
        for snippet in retrieved_context[:3]:
            sections += ["", snippet[:300].strip()]
        sections += [
            "",
            "Use the retrieved context above as framing reference only."
            " The resume and job description above are the primary sources of truth.",
        ]

    if artifact_context:
        sections += ["", "## Retrieved Past Tailored Artifacts"]
        for artifact in artifact_context[:3]:
            if artifact.tailored_summary:
                sections += ["", "Summary", artifact.tailored_summary[:250].strip()]
            top_bullets = (artifact.tailored_bullets or [])[:2]
            if top_bullets:
                sections += ["Top Bullets"] + [str(bullet).strip() for bullet in top_bullets]
            if artifact.fit_gap_analysis:
                sections += ["Fit/Gap Preview", artifact.fit_gap_analysis[:250].strip()]
        sections += [
            "",
            "Use these artifact examples for tone, structure, and positioning inspiration only.",
            "Do not copy claims.",
            "Do not invent experience not present in the candidate resume.",
            "The current resume and job description remain the source of truth.",
        ]

    sections += ["", "Respond with the JSON object only. No explanation. No markdown."]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Revision prompt (M10 — optional single-pass correction)
# ---------------------------------------------------------------------------

# The ## Task: Revise Output header is detected by MockLLMProvider to return a
# correction-specific response shape, distinct from the normal tailoring response.
_REVISION_HEADER = "## Task: Revise Output"


def build_revision_prompt(current_output_json: str, review_notes: str) -> str:
    """
    Build a prompt for the optional revision step.

    Called only when the review node marks revision_needed=True. The LLM receives
    the current (incomplete) output and the specific review notes so it can produce
    a corrected, complete TailoringLLMOutput. At most one revision pass is made —
    a second review is not performed to avoid any possibility of looping.
    """
    # The "## Task: Revise Output" header is the mock detection key for this stage.
    return "\n".join(
        [
            _REVISION_HEADER,
            "",
            "The previous output failed quality review. Produce a corrected, complete version.",
            "Address the specific issues listed in the review notes below.",
            "Your response MUST be valid JSON only. No markdown, no code fences.",
            "",
            "Review notes (what to fix):",
            review_notes,
            "",
            "Required JSON shape:",
            _FINAL_TAILORING_SCHEMA,
            "",
            "## Previous Output (improve this)",
            current_output_json,
            "",
            "Respond with the corrected JSON object only. No explanation. No markdown.",
        ]
    )
