"""
Prompt builders for the four-stage agentic tailoring workflow.

Each builder embeds a ## Task: <stage> header that the MockLLMProvider
uses to detect which schema to return during testing.

The final tailoring prompt produces the same JSON shape as the single-step
tailoring prompt (TailoringLLMOutput), so the existing parsing pipeline is reused.
"""

from app.schemas.agent import FitGapAnalysis, JobDescriptionAnalysis, ResumeAnalysis
from app.schemas.application import ApplicationTailorRequest

# JSON shape examples shown to the LLM in each prompt.
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
) -> str:
    """Stage 3 — identify fit points, gaps, and positioning strategy."""
    resume_skills = ", ".join(resume_analysis.key_skills)
    required_skills = ", ".join(jd_analysis.required_skills)

    return "\n".join(
        [
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
            "",
            "Respond with the JSON object only. No explanation. No markdown.",
        ]
    )


def build_final_tailoring_prompt(
    request: ApplicationTailorRequest,
    resume_analysis: ResumeAnalysis,
    jd_analysis: JobDescriptionAnalysis,
    fit_gap: FitGapAnalysis,
) -> str:
    """Stage 4 — compose final application materials using all prior analyses."""
    fit_summary = "; ".join(fit_gap.fit_points[:2])
    gap_summary = "; ".join(fit_gap.gap_points[:1])

    return "\n".join(
        [
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
            "",
            "Respond with the JSON object only. No explanation. No markdown.",
        ]
    )
