# Single-step tailoring prompt builder.
# Separating prompt construction from service logic means prompt copy can be
# iterated on without touching orchestration code, and prompts can be unit-tested
# independently of any LLM call.
from typing import TYPE_CHECKING

from app.schemas.application import ApplicationTailorRequest

if TYPE_CHECKING:
    from app.models.job_description import JobDescription

# The exact JSON shape we ask the LLM to produce.
# Showing a concrete example rather than a schema description works better with
# current LLMs — they pattern-match on the example more reliably than on type specs.
# Keeping this as a module-level constant makes it easy to review and update.
_JSON_SCHEMA_EXAMPLE = """{
  "tailored_summary": "string",
  "tailored_bullets": ["string", "string", "string"],
  "cover_letter_draft": "string",
  "application_question_answers": ["string", "string", "string"],
  "recruiter_message_draft": "string",
  "fit_gap_analysis": "string",
  "interview_talking_points": ["string", "string", "string"]
}"""


def build_tailoring_prompt(
    request: ApplicationTailorRequest,
    rag_context: list["JobDescription"] | None = None,
) -> str:
    """
    Build a structured prompt that instructs the LLM to return valid JSON only.
    Optional fields are included only when provided.

    When rag_context is supplied, the top retrieved job descriptions are injected
    as additional context before the candidate's resume. Retrieved context improves
    specificity because the LLM can align its output language and emphasis with
    the vocabulary, skills, and responsibilities from real matching roles — rather
    than reasoning only from the single job description provided by the user.
    """
    sections = [
        "You are an expert career coach and resume writer.",
        "",
        "Using the inputs below, produce tailored job application materials.",
        # Explicit "no markdown" instruction reduces the chance of ```json fences
        # appearing in the response, which would break json.loads().
        "Your response MUST be valid JSON only.",
        "Do NOT include markdown, code fences, or any text outside the JSON object.",
        "",
        "Required JSON shape:",
        _JSON_SCHEMA_EXAMPLE,
    ]

    # RAG context injection — only present when RAG is enabled and retrieval returned results.
    # Placed before the resume so the model can use the retrieved roles as reference points
    # when framing the candidate's experience. Empty list is treated the same as None.
    if rag_context:
        sections += ["", "## Similar Roles (retrieved for context)"]
        for i, jd in enumerate(rag_context, 1):
            header = f"### Role {i}: {jd.title}"
            if jd.company:
                header += f" at {jd.company}"
            # Truncate to 500 chars — enough for the model to pick up vocabulary
            # and key requirements without blowing out the context window.
            snippet = (jd.raw_text or "")[:500].strip()
            sections += ["", header, snippet]
        sections += [
            "",
            "Use the roles above as reference points for industry vocabulary and "
            "common requirements. Do NOT copy them verbatim.",
        ]

    sections += [
        "",
        "## Master Resume",
        request.master_resume,
        "",
        "## Job Description",
        request.job_description,
    ]

    # Optional sections are omitted entirely rather than sent as empty strings —
    # an empty "## Company Info" section adds noise and may confuse the model.
    if request.company_info:
        sections += ["", "## Company Info", request.company_info]

    if request.user_preferences:
        sections += ["", "## User Preferences", request.user_preferences]

    # Repeat the JSON-only instruction at the end; LLMs tend to follow the most
    # recent instruction when there is ambiguity.
    sections += [
        "",
        "Respond with the JSON object only. No explanation. No markdown.",
    ]

    return "\n".join(sections)
