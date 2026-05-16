from app.schemas.application import ApplicationTailorRequest

# The exact JSON shape we ask the LLM to produce.
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


def build_tailoring_prompt(request: ApplicationTailorRequest) -> str:
    """
    Build a structured prompt that instructs the LLM to return valid JSON only.
    Optional fields are included only when provided.
    """
    sections = [
        "You are an expert career coach and resume writer.",
        "",
        "Using the inputs below, produce tailored job application materials.",
        "Your response MUST be valid JSON only.",
        "Do NOT include markdown, code fences, or any text outside the JSON object.",
        "",
        "Required JSON shape:",
        _JSON_SCHEMA_EXAMPLE,
        "",
        "## Master Resume",
        request.master_resume,
        "",
        "## Job Description",
        request.job_description,
    ]

    if request.company_info:
        sections += ["", "## Company Info", request.company_info]

    if request.user_preferences:
        sections += ["", "## User Preferences", request.user_preferences]

    sections += [
        "",
        "Respond with the JSON object only. No explanation. No markdown.",
    ]

    return "\n".join(sections)
