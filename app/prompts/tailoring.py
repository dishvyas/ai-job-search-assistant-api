from app.schemas.application import ApplicationTailorRequest


def build_tailoring_prompt(request: ApplicationTailorRequest) -> str:
    """
    Build the tailoring prompt from a request.
    Includes optional fields only when provided.
    """
    sections = [
        "You are an expert career coach and resume writer.",
        "Using the inputs below, produce tailored job application materials.",
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
        "## Output Required",
        "Provide all of the following sections:",
        "1. Tailored Summary",
        "2. Tailored Bullets (3–5 bullet points)",
        "3. Cover Letter Draft",
        "4. Application Question Answers (3 sample answers)",
        "5. Recruiter Message Draft",
        "6. Fit / Gap Analysis",
        "7. Interview Talking Points (3–5 points)",
    ]

    return "\n".join(sections)
