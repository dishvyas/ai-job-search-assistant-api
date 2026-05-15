from app.llm.factory import get_llm_provider
from app.prompts.tailoring import build_tailoring_prompt
from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse


def tailor_application(request: ApplicationTailorRequest) -> ApplicationTailorResponse:
    """
    Tailor a job application using the configured LLM provider.

    The prompt is built from the request, sent to the provider, and the raw
    response is surfaced in the summary and fit/gap fields. The remaining
    fields are structured mock content until full LLM output parsing lands
    in a future milestone.
    """
    prompt = build_tailoring_prompt(request)
    provider = get_llm_provider()
    llm_output = provider.generate_text(prompt)

    resume_snippet = request.master_resume[:60].strip()
    jd_snippet = request.job_description[:60].strip()
    llm_preview = llm_output[:120].strip()

    return ApplicationTailorResponse(
        tailored_summary=(
            f"Analyzed resume starting with: '{resume_snippet}...' "
            f"against role: '{jd_snippet}...'. "
            f"Provider output received: {llm_preview}"
        ),
        tailored_bullets=[
            "Delivered high-impact results leveraging core skills from the master resume",
            "Drove cross-functional initiatives aligned with job description requirements",
            "Quantified achievements mapping to key responsibilities outlined in the role",
        ],
        cover_letter_draft=(
            "Dear Hiring Manager,\n\n"
            "I am excited to apply for this role. My background closely aligns with "
            "the requirements outlined in the job description. "
            "I look forward to discussing how I can contribute to your team.\n\n"
            "Sincerely,\n[Candidate Name]"
        ),
        application_question_answers=[
            "My greatest strength is my ability to adapt quickly to new challenges.",
            "I am motivated by solving complex problems that create real-world impact.",
            "I see this role as a strong match for both my skills and my career goals.",
        ],
        recruiter_message_draft=(
            "Hi [Recruiter Name],\n\n"
            "I came across this opportunity and believe my experience is a strong fit. "
            "I would love to connect and learn more about the role.\n\n"
            "Best,\n[Candidate Name]"
        ),
        fit_gap_analysis=(
            "FIT: Resume shows relevant experience aligned with core JD requirements. "
            "GAP: Some preferred qualifications are not explicitly covered in the resume "
            "— consider highlighting transferable skills in the cover letter. "
            f"[Provider note: {llm_preview}]"
        ),
        interview_talking_points=[
            "Discuss a resume project that directly relates to the job description",
            "Highlight measurable outcomes that demonstrate impact in a similar context",
            "Prepare a question about team challenges to show genuine interest",
        ],
    )
