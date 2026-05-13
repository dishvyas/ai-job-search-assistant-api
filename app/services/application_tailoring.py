from app.schemas.application import ApplicationTailorRequest, ApplicationTailorResponse


def tailor_application(request: ApplicationTailorRequest) -> ApplicationTailorResponse:
    """Return a deterministic mock tailoring response based on the provided resume and JD."""
    resume_snippet = request.master_resume[:60].strip()
    jd_snippet = request.job_description[:60].strip()

    return ApplicationTailorResponse(
        tailored_summary=(
            f"[MOCK] Analyzed resume starting with: '{resume_snippet}...' "
            f"against role: '{jd_snippet}...'. "
            "Candidate demonstrates strong alignment with the target position."
        ),
        tailored_bullets=[
            "[MOCK] Delivered high-impact results leveraging core skills from the master resume",
            "[MOCK] Drove cross-functional initiatives aligned with job description requirements",
            "[MOCK] Quantified achievements mapping to key responsibilities outlined in the role",
        ],
        cover_letter_draft=(
            "[MOCK] Dear Hiring Manager,\n\n"
            "I am excited to apply for this role. My background, as reflected in my resume, "
            "closely aligns with the requirements outlined in the job description. "
            "I look forward to discussing how I can contribute to your team.\n\n"
            "Sincerely,\n[Candidate Name]"
        ),
        application_question_answers=[
            "[MOCK] My greatest strength is my ability to adapt quickly to new challenges.",
            "[MOCK] I am motivated by solving complex problems that create real-world impact.",
            "[MOCK] I see this role as a strong match for both my skills and my career goals.",
        ],
        recruiter_message_draft=(
            "[MOCK] Hi [Recruiter Name],\n\n"
            "I came across this opportunity and believe my experience is a strong fit. "
            "I would love to connect and learn more about the role. "
            "Please find my resume attached.\n\n"
            "Best,\n[Candidate Name]"
        ),
        fit_gap_analysis=(
            "[MOCK] FIT: Resume shows relevant experience aligned with core JD requirements. "
            "GAP: Some preferred qualifications are not explicitly covered in the resume — "
            "consider highlighting transferable skills or adding context in the cover letter."
        ),
        interview_talking_points=[
            "[MOCK] Discuss a resume project that directly relates to the job description",
            "[MOCK] Highlight measurable outcomes that demonstrate impact in a similar context",
            "[MOCK] Prepare a question about team challenges to show genuine interest",
        ],
    )
