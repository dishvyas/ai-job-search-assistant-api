import json

from app.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Deterministic LLM provider for local development and testing.
    Returns valid JSON matching TailoringLLMOutput so the full
    parse -> validate -> map pipeline can be exercised without a real API.
    """

    def generate_text(self, prompt: str) -> str:
        prompt_preview = prompt[:60].strip().replace("\n", " ")
        output = {
            "tailored_summary": (f"[MOCK] Tailored summary based on prompt: '{prompt_preview}...'"),
            "tailored_bullets": [
                "[MOCK] Delivered high-impact results relevant to the job description",
                "[MOCK] Drove cross-functional initiatives aligned with role requirements",
                "[MOCK] Quantified achievements mapping to key responsibilities",
            ],
            "cover_letter_draft": (
                "[MOCK] Dear Hiring Manager,\n\n"
                "I am excited to apply for this role. My background closely aligns "
                "with the requirements outlined in the job description.\n\n"
                "Sincerely,\n[Candidate Name]"
            ),
            "application_question_answers": [
                "[MOCK] My greatest strength is my ability to adapt quickly to new challenges.",
                "[MOCK] I am motivated by solving complex problems that create real-world impact.",
                "[MOCK] I see this role as a strong match for both my skills and career goals.",
            ],
            "recruiter_message_draft": (
                "[MOCK] Hi [Recruiter Name],\n\n"
                "I came across this opportunity and believe my experience is a strong fit. "
                "I would love to connect and learn more.\n\n"
                "Best,\n[Candidate Name]"
            ),
            "fit_gap_analysis": (
                "[MOCK] FIT: Resume shows relevant experience aligned with JD requirements. "
                "GAP: Some preferred qualifications not explicitly covered — "
                "consider highlighting transferable skills in the cover letter."
            ),
            "interview_talking_points": [
                "[MOCK] Discuss a resume project that directly relates to the job description",
                "[MOCK] Highlight measurable outcomes that demonstrate impact in context",
                "[MOCK] Prepare a question about team challenges to show genuine interest",
            ],
        }
        return json.dumps(output)
