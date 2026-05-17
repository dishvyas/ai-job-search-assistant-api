import json

from app.llm.base import LLMProvider

# Stage-header keywords used by the agentic prompt builders.
# The mock detects these to return the correct JSON shape for each stage.
_RESUME_ANALYSIS_HEADER = "## Task: Analyze Resume"
_JD_ANALYSIS_HEADER = "## Task: Analyze Job Description"
_FIT_GAP_HEADER = "## Task: Analyze Fit and Gap"


class MockLLMProvider(LLMProvider):
    """
    Deterministic LLM provider for local development and testing.

    Default response shape: TailoringLLMOutput (used by single-step workflow and
    the agentic final-composition stage).

    Agentic intermediate stages are detected by a ## Task: <stage> header in the
    prompt and return the appropriate Pydantic schema shape.
    """

    def generate_text(self, prompt: str) -> str:
        if _RESUME_ANALYSIS_HEADER in prompt:
            return self._resume_analysis_response()
        if _JD_ANALYSIS_HEADER in prompt:
            return self._jd_analysis_response()
        if _FIT_GAP_HEADER in prompt:
            return self._fit_gap_response()
        return self._tailoring_response(prompt)

    # ------------------------------------------------------------------
    # Agentic stage responses
    # ------------------------------------------------------------------

    def _resume_analysis_response(self) -> str:
        return json.dumps(
            {
                "key_skills": [
                    "[MOCK] Python",
                    "[MOCK] FastAPI",
                    "[MOCK] System Design",
                    "[MOCK] API Development",
                ],
                "relevant_experience": [
                    "[MOCK] 5 years backend engineering",
                    "[MOCK] Distributed systems design",
                    "[MOCK] REST API development",
                ],
                "strengths": [
                    "[MOCK] Strong technical foundation",
                    "[MOCK] Adaptability to new challenges",
                    "[MOCK] Problem-solving ability",
                ],
            }
        )

    def _jd_analysis_response(self) -> str:
        return json.dumps(
            {
                "required_skills": [
                    "[MOCK] Python",
                    "[MOCK] API development",
                    "[MOCK] Cross-functional collaboration",
                ],
                "responsibilities": [
                    "[MOCK] Build scalable backend systems",
                    "[MOCK] Collaborate with product and frontend teams",
                    "[MOCK] Ensure reliability and performance of services",
                ],
                "role_focus": (
                    "[MOCK] Backend engineering with emphasis on reliability, "
                    "scalability, and team collaboration"
                ),
            }
        )

    def _fit_gap_response(self) -> str:
        return json.dumps(
            {
                "fit_points": [
                    "[MOCK] Strong Python background matches role requirements",
                    "[MOCK] API development experience directly relevant",
                    "[MOCK] System design skills align with scalability focus",
                ],
                "gap_points": [
                    "[MOCK] Some preferred qualifications not explicitly covered in resume",
                ],
                "positioning_strategy": (
                    "[MOCK] Emphasise technical depth, adaptability, and measurable "
                    "outcomes in application materials"
                ),
            }
        )

    # ------------------------------------------------------------------
    # Default: TailoringLLMOutput shape (single-step + agentic final stage)
    # ------------------------------------------------------------------

    def _tailoring_response(self, prompt: str) -> str:
        prompt_preview = prompt[:60].strip().replace("\n", " ")
        return json.dumps(
            {
                "tailored_summary": (
                    f"[MOCK] Tailored summary based on prompt: '{prompt_preview}...'"
                ),
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
                    "[MOCK] I am motivated by solving complex problems that create "
                    "real-world impact.",
                    "[MOCK] I see this role as a strong match for both my skills and career goals.",
                ],
                "recruiter_message_draft": (
                    "[MOCK] Hi [Recruiter Name],\n\n"
                    "I came across this opportunity and believe my experience is a strong "
                    "fit. I would love to connect and learn more.\n\n"
                    "Best,\n[Candidate Name]"
                ),
                "fit_gap_analysis": (
                    "[MOCK] FIT: Resume shows relevant experience aligned with JD "
                    "requirements. GAP: Some preferred qualifications not explicitly "
                    "covered — consider highlighting transferable skills in the cover "
                    "letter."
                ),
                "interview_talking_points": [
                    "[MOCK] Discuss a resume project that directly relates to the job description",
                    "[MOCK] Highlight measurable outcomes that demonstrate impact in context",
                    "[MOCK] Prepare a question about team challenges to show genuine interest",
                ],
            }
        )
