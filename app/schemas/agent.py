from pydantic import BaseModel


class ResumeAnalysis(BaseModel):
    """Structured output from the resume analysis stage."""

    key_skills: list[str]
    relevant_experience: list[str]
    strengths: list[str]


class JobDescriptionAnalysis(BaseModel):
    """Structured output from the job description analysis stage."""

    required_skills: list[str]
    responsibilities: list[str]
    role_focus: str


class FitGapAnalysis(BaseModel):
    """Structured output from the fit/gap analysis stage."""

    fit_points: list[str]
    gap_points: list[str]
    positioning_strategy: str
