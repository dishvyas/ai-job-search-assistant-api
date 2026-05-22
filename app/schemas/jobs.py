# Request/response schemas for the /api/v1/jobs routes.
# These are separate from app/schemas/application.py so job-matching concerns
# don't bleed into the tailoring contract.
from datetime import datetime

from pydantic import BaseModel, field_validator


class JobIngestRequest(BaseModel):
    title: str
    company: str | None = None
    location: str | None = None
    raw_text: str
    # Optional structured metadata for filtered retrieval (role_type, tech_stack, etc.)
    metadata: dict | None = None

    @field_validator("title", "raw_text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class JobIngestResponse(BaseModel):
    job_description_id: int
    title: str
    created_at: datetime


class MatchRequest(BaseModel):
    query: str
    filters: dict | None = None
    top_k: int | None = None

    @field_validator("query")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class MatchedJobResponse(BaseModel):
    id: int
    title: str
    company: str | None = None
    location: str | None = None
    similarity_score: float


class MatchResponse(BaseModel):
    results: list[MatchedJobResponse]
    total: int


class CompareRequest(BaseModel):
    query: str
    resume_summary: str

    @field_validator("query", "resume_summary")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class RetrievedJobSummary(BaseModel):
    title: str
    company: str | None = None
    similarity_score: float


class CompareResponse(BaseModel):
    without_rag: str
    with_rag: str
    retrieved_jobs: list[RetrievedJobSummary]
