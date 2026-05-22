# Routes for the RAG job matching pipeline.
# /ingest: store a job description with its embedding
# /match: semantic + metadata search over stored JDs
# /compare: side-by-side demo of generation without vs with RAG context
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.llm.factory import get_llm_provider
from app.llm.parsing import parse_tailoring_response
from app.prompts.tailoring import build_tailoring_prompt
from app.rag.ingest import ingest_job_description
from app.rag.retrieve import retrieve_relevant_jobs
from app.schemas.jobs import (
    CompareRequest,
    CompareResponse,
    JobIngestRequest,
    JobIngestResponse,
    MatchedJobResponse,
    MatchRequest,
    MatchResponse,
    RetrievedJobSummary,
)

router = APIRouter()


@router.post("/ingest", response_model=JobIngestResponse)
def ingest(
    request: JobIngestRequest,
    db: Session = Depends(get_db),
) -> JobIngestResponse:
    """
    Ingest a job description: generate its embedding and persist to DB.
    Returns the stored record's ID and creation timestamp.
    """
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=422,
            detail="RAG is disabled. Set RAG_ENABLED=true to use this endpoint.",
        )
    jd = ingest_job_description(
        db=db,
        title=request.title,
        company=request.company,
        location=request.location,
        raw_text=request.raw_text,
        metadata=request.metadata,
    )
    return JobIngestResponse(
        job_description_id=jd.id,
        title=jd.title,
        created_at=jd.created_at,
    )


@router.post("/match", response_model=MatchResponse)
def match(
    request: MatchRequest,
    db: Session = Depends(get_db),
) -> MatchResponse:
    """
    Semantic + metadata search over stored job descriptions.
    Returns matched JDs with similarity scores above the configured threshold.
    """
    if not settings.rag_enabled:
        raise HTTPException(
            status_code=422,
            detail="RAG is disabled. Set RAG_ENABLED=true to use this endpoint.",
        )
    results = retrieve_relevant_jobs(
        db=db,
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )
    matched = [
        MatchedJobResponse(
            id=jd.id,
            title=jd.title,
            company=jd.company,
            location=jd.location,
            similarity_score=round(score, 4),
        )
        for jd, score in results
    ]
    return MatchResponse(results=matched, total=len(matched))


@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    db: Session = Depends(get_db),
) -> CompareResponse:
    """
    Side-by-side comparison: LLM response without RAG vs with RAG context injected.

    This endpoint is a demo tool — it runs two LLM calls and returns both outputs
    so you can see in one response how retrieved context changes generation quality.
    The retrieved_jobs list shows exactly which JDs were used as context.
    """
    from app.schemas.application import ApplicationTailorRequest

    # Build a minimal tailoring request from the compare inputs.
    tailor_req = ApplicationTailorRequest(
        master_resume=request.resume_summary,
        job_description=request.query,
    )

    provider = get_llm_provider()

    # --- Without RAG ---
    prompt_plain = build_tailoring_prompt(tailor_req, rag_context=None)
    raw_plain = provider.generate_text(prompt_plain)
    try:
        plain_output = parse_tailoring_response(raw_plain)
        without_rag_text = plain_output.tailored_summary
    except Exception:  # noqa: BLE001
        without_rag_text = raw_plain[:500]

    # --- With RAG ---
    retrieved_jobs_summary: list[RetrievedJobSummary] = []
    with_rag_text = without_rag_text  # fallback if no results or RAG disabled

    if settings.rag_enabled:
        results = retrieve_relevant_jobs(db=db, query=request.query)
        rag_jds = [jd for jd, _score in results]
        scores = {jd.id: score for jd, score in results}

        if rag_jds:
            prompt_rag = build_tailoring_prompt(tailor_req, rag_context=rag_jds)
            raw_rag = provider.generate_text(prompt_rag)
            try:
                rag_output = parse_tailoring_response(raw_rag)
                with_rag_text = rag_output.tailored_summary
            except Exception:  # noqa: BLE001
                with_rag_text = raw_rag[:500]

            retrieved_jobs_summary = [
                RetrievedJobSummary(
                    title=jd.title,
                    company=jd.company,
                    similarity_score=round(scores.get(jd.id, 0.0), 4),
                )
                for jd in rag_jds
            ]

    return CompareResponse(
        without_rag=without_rag_text,
        with_rag=with_rag_text,
        retrieved_jobs=retrieved_jobs_summary,
    )
