# Job description ingestion — generates an embedding and persists the full record.
# Separating ingestion from retrieval keeps each file focused and makes it easy
# to swap the embedding model without touching the query path.
from sqlalchemy.orm import Session

from app.models.job_description import JobDescription
from app.rag.embed import generate_embedding


def ingest_job_description(
    db: Session,
    title: str,
    raw_text: str,
    company: str | None = None,
    location: str | None = None,
    metadata: dict | None = None,
) -> JobDescription:
    """
    Generate an embedding for raw_text and persist the job description.

    We embed raw_text (not just the title) because the full description
    contains the skills, requirements, and responsibilities that a candidate
    query should match against. A title like "Senior Engineer" is far too
    generic to produce meaningful similarity scores on its own.
    """
    embedding = generate_embedding(raw_text)

    jd = JobDescription(
        title=title,
        company=company,
        location=location,
        raw_text=raw_text,
        metadata_=metadata,
        embedding=embedding,
    )
    db.add(jd)
    db.commit()
    db.refresh(jd)
    return jd
