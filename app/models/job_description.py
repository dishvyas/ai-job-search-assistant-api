# ORM model for the job_descriptions table.
# Each row represents a single ingested job description with its embedding,
# ready for semantic similarity search via pgvector.
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Structured fields — stored separately from raw_text to support metadata
    # filtering alongside semantic search (e.g. filter by location, then rank by similarity).
    title: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full job description text. We embed raw_text rather than just the title because
    # skills, responsibilities, and requirements are spread throughout the body — the
    # title alone ("Senior Engineer") conveys far less semantic signal than the full text.
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Flexible metadata for structured filters (role_type, tech_stack, seniority, etc.).
    # Stored as JSON rather than separate columns because the relevant attributes vary
    # by role type — forcing a fixed schema would require frequent migrations as new
    # filter dimensions are added. JSON keeps the schema stable while filters evolve.
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    # 1536-dimensional vector produced by text-embedding-3-small.
    # Vector(1536) must match the embedding model's output dimension exactly —
    # pgvector uses the dimension at index creation time for the IVFFlat/HNSW index.
    embedding: Mapped[list | None] = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
