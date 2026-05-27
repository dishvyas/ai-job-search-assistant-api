from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.application import ApplicationTailoringRun
from app.models.run_status import RunStatus
from app.rag.embed import generate_embedding
from app.repositories.application_runs import save_artifact_embedding


def build_artifact_text(run: ApplicationTailoringRun) -> str:
    """Build a compact retrieval document from generated tailoring outputs only."""
    sections: list[str] = []

    if run.tailored_summary:
        sections += ["Summary", run.tailored_summary.strip()]
    if run.tailored_bullets:
        sections += ["Bullets"] + [str(bullet).strip() for bullet in run.tailored_bullets if bullet]
    if run.fit_gap_analysis:
        sections += ["Fit Gap Analysis", run.fit_gap_analysis.strip()]
    if run.interview_talking_points:
        sections += ["Interview Talking Points"] + [
            str(point).strip() for point in run.interview_talking_points if point
        ]
    if run.recruiter_message_draft:
        sections += ["Recruiter Message", run.recruiter_message_draft.strip()]

    return "\n".join(sections).strip()


def store_artifact_embedding_for_run(db: Session, run: ApplicationTailoringRun) -> None:
    """
    Best-effort artifact indexing for completed runs.

    Any failure is swallowed so the user-facing completed run remains completed.
    """
    if not settings.rag_enabled or not settings.artifact_retrieval_enabled:
        return
    if run.status != RunStatus.COMPLETED.value:
        return

    artifact_text = build_artifact_text(run)
    if not artifact_text:
        return

    try:
        embedding = generate_embedding(artifact_text)
        save_artifact_embedding(db, run, embedding)
    except Exception:  # noqa: BLE001
        db.rollback()


def retrieve_similar_artifacts(
    db: Session,
    query: str,
    top_k: int | None = None,
) -> list[ApplicationTailoringRun]:
    """Retrieve similar completed tailored artifacts when vector search is available."""
    if not settings.rag_enabled or not settings.artifact_retrieval_enabled:
        return []

    bind = db.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        # SQLite and other local test dialects skip vector similarity gracefully.
        return []

    try:
        query_vector = generate_embedding(query)
        effective_top_k = top_k if top_k is not None else settings.artifact_retrieval_top_k
        distance_col = ApplicationTailoringRun.artifact_embedding.op("<=>")(query_vector)
        results = (
            db.query(ApplicationTailoringRun, (1 - distance_col).label("similarity"))
            .filter(ApplicationTailoringRun.status == RunStatus.COMPLETED.value)
            .filter(ApplicationTailoringRun.artifact_embedding.isnot(None))
            .order_by(distance_col)
            .limit(effective_top_k)
            .all()
        )
        threshold = settings.artifact_similarity_threshold
        return [run for run, score in results if float(score) >= threshold]
    except Exception:  # noqa: BLE001
        return []
