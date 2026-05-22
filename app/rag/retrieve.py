# Semantic retrieval — the core of the RAG pipeline.
# Retrieval quality directly determines the quality of the enriched LLM prompt;
# bad retrieval (low precision) injects irrelevant context that can actually hurt
# generation quality more than no context at all.
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.job_description import JobDescription
from app.rag.embed import generate_embedding


def _enrich_query(query: str, filters: dict | None) -> str:
    """
    Append structured filter context to the raw query before embedding.

    Query enrichment improves retrieval quality because embedding a plain
    query string like "Python backend engineer" produces a generic vector.
    Adding filter context ("role_type: backend, tech_stack: Python, FastAPI")
    shifts the vector toward the specific semantic neighbourhood we care about,
    reducing false matches from adjacent roles (e.g. data engineering, devops).
    """
    if not filters:
        return query
    # Concatenate key filter values as plain text — the embedding model treats
    # them as additional semantic signal without any special syntax.
    extra = " ".join(f"{k}: {v}" for k, v in filters.items() if v)
    return f"{query} {extra}".strip()


def retrieve_relevant_jobs(
    db: Session,
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
) -> list[tuple[JobDescription, float]]:
    """
    Retrieve the most semantically similar job descriptions for a query.

    Returns a list of (JobDescription, similarity_score) tuples, ordered by
    descending similarity, with low-quality matches filtered out.

    Strategy
    --------
    1. Enrich the query with any structured filter context.
    2. Embed the enriched query to get a query vector.
    3. Use pgvector cosine distance (<=> operator) to rank all stored JDs.
    4. Apply optional metadata filters as SQL WHERE conditions.
    5. Discard matches below the similarity_threshold.

    Why combine semantic + metadata filters?
    Semantic search alone can return plausible-sounding but irrelevant results
    (e.g. a ML role when you're looking for backend). Metadata filters act as
    hard constraints — they eliminate entire categories before ranking, which
    improves precision without sacrificing recall on the remaining set.

    Why apply a similarity threshold?
    Returning the top-k results unconditionally is dangerous: if the corpus
    is small or the query is unusual, even the "best" match might have low
    similarity. Injecting low-similarity context into the LLM prompt can
    actively mislead the model — it's better to inject nothing than noise.
    """
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    enriched = _enrich_query(query, filters)
    query_vector = generate_embedding(enriched)

    # pgvector cosine distance: <=> returns distance (0 = identical, 2 = opposite).
    # We convert to similarity score: similarity = 1 - distance.
    # Ordering by distance ASC gives us the most similar results first.

    distance_col = JobDescription.embedding.op("<=>")(query_vector)
    q = db.query(JobDescription, (1 - distance_col).label("similarity")).filter(
        JobDescription.embedding.isnot(None)
    )

    # Apply structured metadata filters as SQL JSON-path conditions.
    # This runs before the similarity ranking so PostgreSQL can reduce the
    # candidate set before computing distances — more efficient than post-filtering.
    if filters:
        for key, value in filters.items():
            if value is not None:
                # JSON field access: metadata->>'key' = 'value'
                q = q.filter(JobDescription.metadata_[key].astext == str(value))

    results = q.order_by(distance_col).limit(effective_top_k).all()

    # Filter out weak matches. The threshold is configurable so it can be tuned
    # without code changes once real data is available.
    threshold = settings.similarity_threshold
    return [(jd, float(score)) for jd, score in results if float(score) >= threshold]
