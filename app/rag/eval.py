# Basic retrieval quality evaluation.
# This is not a production eval system — it exists to demonstrate that retrieval
# quality should be measured, not assumed. A real eval would use labelled query-
# document pairs and metrics like nDCG or MRR. What we implement here (keyword
# coverage) is a fast, dependency-free proxy that catches obvious mismatches.
from app.models.job_description import JobDescription


def score_retrieval(query: str, retrieved_jobs: list[JobDescription]) -> dict:
    """
    Compute a simple keyword-overlap coverage score for a set of retrieved jobs.

    Returns
    -------
    matched_skills : list[str]
        Skills/keywords from the query that appear in at least one retrieved JD.
    coverage_score : float
        Fraction of query keywords found in the retrieved corpus (0.0–1.0).
    notes : str
        Human-readable interpretation and known limitations.

    Limitations
    -----------
    Coverage score measures keyword overlap, not semantic relevance. A retrieved
    JD could contain all the query keywords yet be completely wrong for the role
    (e.g. a JD that lists Python in a "not required" section). Better eval would
    use: (a) human-labelled relevance judgements, (b) an LLM-as-judge scoring
    the relevance of each retrieved document, or (c) downstream task performance
    (does RAG-enriched tailoring score higher in user studies?).
    """
    if not retrieved_jobs:
        return {
            "matched_skills": [],
            "coverage_score": 0.0,
            "notes": "No documents retrieved — cannot assess coverage.",
        }

    # Extract candidate keywords from the query: lower-cased, whitespace-split tokens
    # longer than 2 characters. Short tokens (e.g. "a", "in") add noise.
    query_keywords = {w.lower() for w in query.split() if len(w) > 2}

    if not query_keywords:
        return {
            "matched_skills": [],
            "coverage_score": 0.0,
            "notes": "Query yielded no meaningful keywords to evaluate.",
        }

    # Aggregate all retrieved text into one corpus for matching.
    corpus = " ".join((jd.raw_text or "") + " " + (jd.title or "") for jd in retrieved_jobs).lower()

    matched = [kw for kw in query_keywords if kw in corpus]
    score = round(len(matched) / len(query_keywords), 3)

    notes = (
        f"{len(matched)}/{len(query_keywords)} query keywords found in retrieved corpus. "
        "Coverage score is a keyword-overlap proxy — not a substitute for labelled "
        "relevance judgements or downstream task evaluation."
    )

    return {
        "matched_skills": sorted(matched),
        "coverage_score": score,
        "notes": notes,
    }
