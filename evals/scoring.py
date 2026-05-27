import json
from pathlib import Path
from typing import Any


def load_eval_case(path: Path) -> dict[str, Any]:
    """Load one eval case JSON file."""
    return json.loads(path.read_text())


def _is_non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return value is not None


def _make_check(
    name: str,
    passed: bool,
    points: int,
    max_points: int,
    details: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "points": points,
        "max_points": max_points,
        "details": details,
    }


def score_output(
    output: dict[str, Any],
    eval_case: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one workflow output against a deterministic eval case."""
    checks: list[dict[str, Any]] = []
    expected_sections = eval_case.get("expected_sections", [])

    missing_sections = [
        section for section in expected_sections if not _is_non_empty(output.get(section))
    ]
    checks.append(
        _make_check(
            name="required_sections_present",
            passed=not missing_sections,
            points=4 if not missing_sections else 0,
            max_points=4,
            details=(
                "All expected sections are present."
                if not missing_sections
                else f"Missing or empty sections: {', '.join(missing_sections)}"
            ),
        )
    )

    bullet_count = len(output.get("tailored_bullets") or [])
    checks.append(
        _make_check(
            name="tailored_bullets_non_empty",
            passed=bullet_count > 0,
            points=2 if bullet_count > 0 else 0,
            max_points=2,
            details=f"tailored_bullets count: {bullet_count}",
        )
    )

    talking_points_count = len(output.get("interview_talking_points") or [])
    checks.append(
        _make_check(
            name="interview_talking_points_non_empty",
            passed=talking_points_count > 0,
            points=2 if talking_points_count > 0 else 0,
            max_points=2,
            details=f"interview_talking_points count: {talking_points_count}",
        )
    )

    cover_letter = output.get("cover_letter_draft") or ""
    checks.append(
        _make_check(
            name="cover_letter_non_empty",
            passed=bool(cover_letter.strip()),
            points=2 if cover_letter.strip() else 0,
            max_points=2,
            details=f"cover_letter_draft length: {len(cover_letter.strip())}",
        )
    )

    fit_gap_analysis = output.get("fit_gap_analysis") or ""
    checks.append(
        _make_check(
            name="fit_gap_analysis_non_empty",
            passed=bool(fit_gap_analysis.strip()),
            points=2 if fit_gap_analysis.strip() else 0,
            max_points=2,
            details=f"fit_gap_analysis length: {len(fit_gap_analysis.strip())}",
        )
    )

    combined_output_text = json.dumps(output).lower()
    expected_keywords = [keyword.lower() for keyword in eval_case.get("expected_keywords", [])]
    matched_keywords = [keyword for keyword in expected_keywords if keyword in combined_output_text]
    checks.append(
        _make_check(
            name="keyword_coverage",
            passed=len(matched_keywords) == len(expected_keywords),
            points=len(matched_keywords),
            max_points=len(expected_keywords),
            details=(
                f"Matched keywords: {matched_keywords}"
                if expected_keywords
                else "No expected_keywords configured."
            ),
        )
    )

    forbidden_terms = [
        term
        for term in eval_case.get("must_not_include", [])
        if term.lower() in combined_output_text
    ]
    checks.append(
        _make_check(
            name="must_not_include",
            passed=not forbidden_terms,
            points=3 if not forbidden_terms else 0,
            max_points=3,
            details=(
                "No forbidden terms detected."
                if not forbidden_terms
                else f"Forbidden terms found: {forbidden_terms}"
            ),
        )
    )

    summary_length = len((output.get("tailored_summary") or "").strip())
    cover_letter_length = len(cover_letter.strip())
    length_ok = (
        summary_length >= 20
        and cover_letter_length >= 40
        and bullet_count >= 2
        and talking_points_count >= 2
    )
    checks.append(
        _make_check(
            name="length_sanity",
            passed=length_ok,
            points=2 if length_ok else 0,
            max_points=2,
            details=(
                f"summary={summary_length}, cover_letter={cover_letter_length}, "
                f"bullets={bullet_count}, talking_points={talking_points_count}"
            ),
        )
    )

    if metadata is not None:
        missing_metadata = [
            field
            for field in ("latency_ms", "provider_used", "generation_attempts")
            if metadata.get(field) is None
        ]
        checks.append(
            _make_check(
                name="metadata_available",
                passed=not missing_metadata,
                points=3 if not missing_metadata else 0,
                max_points=3,
                details=(
                    "Workflow metadata is available."
                    if not missing_metadata
                    else f"Missing metadata fields: {', '.join(missing_metadata)}"
                ),
            )
        )

    total_score = sum(check["points"] for check in checks)
    max_score = sum(check["max_points"] for check in checks)

    return {
        "total_score": total_score,
        "max_score": max_score,
        "passed": total_score == max_score,
        "checks": checks,
    }
