from pathlib import Path

from evals.scoring import load_eval_case, score_output

CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "cases"


def _valid_output() -> dict:
    return {
        "tailored_summary": "Strong experience and fit for the role with measurable impact.",
        "tailored_bullets": [
            "Delivered meaningful impact across backend systems.",
            "Improved reliability for a role-focused platform team.",
        ],
        "cover_letter_draft": (
            "Dear Hiring Manager, I bring relevant experience, clear fit, "
            "and measurable impact for this role. Sincerely, Candidate."
        ),
        "application_question_answers": [
            "I bring relevant experience.",
            "I care about impact.",
            "This role is a strong fit.",
        ],
        "recruiter_message_draft": (
            "Hi, my experience and fit make me a strong candidate for this role."
        ),
        "fit_gap_analysis": "FIT: strong experience and impact. GAP: some tools to learn.",
        "interview_talking_points": ["Discuss impact.", "Discuss relevant role experience."],
        "provider_used": "mock",
        "latency_ms": 12,
        "generation_attempts": 1,
    }


def test_load_eval_case_reads_json():
    case_path = CASES_DIR / "backend_engineer_germany.json"
    eval_case = load_eval_case(case_path)

    assert eval_case["name"] == "backend_engineer_germany"
    assert isinstance(eval_case["expected_keywords"], list)


def test_score_output_passes_for_valid_output():
    eval_case = load_eval_case(CASES_DIR / "backend_engineer_germany.json")
    score = score_output(_valid_output(), eval_case, metadata=_valid_output())

    assert score["passed"] is True
    assert score["total_score"] == score["max_score"]


def test_score_output_fails_when_required_fields_missing():
    eval_case = load_eval_case(CASES_DIR / "backend_engineer_germany.json")
    output = _valid_output()
    output["cover_letter_draft"] = ""

    score = score_output(output, eval_case, metadata=output)

    assert score["passed"] is False
    assert any(
        check["name"] == "required_sections_present" and not check["passed"]
        for check in score["checks"]
    )


def test_must_not_include_violations_are_detected():
    eval_case = load_eval_case(CASES_DIR / "backend_engineer_germany.json")
    output = _valid_output()
    output["cover_letter_draft"] += " SSN"

    score = score_output(output, eval_case, metadata=output)

    forbidden_check = next(
        check for check in score["checks"] if check["name"] == "must_not_include"
    )
    assert forbidden_check["passed"] is False
    assert "SSN" in forbidden_check["details"]


def test_keyword_coverage_contributes_to_score():
    eval_case = load_eval_case(CASES_DIR / "backend_engineer_germany.json")
    output = _valid_output()
    output["tailored_summary"] = "Short summary."
    output["tailored_bullets"] = ["General bullet one.", "General bullet two."]
    output["cover_letter_draft"] = "General letter body without target terms."
    output["application_question_answers"] = ["General answer one.", "General answer two."]
    output["recruiter_message_draft"] = "Hello."
    output["fit_gap_analysis"] = "General analysis."
    output["interview_talking_points"] = ["General point one.", "General point two."]

    score = score_output(output, eval_case, metadata=output)

    keyword_check = next(check for check in score["checks"] if check["name"] == "keyword_coverage")
    assert keyword_check["points"] < keyword_check["max_points"]
