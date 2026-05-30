import json

from evals.run_eval import REPORTS_DIR, main, run_compare, run_workflow_mode


def test_eval_runner_runs_single_step_case(capsys):
    exit_code = main(["--workflow-mode", "single_step", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Workflow mode: single_step" in captured.out
    assert "Provider: mock" in captured.out
    assert "backend_engineer_germany: PASS" in captured.out


def test_eval_runner_runs_agentic_case(capsys):
    exit_code = main(["--workflow-mode", "agentic", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Workflow mode: agentic" in captured.out
    assert "Provider: mock" in captured.out
    assert "backend_engineer_germany: PASS" in captured.out


def test_default_provider_is_mock():
    results = run_workflow_mode("single_step", case_name="backend_engineer_germany")

    assert len(results) == 1
    assert results[0]["run_data"]["provider_used"] == "mock"


def test_run_workflow_mode_returns_scored_results():
    results = run_workflow_mode("single_step", case_name="backend_engineer_germany")

    assert len(results) == 1
    assert results[0]["score"]["passed"] is True
    assert results[0]["run_data"]["provider_used"] == "mock"


def test_provider_mock_preserves_existing_behavior():
    results = run_workflow_mode(
        "single_step",
        provider="mock",
        case_name="backend_engineer_germany",
    )

    assert len(results) == 1
    assert results[0]["score"]["passed"] is True
    assert results[0]["run_data"]["provider_used"] == "mock"


def test_provider_gemini_without_api_key_exits_early(capsys, monkeypatch):
    monkeypatch.setattr("evals.run_eval.settings.gemini_api_key", None)

    exit_code = main(
        [
            "--provider",
            "gemini",
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GEMINI_API_KEY is required" in captured.out


def test_provider_openai_without_api_key_exits_early(capsys, monkeypatch):
    monkeypatch.setattr("evals.run_eval.settings.openai_api_key", None)

    exit_code = main(
        [
            "--provider",
            "openai",
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OPENAI_API_KEY is required" in captured.out


def test_provider_openai_runs_with_fake_client(monkeypatch):
    class FakeResponses:
        def create(self, model: str, input: str):
            return type(
                "FakeResponse",
                (),
                {
                    "output_text": (
                        '{"tailored_summary":"OpenAI summary","tailored_bullets":["b1","b2"],'
                        '"cover_letter_draft":"letter","application_question_answers":["a1"],'
                        '"recruiter_message_draft":"msg","fit_gap_analysis":"fit",'
                        '"interview_talking_points":["p1","p2"]}'
                    )
                },
            )()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr("evals.run_eval.settings.openai_api_key", "test-key")
    monkeypatch.setattr(
        "app.llm.openai.OpenAILLMProvider._build_client",
        lambda self, api_key: FakeClient(),
    )

    results = run_workflow_mode(
        "single_step",
        provider="openai",
        case_name="backend_engineer_germany",
    )

    assert len(results) == 1
    assert results[0]["run_data"]["provider_used"] == "openai"


def test_compare_mode_runs_both_workflows(capsys):
    exit_code = main(["--compare", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Compare mode" in captured.out
    assert "Provider: mock" in captured.out
    assert "single_step:" in captured.out
    assert "agentic:" in captured.out


def test_run_compare_returns_single_step_and_agentic_results():
    results = run_compare(case_name="backend_engineer_germany")

    assert len(results) == 1
    assert results[0]["single_step"]["score"]["passed"] is True
    assert results[0]["agentic"]["score"]["passed"] is True


def test_save_report_writes_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)

    exit_code = main(
        [
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
            "--save-report",
        ]
    )
    captured = capsys.readouterr()

    report_files = list(tmp_path.glob("eval_report_*_mock_single_step.json"))
    assert exit_code == 0
    assert "Saved report:" in captured.out
    assert len(report_files) == 1


def test_saved_report_contains_required_top_level_fields(monkeypatch, tmp_path):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)

    exit_code = main(
        [
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
            "--save-report",
        ]
    )

    report_path = next(tmp_path.glob("eval_report_*_mock_single_step.json"))
    report = json.loads(report_path.read_text())

    assert exit_code == 0
    assert report["provider"] == "mock"
    assert report["workflow_mode"] == "single_step"
    assert report["rag_enabled"] is False
    assert report["artifact_retrieval_enabled"] is False
    assert report["total_cases"] == 1
    assert "results" in report


def test_saved_report_contains_per_case_results(monkeypatch, tmp_path):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)
    main(
        [
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
            "--save-report",
        ]
    )

    report_path = next(tmp_path.glob("eval_report_*_mock_single_step.json"))
    report = json.loads(report_path.read_text())
    result = report["results"][0]

    assert result["case_name"] == "backend_engineer_germany"
    assert result["workflow_mode"] == "single_step"
    assert result["provider"] == "mock"
    assert "route_decision" in result
    assert "revision_needed" in result
    assert "retrieved_context_count" in result
    assert "artifact_context_count" in result
    assert "fallback_reason" in result
    assert "checks" in result


def test_compare_report_includes_both_workflow_modes(monkeypatch, tmp_path):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)

    exit_code = main(["--compare", "--case", "backend_engineer_germany", "--save-report"])

    report_path = next(tmp_path.glob("eval_report_*_mock_compare.json"))
    report = json.loads(report_path.read_text())
    workflow_modes = {result["workflow_mode"] for result in report["results"]}

    assert exit_code == 0
    assert report["total_cases"] == 1
    assert workflow_modes == {"single_step", "agentic"}
    assert "comparison_summaries" in report
    assert report["comparison_summaries"][0]["case_name"] == "backend_engineer_germany"


def test_agentic_report_includes_agent_decision_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)

    exit_code = main(
        [
            "--workflow-mode",
            "agentic",
            "--case",
            "backend_engineer_germany",
            "--save-report",
        ]
    )

    report_path = next(tmp_path.glob("eval_report_*_mock_agentic.json"))
    report = json.loads(report_path.read_text())
    result = report["results"][0]

    assert exit_code == 0
    assert result["route_decision"] is not None
    assert result["revision_needed"] is not None
    assert result["retrieved_context_count"] is not None
    assert result["artifact_context_count"] is not None


def test_generated_reports_are_ignored_by_git():
    gitignore_contents = REPORTS_DIR.parent.parent.joinpath(".gitignore").read_text()
    assert "evals/reports/*.json" in gitignore_contents


def test_saved_report_includes_fallback_reason(monkeypatch, tmp_path):
    monkeypatch.setattr("evals.run_eval.REPORTS_DIR", tmp_path)

    fake_results = [
        {
            "case_name": "backend_engineer_germany",
            "run_id": 123,
            "run_data": {
                "latency_ms": 12,
                "estimated_cost_usd": 0.0,
                "generation_attempts": 2,
                "fallback_used": True,
                "fallback_reason": "LLMProviderUnavailableError: OpenAI request failed",
                "route_decision": None,
                "revision_needed": None,
                "retrieved_context_count": None,
                "artifact_context_count": None,
                "provider_used": "fallback-mock",
            },
            "score": {
                "passed": True,
                "total_score": 10,
                "max_score": 10,
                "checks": [],
            },
        }
    ]

    monkeypatch.setattr("evals.run_eval.run_workflow_mode", lambda *args, **kwargs: fake_results)

    exit_code = main(
        [
            "--workflow-mode",
            "single_step",
            "--case",
            "backend_engineer_germany",
            "--save-report",
        ]
    )

    report_path = next(tmp_path.glob("eval_report_*_mock_single_step.json"))
    report = json.loads(report_path.read_text())
    result = report["results"][0]

    assert exit_code == 0
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "LLMProviderUnavailableError: OpenAI request failed"
