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


def test_generated_reports_are_ignored_by_git():
    gitignore_contents = REPORTS_DIR.parent.parent.joinpath(".gitignore").read_text()
    assert "evals/reports/*.json" in gitignore_contents
