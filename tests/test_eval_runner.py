from evals.run_eval import main, run_compare, run_workflow_mode


def test_eval_runner_runs_single_step_case(capsys):
    exit_code = main(["--workflow-mode", "single_step", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Workflow mode: single_step" in captured.out
    assert "backend_engineer_germany: PASS" in captured.out


def test_eval_runner_runs_agentic_case(capsys):
    exit_code = main(["--workflow-mode", "agentic", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Workflow mode: agentic" in captured.out
    assert "backend_engineer_germany: PASS" in captured.out


def test_run_workflow_mode_returns_scored_results():
    results = run_workflow_mode("single_step", case_name="backend_engineer_germany")

    assert len(results) == 1
    assert results[0]["score"]["passed"] is True
    assert results[0]["run_data"]["provider_used"] == "mock"


def test_compare_mode_runs_both_workflows(capsys):
    exit_code = main(["--compare", "--case", "backend_engineer_germany"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Compare mode" in captured.out
    assert "single_step:" in captured.out
    assert "agentic:" in captured.out


def test_run_compare_returns_single_step_and_agentic_results():
    results = run_compare(case_name="backend_engineer_germany")

    assert len(results) == 1
    assert results[0]["single_step"]["score"]["passed"] is True
    assert results[0]["agentic"]["score"]["passed"] is True
