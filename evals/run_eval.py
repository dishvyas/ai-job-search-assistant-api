import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app

try:
    from evals.scoring import load_eval_case, score_output
except ModuleNotFoundError:  # pragma: no cover - used only when running as a script path
    from scoring import load_eval_case, score_output

CASES_DIR = Path(__file__).parent / "cases"


def discover_eval_cases(case_name: str | None = None) -> list[dict[str, Any]]:
    """Load all eval cases or one named case."""
    case_paths = sorted(CASES_DIR.glob("*.json"))
    if case_name is not None:
        case_paths = [path for path in case_paths if path.stem == case_name]
        if not case_paths:
            raise ValueError(f"Eval case {case_name!r} not found in {CASES_DIR}")
    return [load_eval_case(path) for path in case_paths]


def _build_test_client() -> tuple[TestClient, Any, Any]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()

    def _test_get_db():
        yield session

    app.dependency_overrides[get_db] = _test_get_db
    client = TestClient(app)
    return client, session, engine


def _cleanup_test_client(client: TestClient, session: Any, engine: Any) -> None:
    client.close()
    app.dependency_overrides.pop(get_db, None)
    session.close()
    engine.dispose()


def _run_case(client: TestClient, eval_case: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "master_resume": eval_case["master_resume"],
        "job_description": eval_case["job_description"],
        "company_info": eval_case["company_info"],
        "user_preferences": eval_case["user_preferences"],
    }

    post_response = client.post("/api/v1/applications/tailor", json=payload)
    if post_response.status_code != 200:
        return {
            "case_name": eval_case["name"],
            "run_id": None,
            "run_data": {},
            "score": {
                "total_score": 0,
                "max_score": 1,
                "passed": False,
                "checks": [
                    {
                        "name": "request_succeeded",
                        "passed": False,
                        "points": 0,
                        "max_points": 1,
                        "details": f"POST /tailor returned {post_response.status_code}",
                    }
                ],
            },
        }

    run_id = post_response.json()["run_id"]
    get_response = client.get(f"/api/v1/applications/runs/{run_id}")
    run_data = get_response.json()
    score = score_output(run_data, eval_case, metadata=run_data)
    return {
        "case_name": eval_case["name"],
        "run_id": run_id,
        "run_data": run_data,
        "score": score,
    }


def run_workflow_mode(
    workflow_mode: str,
    case_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run one or more eval cases against a single workflow mode."""
    eval_cases = discover_eval_cases(case_name=case_name)
    client, session, engine = _build_test_client()

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "llm_provider", "mock"))
            stack.enter_context(patch.object(settings, "workflow_mode", workflow_mode))
            stack.enter_context(patch.object(settings, "rag_enabled", False))
            return [_run_case(client, eval_case) for eval_case in eval_cases]
    finally:
        _cleanup_test_client(client, session, engine)


def format_run_report(workflow_mode: str, results: list[dict[str, Any]]) -> str:
    lines = [f"Workflow mode: {workflow_mode}", ""]
    passed_count = 0

    for result in results:
        score = result["score"]
        run_data = result["run_data"]
        status = "PASS" if score["passed"] else "FAIL"
        if score["passed"]:
            passed_count += 1
        lines.append(
            f"- {result['case_name']}: {status} "
            f"({score['total_score']}/{score['max_score']}) "
            f"latency={run_data.get('latency_ms')}ms "
            f"attempts={run_data.get('generation_attempts')} "
            f"provider={run_data.get('provider_used')}"
        )
        for check in score["checks"]:
            marker = "ok" if check["passed"] else "x"
            lines.append(
                f"  [{marker}] {check['name']}: "
                f"{check['points']}/{check['max_points']} - {check['details']}"
            )

    lines.extend(
        [
            "",
            f"Overall: {passed_count}/{len(results)} cases passed",
        ]
    )
    return "\n".join(lines)


def run_compare(case_name: str | None = None) -> list[dict[str, Any]]:
    """Run each eval case once in each workflow mode."""
    single_step_results = run_workflow_mode("single_step", case_name=case_name)
    agentic_results = run_workflow_mode("agentic", case_name=case_name)

    by_case = {result["case_name"]: {"single_step": result} for result in single_step_results}
    for result in agentic_results:
        by_case.setdefault(result["case_name"], {})["agentic"] = result

    return [
        {
            "case_name": case_name_key,
            "single_step": result_pair["single_step"],
            "agentic": result_pair["agentic"],
        }
        for case_name_key, result_pair in sorted(by_case.items())
    ]


def format_compare_report(results: list[dict[str, Any]]) -> str:
    lines = ["Compare mode", ""]
    passed = True

    for result in results:
        single_step = result["single_step"]
        agentic = result["agentic"]
        if not single_step["score"]["passed"] or not agentic["score"]["passed"]:
            passed = False
        lines.append(f"- {result['case_name']}")
        lines.append(
            "  single_step: "
            f"{single_step['score']['total_score']}/{single_step['score']['max_score']} "
            f"latency={single_step['run_data'].get('latency_ms')}ms "
            f"cost={single_step['run_data'].get('estimated_cost_usd')} "
            f"attempts={single_step['run_data'].get('generation_attempts')}"
        )
        lines.append(
            "  agentic:     "
            f"{agentic['score']['total_score']}/{agentic['score']['max_score']} "
            f"latency={agentic['run_data'].get('latency_ms')}ms "
            f"cost={agentic['run_data'].get('estimated_cost_usd')} "
            f"attempts={agentic['run_data'].get('generation_attempts')}"
        )

    lines.extend(["", f"Overall: {'PASS' if passed else 'FAIL'}"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local workflow eval cases.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--workflow-mode",
        choices=("single_step", "agentic"),
        help="Run evals for one workflow mode.",
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Run each case in both single_step and agentic modes.",
    )
    parser.add_argument(
        "--case",
        help="Optional case name without the .json extension.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        results = run_compare(case_name=args.case)
        print(format_compare_report(results))
        return (
            0
            if all(
                result["single_step"]["score"]["passed"] and result["agentic"]["score"]["passed"]
                for result in results
            )
            else 1
        )

    results = run_workflow_mode(args.workflow_mode, case_name=args.case)
    print(format_run_report(args.workflow_mode, results))
    return 0 if all(result["score"]["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
