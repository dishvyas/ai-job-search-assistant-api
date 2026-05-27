import argparse
import json
from contextlib import ExitStack
from datetime import UTC, datetime
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
REPORTS_DIR = Path(__file__).parent / "reports"


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


def _validate_provider(provider: str) -> None:
    """Fail fast for real-provider evals with missing required config."""
    if provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when running evals with --provider gemini.")


def _normalize_case_result(
    workflow_mode: str,
    provider: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten a run result into a report-friendly structure."""
    run_data = result["run_data"]
    score = result["score"]
    return {
        "case_name": result["case_name"],
        "workflow_mode": workflow_mode,
        "provider": provider,
        "passed": score["passed"],
        "total_score": score["total_score"],
        "max_score": score["max_score"],
        "latency_ms": run_data.get("latency_ms"),
        "estimated_cost_usd": run_data.get("estimated_cost_usd"),
        "generation_attempts": run_data.get("generation_attempts"),
        "fallback_used": run_data.get("fallback_used"),
        "route_decision": run_data.get("route_decision"),
        "revision_needed": run_data.get("revision_needed"),
        "retrieved_context_count": run_data.get("retrieved_context_count"),
        "artifact_context_count": run_data.get("artifact_context_count"),
        "run_id": result["run_id"],
        "checks": score["checks"],
    }


def build_report(
    *,
    provider: str,
    workflow_mode: str,
    rag_enabled: bool,
    artifact_retrieval_enabled: bool,
    results: list[dict[str, Any]],
    comparison_summaries: list[dict[str, Any]] | None = None,
    total_cases: int | None = None,
    passed_cases: int | None = None,
) -> dict[str, Any]:
    """Create a structured JSON-friendly eval report."""
    resolved_total_cases = total_cases if total_cases is not None else len(results)
    resolved_passed_cases = (
        passed_cases
        if passed_cases is not None
        else sum(1 for result in results if result["passed"])
    )
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": provider,
        "workflow_mode": workflow_mode,
        "rag_enabled": rag_enabled,
        "artifact_retrieval_enabled": artifact_retrieval_enabled,
        "total_cases": resolved_total_cases,
        "passed_cases": resolved_passed_cases,
        "failed_cases": resolved_total_cases - resolved_passed_cases,
        "results": results,
    }
    if comparison_summaries:
        report["comparison_summaries"] = comparison_summaries
    return report


def save_report(
    report: dict[str, Any],
    *,
    provider: str,
    workflow_mode: str,
) -> Path:
    """Persist a local JSON eval report and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{timestamp}_{provider}_{workflow_mode}.json"
    path = REPORTS_DIR / filename
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def run_workflow_mode(
    workflow_mode: str,
    *,
    provider: str = "mock",
    case_name: str | None = None,
    rag_enabled: bool = False,
    artifact_retrieval_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Run one or more eval cases against a single workflow mode."""
    _validate_provider(provider)
    eval_cases = discover_eval_cases(case_name=case_name)
    client, session, engine = _build_test_client()

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "llm_provider", provider))
            stack.enter_context(patch.object(settings, "workflow_mode", workflow_mode))
            stack.enter_context(patch.object(settings, "rag_enabled", rag_enabled))
            stack.enter_context(
                patch.object(
                    settings,
                    "artifact_retrieval_enabled",
                    artifact_retrieval_enabled,
                )
            )
            return [_run_case(client, eval_case) for eval_case in eval_cases]
    finally:
        _cleanup_test_client(client, session, engine)


def format_run_report(
    workflow_mode: str,
    provider: str,
    results: list[dict[str, Any]],
) -> str:
    lines = [f"Workflow mode: {workflow_mode}", f"Provider: {provider}", ""]
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


def run_compare(
    *,
    provider: str = "mock",
    case_name: str | None = None,
    rag_enabled: bool = False,
    artifact_retrieval_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Run each eval case once in each workflow mode."""
    single_step_results = run_workflow_mode(
        "single_step",
        provider=provider,
        case_name=case_name,
        rag_enabled=rag_enabled,
        artifact_retrieval_enabled=artifact_retrieval_enabled,
    )
    agentic_results = run_workflow_mode(
        "agentic",
        provider=provider,
        case_name=case_name,
        rag_enabled=rag_enabled,
        artifact_retrieval_enabled=artifact_retrieval_enabled,
    )

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


def format_compare_report(provider: str, results: list[dict[str, Any]]) -> str:
    lines = ["Compare mode", f"Provider: {provider}", ""]
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


def build_compare_report_payload(
    *,
    provider: str,
    rag_enabled: bool,
    artifact_retrieval_enabled: bool,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flatten compare-mode results and add simple per-case deltas."""
    flattened_results: list[dict[str, Any]] = []
    comparison_summaries: list[dict[str, Any]] = []

    for result in results:
        single_step = _normalize_case_result("single_step", provider, result["single_step"])
        agentic = _normalize_case_result("agentic", provider, result["agentic"])
        flattened_results.extend([single_step, agentic])
        comparison_summaries.append(
            {
                "case_name": result["case_name"],
                "score_delta": agentic["total_score"] - single_step["total_score"],
                "latency_delta_ms": (
                    (agentic["latency_ms"] or 0) - (single_step["latency_ms"] or 0)
                ),
                "cost_delta_usd": (
                    (agentic["estimated_cost_usd"] or 0.0)
                    - (single_step["estimated_cost_usd"] or 0.0)
                ),
                "attempts_delta": (
                    (agentic["generation_attempts"] or 0)
                    - (single_step["generation_attempts"] or 0)
                ),
            }
        )

    return build_report(
        provider=provider,
        workflow_mode="compare",
        rag_enabled=rag_enabled,
        artifact_retrieval_enabled=artifact_retrieval_enabled,
        results=flattened_results,
        comparison_summaries=comparison_summaries,
        total_cases=len(results),
        passed_cases=sum(
            1
            for result in results
            if result["single_step"]["score"]["passed"] and result["agentic"]["score"]["passed"]
        ),
    )


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
    parser.add_argument(
        "--provider",
        choices=("mock", "gemini"),
        default="mock",
        help="LLM provider to evaluate against. Defaults to mock.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save a structured JSON report under evals/reports/.",
    )
    parser.add_argument(
        "--rag-enabled",
        action="store_true",
        help="Enable job-description RAG during the eval run.",
    )
    parser.add_argument(
        "--artifact-retrieval-enabled",
        action="store_true",
        help="Enable tailored-artifact retrieval during the eval run.",
    )
    args = parser.parse_args(argv)

    try:
        if args.compare:
            results = run_compare(
                provider=args.provider,
                case_name=args.case,
                rag_enabled=args.rag_enabled,
                artifact_retrieval_enabled=args.artifact_retrieval_enabled,
            )
            print(format_compare_report(args.provider, results))
            if args.save_report:
                report = build_compare_report_payload(
                    provider=args.provider,
                    rag_enabled=args.rag_enabled,
                    artifact_retrieval_enabled=args.artifact_retrieval_enabled,
                    results=results,
                )
                path = save_report(report, provider=args.provider, workflow_mode="compare")
                print(f"\nSaved report: {path}")
            return (
                0
                if all(
                    result["single_step"]["score"]["passed"]
                    and result["agentic"]["score"]["passed"]
                    for result in results
                )
                else 1
            )

        results = run_workflow_mode(
            args.workflow_mode,
            provider=args.provider,
            case_name=args.case,
            rag_enabled=args.rag_enabled,
            artifact_retrieval_enabled=args.artifact_retrieval_enabled,
        )
        print(format_run_report(args.workflow_mode, args.provider, results))
        if args.save_report:
            report = build_report(
                provider=args.provider,
                workflow_mode=args.workflow_mode,
                rag_enabled=args.rag_enabled,
                artifact_retrieval_enabled=args.artifact_retrieval_enabled,
                results=[
                    _normalize_case_result(args.workflow_mode, args.provider, result)
                    for result in results
                ],
            )
            path = save_report(report, provider=args.provider, workflow_mode=args.workflow_mode)
            print(f"\nSaved report: {path}")
        return 0 if all(result["score"]["passed"] for result in results) else 1
    except ValueError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
