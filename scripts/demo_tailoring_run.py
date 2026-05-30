from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_REQUEST_FILE = Path(__file__).resolve().parents[1] / "demo" / "tailor_request.json"


def load_request_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def submit_tailoring_request(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    return http_json("POST", f"{base_url}/api/v1/applications/tailor", payload)


def fetch_run(base_url: str, run_id: int) -> dict[str, Any]:
    return http_json("GET", f"{base_url}/api/v1/applications/runs/{run_id}")


def fetch_trace(base_url: str, run_id: int) -> dict[str, Any]:
    return http_json("GET", f"{base_url}/api/v1/applications/runs/{run_id}/trace")


def poll_run_until_terminal(
    base_url: str,
    run_id: int,
    *,
    poll_interval: float = 0.25,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run_data = fetch_run(base_url, run_id)
        if run_data.get("status") in {"completed", "failed"}:
            return run_data
        time.sleep(poll_interval)
    raise TimeoutError(f"Run {run_id} did not reach a terminal state within {timeout_seconds}s")


def build_demo_output(run_data: dict[str, Any], trace_data: dict[str, Any] | None = None) -> str:
    lines = [
        f"run_id: {run_data.get('id')}",
        f"status: {run_data.get('status')}",
        f"provider_used: {run_data.get('provider_used')}",
        f"fallback_used: {run_data.get('fallback_used')}",
        f"fallback_reason: {run_data.get('fallback_reason')}",
        f"latency_ms: {run_data.get('latency_ms')}",
        f"estimated_cost_usd: {run_data.get('estimated_cost_usd')}",
        f"generation_attempts: {run_data.get('generation_attempts')}",
        f"route_decision: {run_data.get('route_decision')}",
        f"revision_needed: {run_data.get('revision_needed')}",
        f"retrieved_context_count: {run_data.get('retrieved_context_count')}",
        f"artifact_context_count: {run_data.get('artifact_context_count')}",
    ]

    if run_data.get("status") == "failed":
        lines.append(f"error_message: {run_data.get('error_message')}")
    else:
        lines.append(f"tailored_summary: {_preview(run_data.get('tailored_summary'))}")

    if trace_data is not None:
        lines.append("")
        lines.append("trace:")
        for step in trace_data.get("steps", []):
            lines.append(
                "  - "
                f"{step.get('step_name')}: status={step.get('status')} "
                f"provider={step.get('provider_used')} "
                f"fallback={step.get('fallback_used')} "
                f"latency_ms={step.get('latency_ms')} "
                f"estimated_cost_usd={step.get('estimated_cost_usd')} "
                f"output={_preview(step.get('output_summary'))}"
            )

    return "\n".join(lines)


def _preview(value: Any, limit: int = 160) -> str:
    if value is None:
        return "None"
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a demo tailoring request against the API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--request-file", default=str(DEFAULT_REQUEST_FILE))
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        payload = load_request_payload(args.request_file)
        job_response = submit_tailoring_request(args.base_url, payload)
        run_id = int(job_response["run_id"])
        run_data = poll_run_until_terminal(args.base_url, run_id)
        trace_data = fetch_trace(args.base_url, run_id) if args.show_trace else None
        print(build_demo_output(run_data, trace_data))
        return 0
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
