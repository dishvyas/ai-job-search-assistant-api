import importlib.util
from pathlib import Path


def _load_demo_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "demo_tailoring_run.py"
    spec = importlib.util.spec_from_file_location("demo_tailoring_run", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_demo_output_for_completed_run():
    module = _load_demo_module()
    run_data = {
        "id": 7,
        "status": "completed",
        "provider_used": "mock",
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 12,
        "estimated_cost_usd": 0.0,
        "generation_attempts": 4,
        "route_decision": "proceed_to_tailoring",
        "revision_needed": False,
        "retrieved_context_count": 0,
        "artifact_context_count": 0,
        "tailored_summary": "Strong fit for the AI backend role.",
    }
    trace_data = {
        "steps": [
            {
                "step_name": "compose_final",
                "status": "completed",
                "provider_used": "mock",
                "fallback_used": False,
                "latency_ms": 2,
                "estimated_cost_usd": 0.0,
                "output_summary": "Generated final output.",
            }
        ]
    }

    output = module.build_demo_output(run_data, trace_data)

    assert "run_id: 7" in output
    assert "status: completed" in output
    assert "fallback_reason: None" in output
    assert "tailored_summary: Strong fit for the AI backend role." in output
    assert "compose_final: status=completed" in output


def test_build_demo_output_for_failed_run():
    module = _load_demo_module()
    run_data = {
        "id": 9,
        "status": "failed",
        "provider_used": None,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 5,
        "estimated_cost_usd": None,
        "generation_attempts": 1,
        "route_decision": None,
        "revision_needed": None,
        "retrieved_context_count": None,
        "artifact_context_count": None,
        "error_message": "Simulated processing failure",
    }

    output = module.build_demo_output(run_data)

    assert "status: failed" in output
    assert "error_message: Simulated processing failure" in output
