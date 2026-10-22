
"""Create or remove one synthetic dashboard integration-test trace.

This does NOT run AI models or execute the C4 engine.

The synthetic record is used only to test:
Python -> JSON -> FastAPI -> React.

Usage (from the repository root):
    python -m scripts.create_dashboard_test_trace
    python -m scripts.create_dashboard_test_trace --remove
"""

from __future__ import annotations

import argparse
from pathlib import Path

from execution.serialization import (
    load_trace_json,
    save_trace_json,
)
from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
    validate_execution_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACE_PATH = (
    PROJECT_ROOT
    / "runs"
    / "execution_traces"
    / "SYNTHETIC-dashboard-integration-001.json"
)


def create_synthetic_trace() -> ExecutionTrace:
    """Create a valid, explicitly synthetic execution trace."""

    steps = (
        ExecutionStep(
            step_index=0,
            parent_step=None,
            model_id="synthetic-planner",
            tool=None,
            input_tokens=120,
            output_tokens=50,
            cost_usd=0.00012,
            wall_ms=180.0,
            status="ok",
            node_name="Synthetic planning step",
        ),
        ExecutionStep(
            step_index=1,
            parent_step=0,
            model_id="synthetic-worker",
            tool="synthetic-tool",
            input_tokens=260,
            output_tokens=110,
            cost_usd=0.00034,
            wall_ms=420.0,
            status="ok",
            node_name="Synthetic worker step",
            parent_steps=(0,),
        ),
        ExecutionStep(
            step_index=2,
            parent_step=1,
            model_id="synthetic-reviewer",
            tool=None,
            input_tokens=160,
            output_tokens=70,
            cost_usd=0.00018,
            wall_ms=230.0,
            status="ok",
            node_name="Synthetic review step",
            parent_steps=(1,),
        ),
    )

    trace = ExecutionTrace(
        execution_id="SYNTHETIC-dashboard-integration-001",
        strategy_step_id="synthetic-strategy-step-001",
        strategy_round=1,
        framework="synthetic-integration-test",
        catalogue_version="synthetic-catalogue-v1",
        adapter_version="synthetic-adapter-v1",
        steps=steps,
        final_status="ok",
    )

    validate_execution_trace(trace)

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--remove",
        action="store_true",
        help="Delete only the synthetic dashboard test trace.",
    )

    args = parser.parse_args()

    if args.remove:
        if TRACE_PATH.exists():
            TRACE_PATH.unlink()
            print("Removed synthetic dashboard test trace.")
        else:
            print("No synthetic dashboard test trace to remove.")
        return

    if TRACE_PATH.exists():
        print("Synthetic test trace already exists.")
        print("Path:", TRACE_PATH)
        print("No existing file was overwritten.")
        return

    trace = create_synthetic_trace()

    saved_path = save_trace_json(
        trace,
        TRACE_PATH,
    )

    # Read the file back through the official C4 serializer.
    restored = load_trace_json(saved_path)

    assert restored == trace

    print("PASS: Synthetic execution trace validated")
    print("PASS: JSON serialization succeeded")
    print("PASS: JSON deserialization succeeded")
    print("PASS: Round-trip integrity verified")
    print()
    print("Execution ID:", restored.execution_id)
    print("Framework:", restored.framework)
    print("Steps:", len(restored.steps))
    print("Status:", restored.final_status)
    print("Saved path:", saved_path)
    print()
    print("WARNING: Synthetic integration-test data only.")
    print("This is NOT a real AI execution or research result.")


if __name__ == "__main__":
    main()
