
"""Run a genuine local Python task through the C4 ExecutionEngine.

This is a local callable execution, NOT an LLM or AI-agent run.

The existing CallableAdapter measures runtime and records:
- Execution success/failure
- Actual callable wall time
- Approximate whitespace-based token counts
- Zero API cost

Usage:
    python -m scripts.run_dashboard_local_execution
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

from controller.contracts import (
    ModelCard,
    Strategy,
    validate_strategy,
)

from execution.adapters.callable import CallableAdapter
from execution.catalogue import (
    ResourceCatalogue,
    validate_resource_catalogue,
)
from execution.engine import ExecutionEngine
from execution.serialization import (
    load_trace_json,
    save_trace_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACE_DIRECTORY = (
    PROJECT_ROOT / "runs" / "execution_traces"
)

MODEL_ID = "local-text-normalizer"


def normalize_text(input_text: str) -> str:
    """Perform real local text processing using Python."""

    normalized = " ".join(input_text.split())

    return normalized.upper()


def main() -> None:
    # A local callable is a non-LLM execution resource.
    # It has no external model API cost.
    model_card = ModelCard(
        model_id=MODEL_ID,
        provider="local-python",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local-offline-no-api-billing",
        price_date=date.today().isoformat(),
        capability=[],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [model_card]
    )

    validate_resource_catalogue(catalogue)

    strategy = Strategy(
        step_id="dashboard-local-processing",
        round=1,
        model_id=MODEL_ID,
        halt="stop",
        tools=[],
        memory="none",
        depth="low",
        verification="none",
        topology="single",
        complexity=0.0,
        confidence=1.0,
    )

    validate_strategy(
        strategy,
        [model_card],
    )

    adapter = CallableAdapter(
        runner=normalize_text,
    )

    engine = ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )

    input_text = (
        "Neural-Core uses an execution engine "
        "to process local Python tasks."
    )

    execution_id = (
        "LOCAL-CALLABLE-" + uuid4().hex[:12]
    )

    # This invokes the genuine Pillar 4 execution engine.
    result = engine.execute(
        strategy=strategy,
        input_text=input_text,
        execution_id=execution_id,
    )

    # Never overwrite an existing trace.
    TRACE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        TRACE_DIRECTORY / f"{execution_id}.json"
    )

    if destination.exists():
        raise FileExistsError(
            f"Trace already exists: {destination}"
        )

    saved_path = save_trace_json(
        result.trace,
        destination,
    )

    # Verify using the existing official serializer.
    restored = load_trace_json(saved_path)

    assert restored == result.trace

    print()
    print("=" * 60)
    print("NEURAL-CORE C4 LOCAL EXECUTION")
    print("=" * 60)
    print()

    print("Execution ID:", execution_id)
    print("Framework:", result.trace.framework)
    print("Status:", result.trace.final_status)
    print("Steps:", len(result.trace.steps))
    print()

    print("Input:")
    print(input_text)
    print()

    print("Actual Python output:")
    print(result.output_text)
    print()

    for step in result.trace.steps:
        print("Recorded execution step:", step.step_index)
        print("Resource:", step.model_id)
        print("Status:", step.status)
        print("Input tokens (estimated):", step.input_tokens)
        print("Output tokens (estimated):", step.output_tokens)
        print("Measured callable time (ms):", step.wall_ms)
        print("Recorded API cost (USD):", step.cost_usd)

    print()
    print("Saved trace:", saved_path)
    print()
    print("PASS: C4 ExecutionEngine executed a local function.")
    print("PASS: C4 CallableAdapter generated its runtime trace.")
    print("PASS: Trace serialization and reload succeeded.")
    print()
    print("NOTE: Local Python execution, not an LLM run.")
    print("NOTE: Token counts are approximate.")
    print("NOTE: No external AI service was called.")


if __name__ == "__main__":
    main()
