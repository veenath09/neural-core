
"""Task 12 Part C: Concurrent CrewAI execution tests.

Tests multiple local CrewAI Flows submitted from threads
using the same Pillar 4 ExecutionEngine.

No LLM calls or API keys are required.
"""

from concurrent.futures import ThreadPoolExecutor
import time

from crewai.flow.flow import Flow, start, listen

from controller.contracts import ModelCard, Strategy

from execution.adapters.crewai_methods import (
    CrewAIMethodAdapter,
)
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine
from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)
from execution.trace import validate_execution_trace


class ConcurrentResearchFlow(Flow):
    """A two-method local CrewAI Flow."""

    @start()
    def uppercase(self):
        # Encourages overlap between concurrent executions.
        time.sleep(0.05)

        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        time.sleep(0.02)

        return previous_result + "!"


def make_engine():
    """Build one shared execution engine."""

    card = ModelCard(
        model_id="concurrent-local-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-16",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    adapter = CrewAIMethodAdapter(
        flow_factory=ConcurrentResearchFlow,
        method_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def execute_one(engine, word):
    """Execute a single Flow with a unique identity."""

    strategy = Strategy(
        step_id=f"strategy-{word}",
        round=1,
        model_id="concurrent-local-model",
        halt="stop",
    )

    return engine.execute(
        strategy=strategy,
        input_text=word,
        execution_id=f"execution-{word}",
    )


def run_concurrently(words):
    """Submit several executions using one shared engine."""

    engine = make_engine()

    with ThreadPoolExecutor(
        max_workers=len(words)
    ) as pool:

        futures = {
            word: pool.submit(
                execute_one,
                engine,
                word,
            )
            for word in words
        }

        return {
            word: future.result(timeout=30)
            for word, future in futures.items()
        }


def test_concurrent_outputs_are_independent():
    """Each concurrent execution gets its own output."""

    words = ("apple", "banana", "cherry")

    results = run_concurrently(words)

    assert len(results) == 3

    for word in words:

        result = results[word]

        assert result.output_text == (
            word.upper() + "!"
        )

        assert result.trace.final_status == "ok"


def test_concurrent_traces_are_not_mixed():
    """Each Flow must capture only its own method events."""

    words = ("alpha", "beta", "gamma")

    results = run_concurrently(words)

    execution_ids = set()

    for word in words:

        trace = results[word].trace

        validate_execution_trace(trace)

        assert trace.execution_id == (
            f"execution-{word}"
        )

        assert trace.strategy_step_id == (
            f"strategy-{word}"
        )

        assert trace.framework == "crewai"

        # Exactly two methods per Flow.
        assert len(trace.steps) == 2

        assert [
            step.node_name
            for step in trace.steps
        ] == [
            "uppercase",
            "add_exclamation",
        ]

        assert trace.steps[0].dependencies == ()
        assert trace.steps[1].dependencies == (0,)

        assert all(
            step.status == "ok"
            for step in trace.steps
        )

        execution_ids.add(trace.execution_id)

    # All three execution IDs must be different.
    assert len(execution_ids) == 3


def test_concurrent_traces_support_json():
    """Concurrent execution records must remain serializable."""

    words = ("first", "second")

    results = run_concurrently(words)

    for word in words:

        original = results[word].trace

        data = trace_to_dict(original)

        restored = trace_from_dict(data)

        assert restored == original

        assert restored.execution_id == (
            f"execution-{word}"
        )
