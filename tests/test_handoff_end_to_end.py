
"""Task 15 Part B: Final Pillar 4 end-to-end verification.

Tests:
- Real LangGraph and CrewAI workflows
- Framework routing
- Controller Strategy compatibility
- Measurement handoff construction
- JSON file persistence
- JSON loading and validation
- Preservation of parallel dependencies

All workflows use local Python functions.
No LLM API keys or paid inference required.
"""

import json
from dataclasses import asdict
from typing import TypedDict

import pytest

from crewai.flow.flow import Flow, start, listen

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

from controller.contracts import ModelCard, Strategy

from execution.adapters.crewai_methods import (
    CrewAIMethodAdapter,
)

from execution.adapters.langgraph_nodes import (
    LangGraphNodeAdapter,
)

from execution.catalogue import ResourceCatalogue

from execution.handoff import build_measurement_handoff

from execution.handoff_serialization import (
    load_handoff_json,
    save_handoff_json,
    validate_measurement_handoff,
)

from execution.router import ExecutionRouter


# ---------------------------------------------------------
# 1. LangGraph parallel workflow
# ---------------------------------------------------------

class GraphState(TypedDict, total=False):
    input_text: str
    left_text: str
    right_text: str
    output_text: str


def build_langgraph():

    def left(state: GraphState):
        return {
            "left_text": state["input_text"].upper()
        }

    def right(state: GraphState):
        return {
            "right_text": state["input_text"].lower()
        }

    def join(state: GraphState):
        return {
            "output_text": (
                state["left_text"]
                + " | "
                + state["right_text"]
            )
        }

    builder = StateGraph(GraphState)

    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("join", join)

    builder.add_edge(START, "left")
    builder.add_edge(START, "right")

    builder.add_edge(
        ["left", "right"],
        "join",
    )

    builder.add_edge("join", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


# ---------------------------------------------------------
# 2. CrewAI sequential workflow
# ---------------------------------------------------------

class ResearchFlow(Flow):

    @start()
    def uppercase(self):
        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        return previous_result + "!"


# ---------------------------------------------------------
# 3. Shared execution router
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def router():

    card = ModelCard(
        model_id="final-local-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local integration test",
        price_date="2026-10-16",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    langgraph_adapter = LangGraphNodeAdapter(
        graph=build_langgraph(),
        node_parents={
            "left": None,
            "right": None,
            "join": ("left", "right"),
        },
    )

    crewai_adapter = CrewAIMethodAdapter(
        flow_factory=ResearchFlow,
        method_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )

    return ExecutionRouter(
        catalogue=catalogue,
        adapters={
            "langgraph": langgraph_adapter,
            "crewai": crewai_adapter,
        },
    )


# ---------------------------------------------------------
# 4. Controller Strategy
# ---------------------------------------------------------

def make_strategy():

    return Strategy(
        step_id="final-handoff-step",
        round=1,
        model_id="final-local-model",
        halt="stop",
    )


# ---------------------------------------------------------
# 5. Execute, save, and load
# ---------------------------------------------------------

def execute_save_load(router, framework, tmp_path):

    strategy = make_strategy()

    strategy_before = asdict(strategy)

    result = router.execute(
        strategy=strategy,
        input_text="hello research",
        execution_id=f"final-{framework}-001",
        framework=framework,
    )

    latency_basis = {
        "langgraph": "node_event_observed",
        "crewai": "method_event_observed",
    }[framework]

    # These specific workflows contain no LLM calls.
    handoff = build_measurement_handoff(
        result,
        token_accounting="not_applicable",
        cost_accounting="not_applicable",
        latency_basis=latency_basis,
    )

    file_path = (
        tmp_path
        / "measurement_handoffs"
        / f"{framework}.json"
    )

    save_handoff_json(
        handoff,
        file_path,
    )

    restored = load_handoff_json(
        file_path
    )

    return {
        "result": result,
        "original": handoff,
        "restored": restored,
        "file_path": file_path,
        "strategy_before": strategy_before,
        "strategy_after": asdict(strategy),
    }


# ---------------------------------------------------------
# Test 1 and 2:
# Both frameworks complete the end-to-end pipeline
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "framework",
    ["langgraph", "crewai"],
)
def test_framework_end_to_end(
    router,
    framework,
    tmp_path,
):

    data = execute_save_load(
        router,
        framework,
        tmp_path,
    )

    result = data["result"]
    original = data["original"]
    restored = data["restored"]

    expected_output = {
        "langgraph": (
            "HELLO RESEARCH | hello research"
        ),
        "crewai": "HELLO RESEARCH!",
    }[framework]

    assert result.output_text == expected_output
    assert result.trace.final_status == "ok"

    assert restored == original

    assert restored.execution_id == (
        f"final-{framework}-001"
    )

    assert restored.framework == framework
    assert restored.final_status == "ok"

    assert restored.input_tokens == 0
    assert restored.output_tokens == 0
    assert restored.total_cost_usd == 0.0

    assert restored.token_accounting == (
        "not_applicable"
    )

    assert restored.cost_accounting == (
        "not_applicable"
    )

    assert data["file_path"].exists()

    assert (
        data["strategy_before"]
        == data["strategy_after"]
    )

    validate_measurement_handoff(restored)


# ---------------------------------------------------------
# Test 3:
# LangGraph parallel dependencies survive JSON persistence
# ---------------------------------------------------------

def test_parallel_dependencies_survive_persistence(
    router,
    tmp_path,
):

    data = execute_save_load(
        router,
        "langgraph",
        tmp_path,
    )

    trace = data["restored"].trace

    assert len(trace.steps) == 3

    steps = {
        step.node_name: step
        for step in trace.steps
    }

    assert set(steps) == {
        "left",
        "right",
        "join",
    }

    assert steps["left"].dependencies == ()
    assert steps["right"].dependencies == ()

    expected_parents = {
        steps["left"].step_index,
        steps["right"].step_index,
    }

    assert set(
        steps["join"].dependencies
    ) == expected_parents

    assert len(
        steps["join"].dependencies
    ) == 2


# ---------------------------------------------------------
# Test 4:
# CrewAI method dependencies survive JSON persistence
# ---------------------------------------------------------

def test_crewai_dependencies_survive_persistence(
    router,
    tmp_path,
):

    data = execute_save_load(
        router,
        "crewai",
        tmp_path,
    )

    trace = data["restored"].trace

    assert len(trace.steps) == 2

    first, second = trace.steps

    assert first.node_name == "uppercase"

    assert second.node_name == (
        "add_exclamation"
    )

    assert first.dependencies == ()
    assert second.dependencies == (0,)


# ---------------------------------------------------------
# Test 5:
# Saved JSON retains measurement provenance
# ---------------------------------------------------------

def test_saved_json_preserves_provenance(
    router,
    tmp_path,
):

    data = execute_save_load(
        router,
        "langgraph",
        tmp_path,
    )

    saved = json.loads(
        data["file_path"].read_text(
            encoding="utf-8"
        )
    )

    assert saved["handoff_version"] == "v1"

    assert saved["token_accounting"] == (
        "not_applicable"
    )

    assert saved["cost_accounting"] == (
        "not_applicable"
    )

    assert saved["latency_basis"] == (
        "node_event_observed"
    )

    assert saved["trace"]["framework"] == (
        "langgraph"
    )

    assert len(
        saved["trace"]["steps"]
    ) == 3
