
"""Task 12 Part B: LangGraph event integrity tests.

Tests incomplete, invalid, and failed task-event streams.

Uses a synthetic graph event stream so missing events can
be tested without depending on LangGraph internals.
"""

from controller.contracts import ModelCard, Strategy

from execution.adapters.base import ExecutionRequest
from execution.adapters.langgraph_nodes import (
    LangGraphNodeAdapter,
)
from execution.catalogue import ResourceEntry
from execution.trace import validate_execution_trace


class FakeEventGraph:
    """A controllable substitute for LangGraph's event stream."""

    def __init__(self, events):
        self.events = events

    def stream(self, *args, **kwargs):
        yield from self.events


def start_event(task_id, node_name):
    """Simulate a LangGraph task-start event."""

    return {
        "type": "tasks",
        "data": {
            "id": task_id,
            "name": node_name,
            "input": {
                "input_text": "hello research",
            },
        },
    }


def finish_event(task_id, node_name, error=None):
    """Simulate a LangGraph task-finish event."""

    return {
        "type": "tasks",
        "data": {
            "id": task_id,
            "name": node_name,
            "error": error,
        },
    }


def state_event(output_text="HELLO RESEARCH"):
    """Simulate a completed LangGraph state update."""

    return {
        "type": "values",
        "data": {
            "input_text": "hello research",
            "output_text": output_text,
        },
    }


def make_request():
    """Create the standard execution request."""

    card = ModelCard(
        model_id="local-langgraph-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-16",
        capability=[0.5, -10.0],
    )

    strategy = Strategy(
        step_id="langgraph-event-step",
        round=1,
        model_id="local-langgraph-model",
        halt="stop",
    )

    return ExecutionRequest(
        execution_id="langgraph-event-001",
        strategy=strategy,
        resource=ResourceEntry(card=card),
        catalogue_version="v1",
        input_text="hello research",
    )


def execute_events(events, node_parents=None):
    """Execute one synthetic LangGraph event stream."""

    if node_parents is None:
        node_parents = {
            "uppercase": None,
        }

    adapter = LangGraphNodeAdapter(
        graph=FakeEventGraph(events),
        node_parents=node_parents,
    )

    return adapter.execute(make_request())


def test_complete_execution_succeeds():
    """A complete node execution produces a successful trace."""

    events = [
        start_event("task-1", "uppercase"),
        finish_event("task-1", "uppercase"),
        state_event(),
    ]

    result = execute_events(events)

    assert result.output_text == "HELLO RESEARCH"
    assert result.trace.final_status == "ok"

    assert len(result.trace.steps) == 1
    assert result.trace.steps[0].node_name == "uppercase"
    assert result.trace.steps[0].status == "ok"

    validate_execution_trace(result.trace)


def test_missing_finish_event_is_error():
    """A started task cannot be considered successful."""

    events = [
        start_event("task-1", "uppercase"),
    ]

    result = execute_events(events)

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    assert any(
        step.status == "error"
        for step in result.trace.steps
    )

    validate_execution_trace(result.trace)


def test_missing_start_event_is_error():
    """A task cannot finish without a matching start."""

    events = [
        finish_event("task-1", "uppercase"),
        state_event(),
    ]

    result = execute_events(events)

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    validate_execution_trace(result.trace)


def test_no_task_events_is_error():
    """State events alone do not prove a node executed."""

    events = [
        state_event(),
    ]

    result = execute_events(events)

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    assert any(
        step.status == "error"
        for step in result.trace.steps
    )

    validate_execution_trace(result.trace)


def test_failed_node_is_recorded():
    """A LangGraph node failure must appear in the trace."""

    events = [
        start_event("task-1", "uppercase"),
        finish_event(
            "task-1",
            "uppercase",
            error=RuntimeError("Deliberate failure"),
        ),
    ]

    result = execute_events(events)

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    assert result.trace.steps[0].status == "error"

    validate_execution_trace(result.trace)


def test_duplicate_finish_event_is_error():
    """A task must not finish twice."""

    events = [
        start_event("task-1", "uppercase"),
        finish_event("task-1", "uppercase"),
        finish_event("task-1", "uppercase"),
        state_event(),
    ]

    result = execute_events(events)

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    validate_execution_trace(result.trace)


def test_missing_parent_completion_is_error():
    """A child cannot record a dependency on an unfinished parent."""

    events = [
        start_event("task-2", "child"),
        finish_event("task-2", "child"),
        state_event(),
    ]

    result = execute_events(
        events,
        node_parents={
            "parent": None,
            "child": "parent",
        },
    )

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    validate_execution_trace(result.trace)


def test_multiple_unfinished_nodes_is_error():
    """A stream ending with parallel unfinished nodes must fail."""

    events = [
        start_event("task-1", "left"),
        start_event("task-2", "right"),
    ]

    result = execute_events(
        events,
        node_parents={
            "left": None,
            "right": None,
        },
    )

    assert result.trace.final_status == "error"
    assert result.output_text == ""

    validate_execution_trace(result.trace)
