
"""Task 12 Part B: CrewAI event integrity tests.

Tests missing, duplicated, unordered, failed, and invalid
method events without making LLM API calls.
"""

from datetime import datetime, timedelta, timezone

import pytest

from controller.contracts import ModelCard, Strategy
from execution.adapters.base import ExecutionRequest
from execution.adapters.crewai_methods import CrewAIMethodAdapter
from execution.catalogue import ResourceEntry


BASE_TIME = datetime(
    2026, 10, 16,
    tzinfo=timezone.utc,
)


def make_event(
    sequence: int,
    milliseconds: int,
    phase: str,
    method_name: str,
):
    """Create one synthetic CrewAI method event."""

    timestamp = BASE_TIME + timedelta(
        milliseconds=milliseconds
    )

    error_type = (
        "RuntimeError"
        if phase == "failed"
        else None
    )

    return (
        sequence,
        timestamp,
        phase,
        method_name,
        error_type,
    )


def make_adapter():
    """Create an adapter for event parsing tests."""

    # These tests inspect only _build_steps.
    # We do not execute a real Flow here.
    return CrewAIMethodAdapter(
        flow_factory=lambda: None,
        method_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )


def make_request():
    """Build a valid execution request."""

    card = ModelCard(
        model_id="local-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-16",
        capability=[0.5, -10.0],
    )

    strategy = Strategy(
        step_id="event-test-step",
        round=1,
        model_id="local-model",
        halt="stop",
    )

    return ExecutionRequest(
        execution_id="event-integrity-001",
        strategy=strategy,
        resource=ResourceEntry(card=card),
        catalogue_version="v1",
        input_text="hello research",
    )


def build_steps(records):
    """Convert synthetic events into execution steps."""

    adapter = make_adapter()

    return adapter._build_steps(
        records,
        make_request(),
    )


def test_valid_sequential_events():
    """A complete two-method Flow is valid."""

    records = [
        make_event(1, 0, "start", "uppercase"),
        make_event(2, 10, "finish", "uppercase"),
        make_event(3, 10, "start", "add_exclamation"),
        make_event(4, 15, "finish", "add_exclamation"),
    ]

    steps = build_steps(records)

    assert len(steps) == 2

    assert steps[0].node_name == "uppercase"
    assert steps[1].node_name == "add_exclamation"

    assert steps[0].dependencies == ()
    assert steps[1].dependencies == (0,)

    assert steps[0].wall_ms == 10.0
    assert steps[1].wall_ms == 5.0


def test_events_out_of_arrival_order():
    """Emission sequence determines event order."""

    records = [
        make_event(4, 15, "finish", "add_exclamation"),
        make_event(2, 10, "finish", "uppercase"),
        make_event(1, 0, "start", "uppercase"),
        make_event(3, 10, "start", "add_exclamation"),
    ]

    steps = build_steps(records)

    assert len(steps) == 2
    assert steps[0].node_name == "uppercase"
    assert steps[1].node_name == "add_exclamation"


def test_missing_start_event_rejected():
    """A FINISH event requires a matching START."""

    records = [
        make_event(2, 10, "finish", "uppercase"),
    ]

    with pytest.raises(
        ValueError,
        match="without a start event",
    ):
        build_steps(records)


def test_missing_finish_event_rejected():
    """A START without FINISH is incomplete."""

    records = [
        make_event(1, 0, "start", "uppercase"),
    ]

    with pytest.raises(
        ValueError,
        match="Unmatched CrewAI method start events",
    ):
        build_steps(records)


def test_duplicate_event_sequence_rejected():
    """Two events cannot share one emission sequence."""

    records = [
        make_event(1, 0, "start", "uppercase"),
        make_event(1, 10, "finish", "uppercase"),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate CrewAI event sequence",
    ):
        build_steps(records)


def test_repeated_method_rejected():
    """Repeated methods are unsupported in adapter v1."""

    records = [
        make_event(1, 0, "start", "uppercase"),
        make_event(2, 10, "finish", "uppercase"),
        make_event(3, 12, "start", "uppercase"),
        make_event(4, 20, "finish", "uppercase"),
    ]

    with pytest.raises(
        ValueError,
        match="Repeated method",
    ):
        build_steps(records)


def test_missing_parent_completion_rejected():
    """A dependent method cannot finish before its parent."""

    records = [
        make_event(1, 0, "start", "add_exclamation"),
        make_event(2, 10, "finish", "add_exclamation"),
    ]

    with pytest.raises(
        ValueError,
        match="has not completed",
    ):
        build_steps(records)


def test_failed_method_event_recorded():
    """CrewAI failure events must produce failed steps."""

    records = [
        make_event(1, 0, "start", "uppercase"),
        make_event(2, 10, "failed", "uppercase"),
    ]

    steps = build_steps(records)

    assert len(steps) == 1

    assert steps[0].status == "error"
    assert steps[0].error_type == "RuntimeError"
    assert steps[0].node_name == "uppercase"


def test_unknown_event_phase_rejected():
    """Unknown event types must not appear successful."""

    records = [
        make_event(1, 0, "start", "uppercase"),
        make_event(2, 10, "paused", "uppercase"),
    ]

    with pytest.raises(
        ValueError,
        match="Unknown CrewAI event phase",
    ):
        build_steps(records)
