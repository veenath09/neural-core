
"""Tests for the Neural-Core read-only dashboard API.

No real models, agents, or external APIs are executed.
Tests use temporary directories and synthetic trace objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.dashboard as dashboard


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolate tests from actual saved execution traces."""

    monkeypatch.setattr(
        dashboard,
        "TRACE_DIRECTORY",
        tmp_path / "execution_traces",
    )

    return TestClient(dashboard.app)


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["mode"] == "read_only"
    assert data["trace_directory_exists"] is False


def test_empty_execution_history(client):
    response = client.get("/api/executions")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 0
    assert data["executions"] == []
    assert data["measurement_source"] == "saved_execution_traces"


def test_empty_metrics(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_executions"] == 0
    assert data["completed_executions"] == 0
    assert data["error_executions"] == 0
    assert data["timeout_executions"] == 0
    assert data["total_cost_usd"] == 0

    assert data["success_rate"] is None
    assert data["mean_cost_usd"] is None
    assert data["quality"] is None
    assert data["average_latency_ms"] is None


def test_unknown_execution_returns_404(client):
    response = client.get(
        "/api/executions/nonexistent-execution"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Execution trace not found"


def test_invalid_json_trace_is_skipped(
    client,
    tmp_path,
):
    folder = tmp_path / "execution_traces"
    folder.mkdir()

    (folder / "invalid.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    response = client.get("/api/executions")

    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_oversized_trace_is_skipped(
    client,
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "execution_traces"
    folder.mkdir()

    (folder / "large.json").write_text(
        "x" * 100,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        dashboard,
        "MAX_TRACE_FILE_BYTES",
        20,
    )

    response = client.get("/api/executions")

    assert response.status_code == 200
    assert response.json()["total"] == 0


def make_fake_trace(
    execution_id,
    status,
    cost,
):
    """Minimal synthetic trace-shaped object."""

    step = SimpleNamespace(
        model_id="synthetic-model",
        cost_usd=cost,
        input_tokens=100,
        output_tokens=50,
        wall_ms=200.0,
    )

    return SimpleNamespace(
        execution_id=execution_id,
        framework="synthetic-framework",
        final_status=status,
        strategy_step_id="synthetic-step",
        strategy_round=0,
        catalogue_version="synthetic-catalogue",
        adapter_version="synthetic-adapter",
        steps=(step,),
    )


def test_execution_summary_calculation():
    trace = make_fake_trace(
        "synthetic-001",
        "ok",
        0.002,
    )

    summary = dashboard.trace_summary(trace)

    assert summary["execution_id"] == "synthetic-001"
    assert summary["step_count"] == 1
    assert summary["model_ids"] == ["synthetic-model"]
    assert summary["total_cost_usd"] == pytest.approx(0.002)
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50
    assert summary["total_step_wall_ms"] == 200.0
    assert summary["quality"] is None


def test_metrics_with_synthetic_traces(
    client,
    monkeypatch,
):
    traces = [
        make_fake_trace("synthetic-001", "ok", 0.002),
        make_fake_trace("synthetic-002", "error", 0.003),
        make_fake_trace("synthetic-003", "timeout", 0.001),
    ]

    monkeypatch.setattr(
        dashboard,
        "load_saved_traces",
        lambda: traces,
    )

    response = client.get("/api/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_executions"] == 3
    assert data["completed_executions"] == 1
    assert data["error_executions"] == 1
    assert data["timeout_executions"] == 1

    assert data["success_rate"] == pytest.approx(1 / 3)
    assert data["total_cost_usd"] == pytest.approx(0.006)
    assert data["mean_cost_usd"] == pytest.approx(0.002)


def test_execution_list_with_synthetic_trace(
    client,
    monkeypatch,
):
    trace = make_fake_trace(
        "synthetic-001",
        "ok",
        0.004,
    )

    monkeypatch.setattr(
        dashboard,
        "load_saved_traces",
        lambda: [trace],
    )

    response = client.get("/api/executions")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert data["executions"][0]["execution_id"] == "synthetic-001"
    assert data["executions"][0]["total_cost_usd"] == pytest.approx(
        0.004
    )


def test_api_does_not_accept_execution_post(client):
    response = client.post(
        "/api/executions",
        json={"input_text": "synthetic request"},
    )

    assert response.status_code == 405


def test_localhost_cors_is_allowed(client):
    response = client.get(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
        },
    )

    assert response.status_code == 200

    assert response.headers.get(
        "access-control-allow-origin"
    ) == "http://localhost:5173"


def test_unapproved_origin_is_not_allowed(client):
    response = client.get(
        "/api/health",
        headers={
            "Origin": "https://example.com",
        },
    )

    assert response.status_code == 200

    assert response.headers.get(
        "access-control-allow-origin"
    ) is None
