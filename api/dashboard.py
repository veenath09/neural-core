
"""Neural-Core Execution Dashboard API.

Read-only FastAPI service for Pillar 4 execution traces.

Uses the existing execution serialization module.

Endpoints:
    GET /api/health
    GET /api/executions
    GET /api/executions/{execution_id}
    GET /api/metrics

The API:
- Never executes agents or model calls
- Never creates fake execution records
- Reads only saved, validated execution JSON traces
- Returns empty results when no traces exist

Development server should bind to localhost only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from execution.serialization import (
    load_trace_json,
    trace_to_dict,
)
from execution.trace import ExecutionTrace


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACE_DIRECTORY = (
    PROJECT_ROOT / "runs" / "execution_traces"
)

MAX_TRACE_FILE_BYTES = 2 * 1024 * 1024

app = FastAPI(
    title="Neural-Core Dashboard API",
    description=(
        "Read-only access to validated Pillar 4 "
        "execution traces and metrics."
    ),
    version="0.1.0",
)

# Allow only our local React development server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_saved_traces() -> list[ExecutionTrace]:
    """Read validated execution traces from disk.

    Invalid files are skipped and logged.
    We do not convert unrelated training traces.
    """

    if not TRACE_DIRECTORY.exists():
        return []

    if not TRACE_DIRECTORY.is_dir():
        logger.warning(
            "Trace location is not a directory: %s",
            TRACE_DIRECTORY,
        )
        return []

    traces = []

    for path in sorted(
        TRACE_DIRECTORY.glob("*.json")
    ):
        try:
            # Avoid following symlinked files outside
            # the designated execution trace folder.
            if path.is_symlink():
                logger.warning(
                    "Skipping symlink: %s",
                    path,
                )
                continue

            if path.stat().st_size > MAX_TRACE_FILE_BYTES:
                logger.warning(
                    "Skipping oversized trace: %s",
                    path,
                )
                continue

            trace = load_trace_json(path)

            traces.append(trace)

        except (
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Skipping invalid execution trace %s: %s",
                path,
                exc,
            )

    return traces


def trace_summary(
    trace: ExecutionTrace,
) -> dict:
    """Build a dashboard summary from actual trace data."""

    steps = trace.steps

    model_ids = sorted(
        {
            step.model_id
            for step in steps
        }
    )

    total_cost = sum(
        step.cost_usd
        for step in steps
    )

    total_input_tokens = sum(
        step.input_tokens
        for step in steps
    )

    total_output_tokens = sum(
        step.output_tokens
        for step in steps
    )

    # Sum of individual step durations.
    # This is NOT necessarily end-to-end latency:
    # parallel steps may overlap.
    total_step_wall_ms = sum(
        step.wall_ms
        for step in steps
    )

    return {
        "execution_id": trace.execution_id,
        "framework": trace.framework,
        "status": trace.final_status,
        "strategy_step_id": trace.strategy_step_id,
        "strategy_round": trace.strategy_round,
        "catalogue_version": trace.catalogue_version,
        "adapter_version": trace.adapter_version,
        "step_count": len(steps),
        "model_ids": model_ids,
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_step_wall_ms": total_step_wall_ms,
        "quality": None,
        "measurement_source": "saved_execution_trace",
    }


@app.get("/api/health")
def get_health():
    """Return API status without invoking the execution engine."""

    return {
        "status": "ok",
        "service": "neural-core-dashboard",
        "version": "0.1.0",
        "mode": "read_only",
        "trace_directory_exists": (
            TRACE_DIRECTORY.is_dir()
        ),
    }


@app.get("/api/executions")
def get_executions():
    """List existing validated execution traces."""

    traces = load_saved_traces()

    return {
        "measurement_source": "saved_execution_traces",
        "total": len(traces),
        "executions": [
            trace_summary(trace)
            for trace in traces
        ],
    }


@app.get("/api/executions/{execution_id}")
def get_execution(execution_id: str):
    """Return one execution trace by its validated ID.

    The ID is matched against loaded traces and
    never used to construct a filesystem path.
    """

    for trace in load_saved_traces():
        if trace.execution_id == execution_id:
            return {
                "measurement_source": "saved_execution_trace",
                "summary": trace_summary(trace),
                "trace": trace_to_dict(trace),
            }

    raise HTTPException(
        status_code=404,
        detail="Execution trace not found",
    )


@app.get("/api/metrics")
def get_metrics():
    """Calculate metrics from real saved execution traces."""

    traces = load_saved_traces()

    summaries = [
        trace_summary(trace)
        for trace in traces
    ]

    total = len(summaries)

    completed = sum(
        item["status"] == "ok"
        for item in summaries
    )

    errors = sum(
        item["status"] == "error"
        for item in summaries
    )

    timeouts = sum(
        item["status"] == "timeout"
        for item in summaries
    )

    total_cost = sum(
        item["total_cost_usd"]
        for item in summaries
    )

    return {
        "measurement_source": "saved_execution_traces",
        "total_executions": total,
        "completed_executions": completed,
        "error_executions": errors,
        "timeout_executions": timeouts,
        "success_rate": (
            completed / total
            if total > 0
            else None
        ),
        "total_cost_usd": total_cost,
        "mean_cost_usd": (
            total_cost / total
            if total > 0
            else None
        ),
        "average_latency_ms": None,
        "latency_note": (
            "End-to-end execution latency is not "
            "recorded in the current trace schema."
        ),
        "quality": None,
    }
