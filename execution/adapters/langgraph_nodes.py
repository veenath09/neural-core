
"""Node-level LangGraph tracing for Pillar 4.

Supports sequential and parallel DAG workflows with explicitly
declared node dependencies.

Repeated execution of the same node (loops) is not supported yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from uuid import uuid4

from execution.adapters.base import AdapterResult, ExecutionRequest
from execution.adapters.langgraph import LangGraphAdapter
from execution.trace import ExecutionStep, ExecutionTrace


ParentSpec = str | tuple[str, ...] | None


class LangGraphNodeAdapter(LangGraphAdapter):
    """LangGraph adapter with node-level and multi-parent tracing."""

    def __init__(
        self,
        graph,
        *,
        node_parents: Mapping[str, ParentSpec],
        adapter_version: str = "v1",
    ) -> None:
        super().__init__(
            graph=graph,
            adapter_version=adapter_version,
        )
        self.node_parents = dict(node_parents)

    def _resolve_parents(
        self,
        node_name: str,
        completed: dict[str, int],
    ) -> tuple[int | None, tuple[int, ...] | None]:
        """Convert parent node names into execution-step indexes."""

        if node_name not in self.node_parents:
            raise ValueError(
                f"No parent mapping for {node_name!r}"
            )

        specification = self.node_parents[node_name]

        if specification is None:
            parent_names = ()
        elif isinstance(specification, str):
            parent_names = (specification,)
        else:
            parent_names = tuple(specification)

        if len(parent_names) != len(set(parent_names)):
            raise ValueError(
                f"Duplicate parent names for {node_name!r}"
            )

        indexes = []

        for parent_name in parent_names:
            if parent_name not in completed:
                raise ValueError(
                    f"Parent {parent_name!r} has not completed"
                )

            indexes.append(completed[parent_name])

        # Completion order determines trace indexes.
        indexes = tuple(sorted(indexes))

        if not indexes:
            return None, ()

        if len(indexes) == 1:
            # Preserve the old single-parent representation.
            return indexes[0], None

        # Multiple dependencies use the new Task 8 field.
        return None, indexes

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:

        starts: dict[str, tuple[str, float]] = {}
        completed: dict[str, int] = {}

        steps: list[ExecutionStep] = []

        final_state = None
        output_text = ""
        final_status = "ok"

        graph_started = perf_counter()

        try:
            stream = self.graph.stream(
                self.input_builder(request),
                config={
                    "configurable": {
                        "thread_id": str(uuid4())
                    }
                },
                stream_mode=["tasks", "values"],
                version="v2",
            )

            for event in stream:

                event_type = event["type"]
                data = event["data"]

                if event_type == "values":
                    final_state = data
                    continue

                if event_type != "tasks":
                    continue

                task_id = data["id"]

                # Start of a node execution
                if "input" in data:
                    starts[task_id] = (
                        data["name"],
                        perf_counter(),
                    )
                    continue

                # Completion of a node execution
                if task_id not in starts:
                    raise ValueError(
                        "Task finished without a matching start"
                    )

                node_name, started = starts.pop(task_id)

                if node_name in completed:
                    raise ValueError(
                        f"Repeated node {node_name!r} "
                        "is not yet supported"
                    )

                parent_step, parent_steps = (
                    self._resolve_parents(
                        node_name,
                        completed,
                    )
                )

                elapsed_ms = (
                    perf_counter() - started
                ) * 1000.0

                error = data.get("error")
                failed = error is not None

                step = ExecutionStep(
                    step_index=len(steps),
                    parent_step=parent_step,
                    parent_steps=parent_steps,
                    model_id=request.resource.card.model_id,
                    tool=None,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    wall_ms=elapsed_ms,
                    status="error" if failed else "ok",
                    error_type=(
                        type(error).__name__
                        if failed else None
                    ),
                    node_name=node_name,
                )

                steps.append(step)
                completed[node_name] = step.step_index

                if failed:
                    raise RuntimeError(
                        f"LangGraph node {node_name!r} failed"
                    )

            if starts:
                raise ValueError(
                    "Some LangGraph tasks did not finish"
                )

            if not steps:
                raise ValueError(
                    "LangGraph emitted no completed node tasks"
                )

            output_text = self.output_extractor(
                final_state
            )

        except Exception as exc:

            final_status = "error"

            # Preserve completed steps and record the failure.
            if not steps or steps[-1].status != "error":

                node_name = None
                parent_index = None

                elapsed_ms = (
                    perf_counter() - graph_started
                ) * 1000.0

                if len(starts) == 1:
                    node_name, started = next(
                        iter(starts.values())
                    )

                    elapsed_ms = (
                        perf_counter() - started
                    ) * 1000.0

                    parent_spec = self.node_parents.get(
                        node_name
                    )

                    if isinstance(parent_spec, str):
                        parent_index = completed.get(
                            parent_spec
                        )

                steps.append(
                    ExecutionStep(
                        step_index=len(steps),
                        parent_step=parent_index,
                        model_id=request.resource.card.model_id,
                        tool=None,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        wall_ms=elapsed_ms,
                        status="error",
                        error_type=type(exc).__name__,
                        node_name=node_name,
                    )
                )

            output_text = ""

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=tuple(steps),
            final_status=final_status,
        )

        return AdapterResult(
            output_text=output_text,
            trace=trace,
        )
