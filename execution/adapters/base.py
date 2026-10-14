"""Framework-independent execution adapter contract for Pillar 4.

Concrete adapters such as LangGraphAdapter or CrewAIAdapter must implement
this interface so the execution engine does not depend on one framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from controller.contracts import Strategy
from execution.catalogue import ResourceEntry
from execution.trace import ExecutionTrace


@dataclass(frozen=True)
class ExecutionRequest:
    """Input passed from the execution engine to a framework adapter."""

    execution_id: str
    strategy: Strategy
    resource: ResourceEntry
    catalogue_version: str
    input_text: str


@dataclass(frozen=True)
class AdapterResult:
    """Result returned by a framework adapter."""

    output_text: str
    trace: ExecutionTrace


class ExecutionAdapter(ABC):
    """Base contract implemented by every supported execution framework."""

    framework: str
    adapter_version: str

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> AdapterResult:
        """Execute one strategy using the selected framework resource."""
        raise NotImplementedError


def validate_execution_request(request: ExecutionRequest) -> None:
    """Validate a request before it is handed to an adapter."""

    errors: list[str] = []

    if not request.execution_id:
        errors.append("execution_id cannot be empty")

    if not request.strategy.step_id:
        errors.append("strategy step_id cannot be empty")

    if request.strategy.model_id != request.resource.card.model_id:
        errors.append(
            "strategy model_id does not match resolved resource model_id"
        )

    if not request.resource.available:
        errors.append("resolved resource is unavailable")

    if not request.catalogue_version:
        errors.append("catalogue_version cannot be empty")

    if errors:
        raise ValueError("; ".join(errors))


def validate_adapter_result(
    adapter: ExecutionAdapter,
    request: ExecutionRequest,
    result: AdapterResult,
) -> None:
    """Check that adapter output belongs to the request that was executed."""

    errors: list[str] = []

    trace = result.trace

    if trace.execution_id != request.execution_id:
        errors.append(
            "trace execution_id does not match request execution_id"
        )

    if trace.strategy_step_id != request.strategy.step_id:
        errors.append(
            "trace strategy_step_id does not match request strategy"
        )

    if trace.strategy_round != request.strategy.round:
        errors.append(
            "trace strategy_round does not match request strategy"
        )

    if trace.framework != adapter.framework:
        errors.append(
            "trace framework does not match adapter framework"
        )

    if trace.adapter_version != adapter.adapter_version:
        errors.append(
            "trace adapter_version does not match adapter version"
        )

    if trace.catalogue_version != request.catalogue_version:
        errors.append(
            "trace catalogue_version does not match request catalogue"
        )

    if errors:
        raise ValueError("; ".join(errors))