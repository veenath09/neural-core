"""Framework-independent execution engine for Pillar 4."""

from __future__ import annotations

from controller.contracts import Strategy
from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
    validate_adapter_result,
    validate_execution_request,
)
from execution.catalogue import ResourceCatalogue
from execution.trace import validate_execution_trace


class ExecutionEngine:
    """Connect controller strategies to runtime framework adapters."""

    def __init__(
        self,
        catalogue: ResourceCatalogue,
        adapter: ExecutionAdapter,
    ) -> None:
        self.catalogue = catalogue
        self.adapter = adapter

    def execute(
        self,
        strategy: Strategy,
        input_text: str,
        execution_id: str,
    ) -> AdapterResult:
        """Execute one controller strategy through the configured adapter."""

        if not execution_id:
            raise ValueError("execution_id cannot be empty")

        resource = self.catalogue.resolve(strategy)

        request = ExecutionRequest(
            execution_id=execution_id,
            strategy=strategy,
            resource=resource,
            catalogue_version=self.catalogue.version,
            input_text=input_text,
        )

        validate_execution_request(request)

        result = self.adapter.execute(request)

        validate_execution_trace(result.trace)

        validate_adapter_result(
            self.adapter,
            request,
            result,
        )

        return result