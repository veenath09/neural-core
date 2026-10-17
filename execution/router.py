
"""Framework selection for Pillar 4 execution.

Selects a registered execution adapter without changing the
controller Strategy contract or the existing ExecutionEngine.
"""

from __future__ import annotations

from collections.abc import Mapping

from controller.contracts import Strategy
from execution.adapters.base import AdapterResult, ExecutionAdapter
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine


class ExecutionRouter:
    """Dispatch controller strategies to a selected framework."""

    def __init__(
        self,
        catalogue: ResourceCatalogue,
        adapters: Mapping[str, ExecutionAdapter],
    ) -> None:
        if not adapters:
            raise ValueError(
                "At least one execution adapter is required"
            )

        for name, adapter in adapters.items():
            if not name or name != adapter.framework:
                raise ValueError(
                    "Adapter registry key must match "
                    "the adapter framework"
                )

        self.catalogue = catalogue
        self.adapters = dict(adapters)

    @property
    def available_frameworks(self) -> tuple[str, ...]:
        """Return the names of registered frameworks."""
        return tuple(sorted(self.adapters))

    def execute(
        self,
        strategy: Strategy,
        input_text: str,
        execution_id: str,
        *,
        framework: str,
    ) -> AdapterResult:
        """Execute a strategy through the requested framework."""

        if framework not in self.adapters:
            raise ValueError(
                f"Unsupported execution framework: {framework!r}"
            )

        adapter = self.adapters[framework]

        engine = ExecutionEngine(
            catalogue=self.catalogue,
            adapter=adapter,
        )

        return engine.execute(
            strategy=strategy,
            input_text=input_text,
            execution_id=execution_id,
        )
