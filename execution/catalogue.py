"""Runtime resource catalogue for Pillar 4.

The catalogue wraps the shared controller ModelCard contract with
execution-time information such as availability and expected latency.
It does not redefine model metadata owned by controller.contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from controller.contracts import ModelCard, Strategy


CATALOGUE_VERSION = "v1"


@dataclass(frozen=True)
class ResourceEntry:
    """Execution-time information for one model resource."""

    card: ModelCard
    available: bool = True
    expected_latency_ms: float | None = None
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class ResourceCatalogue:
    """Versioned collection of resources available to the executor."""

    resources: tuple[ResourceEntry, ...]
    version: str = CATALOGUE_VERSION

    @classmethod
    def from_model_cards(
        cls,
        cards: list[ModelCard],
        version: str = CATALOGUE_VERSION,
    ) -> "ResourceCatalogue":
        """Build a catalogue directly from the controller model pool."""

        return cls(
            resources=tuple(ResourceEntry(card=card) for card in cards),
            version=version,
        )

    def get(self, model_id: str) -> ResourceEntry:
        """Return one catalogue entry by model id."""

        for resource in self.resources:
            if resource.card.model_id == model_id:
                return resource

        raise KeyError(f"model_id {model_id!r} not found in resource catalogue")

    def resolve(self, strategy: Strategy) -> ResourceEntry:
        """Resolve the model requested by a controller strategy."""

        try:
            resource = self.get(strategy.model_id)
        except KeyError as exc:
            raise ValueError(
                f"strategy requested unknown model {strategy.model_id!r}"
            ) from exc

        if not resource.available:
            raise ValueError(
                f"strategy requested unavailable model {strategy.model_id!r}"
            )

        return resource

    def available_model_cards(self) -> list[ModelCard]:
        """Return shared ModelCards currently available for execution."""

        return [
            resource.card
            for resource in self.resources
            if resource.available
        ]

    def available_model_ids(self) -> list[str]:
        """Return ids of models currently available for execution."""

        return [
            resource.card.model_id
            for resource in self.resources
            if resource.available
        ]


def validate_resource_entry(resource: ResourceEntry) -> None:
    """Validate execution-specific metadata for one resource."""

    errors: list[str] = []

    if not resource.card.model_id:
        errors.append("model_id cannot be empty")

    if (
        resource.expected_latency_ms is not None
        and resource.expected_latency_ms < 0
    ):
        errors.append("expected_latency_ms cannot be negative")

    if (
        resource.max_context_tokens is not None
        and resource.max_context_tokens <= 0
    ):
        errors.append("max_context_tokens must be positive")

    if errors:
        raise ValueError("; ".join(errors))


def validate_resource_catalogue(catalogue: ResourceCatalogue) -> None:
    """Validate a complete resource catalogue."""

    errors: list[str] = []

    if not catalogue.version:
        errors.append("catalogue version cannot be empty")

    model_ids = [
        resource.card.model_id
        for resource in catalogue.resources
    ]

    if len(model_ids) != len(set(model_ids)):
        errors.append("resource catalogue contains duplicate model_id values")

    for resource in catalogue.resources:
        try:
            validate_resource_entry(resource)
        except ValueError as exc:
            errors.append(
                f"invalid resource {resource.card.model_id!r}: {exc}"
            )

    if errors:
        raise ValueError("; ".join(errors))