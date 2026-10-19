
"""C3-11: Pre-call spending safeguards for live LLM routing.

This module wraps the existing LLMRouter without modifying
its original decision or execution contracts.

Safety controls:
- Provider calls are disabled by default.
- Check estimated routing cost before each call.
- Respect the remaining simulated episode budget.
- Respect a per-request cost limit.
- Respect a cumulative routing budget.
- Respect the maximum number of API calls.
- Stop further calls when provider usage is uncertain.
- Reconcile actual provider usage after each response.
- Charge invalid responses when usage was reported.

IMPORTANT:
These are conservative, estimate-based safeguards.
They cannot guarantee an absolute provider billing cap.

No client, credentials, or network connection are
created by this module.
"""

from __future__ import annotations

import json
import math

from typing import Any

from measurement.llm_router import (
    LLMRouter,
    RouterCallRecord,
    RouterPricing,
)


class RoutingBudgetError(RuntimeError):
    """A routing request was blocked by spending controls."""


class RoutingSpendGuard:
    """Track and restrict estimated provider routing expenses.

    A guard instance must be reused for every routing
    decision belonging to the same evaluation run.

    The guard is synchronous and must not be shared
    across concurrently executing API requests.
    """

    def __init__(
        self,
        *,
        max_total_usd: float,
        max_request_usd: float,
        max_calls: int,
        enabled: bool = False,
        allow_synthetic_pricing_for_tests: bool = False,
        max_input_bytes: int = 20000,
        input_token_multiplier: int = 2,
        message_overhead_tokens: int = 512,
    ) -> None:

        for name, value in (
            ("max_total_usd", max_total_usd),
            ("max_request_usd", max_request_usd),
        ):
            if isinstance(value, bool):
                raise ValueError(
                    f"{name} must be positive and finite"
                )

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be positive and finite"
                ) from exc

            if not math.isfinite(number) or number <= 0:
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        for name, value, minimum in (
            ("max_calls", max_calls, 1),
            ("max_input_bytes", max_input_bytes, 1),
            ("input_token_multiplier", input_token_multiplier, 1),
            ("message_overhead_tokens", message_overhead_tokens, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}"
                )

        if not isinstance(enabled, bool):
            raise ValueError(
                "enabled must be a boolean"
            )

        if not isinstance(
            allow_synthetic_pricing_for_tests,
            bool,
        ):
            raise ValueError(
                "allow_synthetic_pricing_for_tests "
                "must be a boolean"
            )

        self.max_total_usd = float(
            max_total_usd
        )

        self.max_request_usd = float(
            max_request_usd
        )

        self.max_calls = max_calls

        self.enabled = enabled

        self.allow_synthetic_pricing_for_tests = (
            allow_synthetic_pricing_for_tests
        )

        self.max_input_bytes = max_input_bytes

        self.input_token_multiplier = (
            input_token_multiplier
        )

        self.message_overhead_tokens = (
            message_overhead_tokens
        )

        self.actual_spent_usd = 0.0

        self.calls_started = 0

        self.usage_uncertain = False

        self.blocked = False

        self._pending = False

        self._estimated_pending_usd = 0.0

        self.estimate_exceeded = False

    @property
    def remaining_run_budget_usd(self) -> float:
        """Remaining cumulative routing allowance."""

        return max(
            0.0,
            self.max_total_usd
            - self.actual_spent_usd,
        )

    def preflight(
        self,
        *,
        messages: list[dict[str, str]],
        budget_left_usd: float,
        max_output_tokens: int,
        pricing: RouterPricing,
    ) -> float:
        """Validate a proposed call before network access.

        Returns its conservatively estimated cost.

        The token estimate is intentionally padded and
        must not be interpreted as exact OpenAI usage.
        """

        if not self.enabled:
            raise RoutingBudgetError(
                "Live routing is disabled. "
                "Explicit authorization is required."
            )

        if self.blocked or self.usage_uncertain:
            raise RoutingBudgetError(
                "Routing is blocked because the "
                "spending state is unsafe or uncertain"
            )

        if self._pending:
            raise RoutingBudgetError(
                "A previous routing request "
                "has not been reconciled"
            )

        if self.calls_started >= self.max_calls:
            raise RoutingBudgetError(
                "Maximum routing call count reached"
            )

        if (
            isinstance(budget_left_usd, bool)
            or not math.isfinite(float(budget_left_usd))
            or budget_left_usd < 0
        ):
            raise RoutingBudgetError(
                "Invalid remaining episode budget"
            )

        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens < 1
        ):
            raise RoutingBudgetError(
                "Invalid maximum output token count"
            )

        source = pricing.price_source.lower()

        if (
            (
                "synthetic" in source
                or "mock" in source
                or "test" in source
            )
            and not self.allow_synthetic_pricing_for_tests
        ):
            raise RoutingBudgetError(
                "Synthetic pricing cannot authorize "
                "live provider spending"
            )

        # The serialized input size is measurable
        # without contacting a provider.
        encoded = json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        if len(encoded) > self.max_input_bytes:
            raise RoutingBudgetError(
                "Routing input exceeds the "
                "configured maximum byte size"
            )

        # Conservative estimate, NOT a guaranteed
        # provider-side token upper bound.
        estimated_input_tokens = (
            len(encoded)
            * self.input_token_multiplier
            + self.message_overhead_tokens
        )

        estimated_cost = pricing.calculate(
            estimated_input_tokens,
            max_output_tokens,
        )

        if estimated_cost > self.max_request_usd:
            raise RoutingBudgetError(
                "Estimated request cost exceeds "
                "the per-request spending limit"
            )

        if estimated_cost > self.remaining_run_budget_usd:
            raise RoutingBudgetError(
                "Estimated request cost exceeds "
                "the remaining evaluation budget"
            )

        if estimated_cost > budget_left_usd:
            raise RoutingBudgetError(
                "Estimated request cost exceeds "
                "the remaining episode budget"
            )

        # Reserve one attempt. No network call occurs
        # within preflight().
        self.calls_started += 1

        self._pending = True

        self._estimated_pending_usd = (
            estimated_cost
        )

        return estimated_cost

    def reconcile(
        self,
        record: RouterCallRecord,
    ) -> None:
        """Account for an API response with recorded usage.

        This runs even when an LLM response contains
        invalid JSON or an invalid routing selection.
        """

        if not self._pending:
            raise RoutingBudgetError(
                "No routing request is awaiting reconciliation"
            )

        charge = float(
            record.routing_cost_usd
        )

        if not math.isfinite(charge) or charge < 0:
            self.mark_usage_uncertain()

            raise RoutingBudgetError(
                "Invalid provider routing charge"
            )

        if (
            record.usage_source != "provider"
            and not self.allow_synthetic_pricing_for_tests
        ):
            self.mark_usage_uncertain()

            raise RoutingBudgetError(
                "Provider routing record must "
                "contain provider-reported usage"
            )

        estimate = self._estimated_pending_usd

        self._pending = False
        self._estimated_pending_usd = 0.0

        # Always preserve the actual recorded charge,
        # even when it exceeds the estimate.
        self.actual_spent_usd += charge

        if charge > estimate:
            self.estimate_exceeded = True

            # A previous estimate proved insufficient.
            # Do not authorize another live call.
            self.blocked = True

        if charge > self.max_request_usd:
            self.blocked = True

        if self.actual_spent_usd > self.max_total_usd:
            self.blocked = True

    def mark_usage_uncertain(self) -> None:
        """Freeze further requests after an unknown charge.

        This can happen when a provider request fails,
        times out, or omits reliable token usage.

        The current monetary total is then only the
        amount successfully accounted for, not a
        guarantee of the full provider bill.
        """

        self._pending = False

        self._estimated_pending_usd = 0.0

        self.usage_uncertain = True

        self.blocked = True


class BudgetGuardedLLMRouter(LLMRouter):
    """LLMRouter with mandatory spending preflight.

    Inherits the existing LLMRouter contract.

    This subclass remains compatible with
    measurement.evaluate.evaluate_episode(),
    which already handles LLMRouter instances.
    """

    def __init__(
        self,
        *,
        spending_guard: RoutingSpendGuard,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            **kwargs
        )

        if not isinstance(
            spending_guard,
            RoutingSpendGuard,
        ):
            raise TypeError(
                "spending_guard must be "
                "a RoutingSpendGuard"
            )

        self.spending_guard = (
            spending_guard
        )

    def decide(self, task):
        """Check estimated spending before routing.

        Actual provider usage is reconciled even
        when the resulting routing choice is invalid.
        """

        backend_max_output = getattr(
            self.backend,
            "max_output_tokens",
            None,
        )

        # A live-capable backend must expose the
        # maximum output tokens it requests.
        if backend_max_output is None:
            raise RoutingBudgetError(
                "Guarded routing requires a backend "
                "with max_output_tokens"
            )

        messages = self.build_messages(
            task
        )

        # This must happen BEFORE backend.complete().
        self.spending_guard.preflight(
            messages=messages,
            budget_left_usd=(
                task.budget_left_usd
            ),
            max_output_tokens=backend_max_output,
            pricing=self.pricing,
        )

        records_before = len(
            self.call_records
        )

        try:
            strategy = super().decide(
                task
            )

        except Exception:

            records_after = len(
                self.call_records
            )

            if records_after == records_before + 1:
                # The backend returned usage, but the
                # choice may have been invalid.
                self.spending_guard.reconcile(
                    self.call_records[-1]
                )

            else:
                # A request may have been billed even
                # though usage was not returned.
                self.spending_guard.mark_usage_uncertain()

            raise

        records_after = len(
            self.call_records
        )

        if records_after != records_before + 1:
            self.spending_guard.mark_usage_uncertain()

            raise RoutingBudgetError(
                "Provider call did not produce "
                "exactly one accounting record"
            )

        self.spending_guard.reconcile(
            self.call_records[-1]
        )

        return strategy
