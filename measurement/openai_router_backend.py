
"""C3-11: OpenAI Responses API backend for LLM routing.

Implements the existing LLMRouterBackend protocol.

The OpenAI client must be explicitly supplied. This
module does not create a client, read API credentials,
or make network requests during import or construction.

Use a fake client for offline tests.

IMPORTANT:
    Do not enable live evaluation until provider pricing,
    spending limits and train-derived model capability
    cards have been verified.
"""

from __future__ import annotations

from typing import Any

from measurement.llm_router import RouterResponse


ROUTING_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "routing_choice",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
            },
            "halt": {
                "type": "string",
                "enum": ["stop", "escalate"],
            },
        },
        "required": ["model_id", "halt"],
        "additionalProperties": False,
    },
}


class OpenAIResponsesBackend:
    """Adapter for the OpenAI Responses API.

    The injected client must provide:
        client.responses.create(...)

    This makes the backend independently testable
    without an API key or network connection.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_output_tokens: int = 128,
    ) -> None:

        if client is None:
            raise ValueError(
                "An explicit OpenAI client is required"
            )

        if (
            not isinstance(model, str)
            or not model.strip()
        ):
            raise ValueError(
                "A router model ID is required"
            )

        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens < 1
        ):
            raise ValueError(
                "max_output_tokens must be positive"
            )

        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def complete(
        self,
        messages: list[dict[str, str]],
    ) -> RouterResponse:
        """Request one routing decision and capture usage.

        Returns a RouterResponse compatible with LLMRouter.

        Provider-reported usage is required. Missing
        token accounting causes an explicit failure,
        never a fabricated zero-cost result.
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Routing messages cannot be empty"
            )

        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in
                ("system", "developer", "user")
                or not isinstance(
                    message.get("content"), str
                )
            ):
                raise ValueError(
                    "Invalid routing message"
                )

        # This is the only provider-call boundary.
        # Offline tests inject a fake client.
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            text={
                "format": ROUTING_RESPONSE_FORMAT,
            },
            max_output_tokens=self.max_output_tokens,
            store=False,
        )

        usage = getattr(response, "usage", None)

        if usage is None:
            raise RuntimeError(
                "Provider response has no token usage; "
                "billing cannot be reconciled"
            )

        input_tokens = getattr(
            usage, "input_tokens", None
        )

        output_tokens = getattr(
            usage, "output_tokens", None
        )

        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise RuntimeError(
                    f"Missing or invalid provider {name}"
                )

        actual_model = getattr(
            response, "model", None
        )

        if (
            not isinstance(actual_model, str)
            or not actual_model.strip()
        ):
            raise RuntimeError(
                "Provider response is missing its model ID"
            )

        status = getattr(
            response, "status", None
        )

        content = getattr(
            response, "output_text", None
        )

        # If the provider completed a billable response
        # but did not produce usable JSON, return empty
        # content. LLMRouter will record the charge and
        # mark the choice invalid.
        if status != "completed":
            content = ""

        if not isinstance(content, str):
            content = ""

        return RouterResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            router_model=actual_model,
            usage_source="provider",
        )
