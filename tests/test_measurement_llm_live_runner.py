
"""C3-11 integration tests for the guarded live runner.

IMPORTANT:
These tests use a fake OpenAI client.
No real OpenAI requests are made.
No actual API key is required.

Tests cover:
1. Offline execution without an API key.
2. Explicit billing authorization requirements.
3. Missing API-key protection.
4. End-to-end evaluation with simulated provider calls.

The tests use real RouterBench test data and the original
C3-03 capability cards while simulating OpenAI responses.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from data.routerbench import POOL
from scripts import evaluate_llm_live as live


# ============================================================
# FAKE OPENAI RESPONSES API
# ============================================================

class FakeResponses:
    """Simulate the OpenAI Responses API without network calls."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        """Return a simulated provider response."""

        self.calls.append(kwargs)

        return SimpleNamespace(
            model=kwargs["model"],
            status="completed",
            output_text=json.dumps(
                {
                    "model_id": POOL[0],
                    "halt": "stop",
                }
            ),
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=20,
            ),
        )


# ============================================================
# TEST ARGUMENT BUILDER
# ============================================================

def make_args(*extra):
    """Build arguments using the actual evaluation CLI."""

    return live.build_parser().parse_args(
        [
            "--eval-id",
            "c3-11-fake-provider-test",
            "--episodes",
            "1",
            "--seeds",
            "0",
            "--model",
            "test-router-model",
            *extra,
        ]
    )


# ============================================================
# TEST 1: OFFLINE MODE REQUIRES NO API KEY
# ============================================================

def test_offline_mode_requires_no_key(
    monkeypatch,
    tmp_path,
):
    """Offline execution must not require OpenAI access.

    Real research data is loaded before moving to the
    temporary test directory.

    This prevents relative dataset paths from causing
    FileNotFoundError during the test.
    """

    # Load actual research data from the project root.
    inputs = live.load_inputs(1)

    # Supply the already-loaded data when the runner
    # requests it from inside the temporary directory.
    monkeypatch.setattr(
        live,
        "load_inputs",
        lambda episodes: inputs,
    )

    # Ensure no API key is available.
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    # Use a temporary directory to isolate test output.
    monkeypatch.chdir(tmp_path)

    args = make_args()

    # Run offline preflight.
    result = live.run(args)

    # Offline mode must not produce live predictions.
    assert result is None

    # No evaluation result directory should be created.
    assert not (
        tmp_path / "results"
    ).exists()


# ============================================================
# TEST 2: LIVE MODE REQUIRES EXPLICIT APPROVAL
# ============================================================

def test_live_mode_requires_explicit_approval(
    monkeypatch,
):
    """The live flag alone must not authorize billing."""

    # Dummy API key; never used for a real request.
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "fake-key-for-offline-test",
    )

    args = make_args(
        "--enable-live-api",
    )

    # Live requests must be rejected without the
    # explicit billing confirmation.
    with pytest.raises(PermissionError):
        live.validate_configuration(args)


# ============================================================
# TEST 3: LIVE MODE REQUIRES AN API KEY
# ============================================================

def test_live_mode_requires_key(
    monkeypatch,
):
    """Live evaluation must reject missing credentials."""

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    args = make_args(
        "--enable-live-api",
        "--acknowledge-billing",
        "BILLABLE_CALLS_APPROVED",
        "--input-usd-per-million",
        "0.05",
        "--output-usd-per-million",
        "0.40",
        "--price-source",
        "https://openai.com/api/pricing/",
        "--max-total-usd",
        "1",
        "--max-request-usd",
        "0.1",
        "--max-calls",
        "10",
    )

    # Even with billing authorization, the absence
    # of an API key must stop the evaluation.
    with pytest.raises(
        EnvironmentError,
        match="OPENAI_API_KEY",
    ):
        live.validate_configuration(args)


# ============================================================
# TEST 4: END-TO-END FAKE PROVIDER EVALUATION
# ============================================================

def test_fake_provider_evaluation_end_to_end(
    monkeypatch,
    tmp_path,
):
    """Run one complete RouterBench test episode.

    Uses:
    - Real RouterBench held-out test data
    - Original C3-03 model capability cards
    - Existing evaluation and scoring logic
    - Fake OpenAI provider responses
    - Actual spending-guard and accounting logic

    No network connection or real API usage occurs.
    """

    # --------------------------------------------------------
    # STEP 1: Load research data before changing directories.
    # --------------------------------------------------------

    inputs = live.load_inputs(1)

    # Reuse loaded data inside the temporary folder.
    monkeypatch.setattr(
        live,
        "load_inputs",
        lambda episodes: inputs,
    )

    monkeypatch.chdir(tmp_path)

    # --------------------------------------------------------
    # STEP 2: Configure a dummy API key.
    # --------------------------------------------------------

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "fake-key-for-offline-test",
    )

    # --------------------------------------------------------
    # STEP 3: Create a fake OpenAI client.
    # --------------------------------------------------------

    fake_responses = FakeResponses()

    def fake_openai_client(**kwargs):
        """Replace OpenAI SDK construction in this test."""

        # The live runner must disable automatic retries.
        assert kwargs["max_retries"] == 0

        return SimpleNamespace(
            responses=fake_responses,
        )

    # Replace the SDK client constructor so no real
    # OpenAI client or network connection is created.
    import openai

    monkeypatch.setattr(
        openai,
        "OpenAI",
        fake_openai_client,
    )

    # --------------------------------------------------------
    # STEP 4: Configure a simulated live evaluation.
    # --------------------------------------------------------

    args = make_args(
        "--enable-live-api",
        "--acknowledge-billing",
        "BILLABLE_CALLS_APPROVED",
        "--input-usd-per-million",
        "0.05",
        "--output-usd-per-million",
        "0.40",
        "--price-source",
        "https://openai.com/api/pricing/",
        "--max-total-usd",
        "1",
        "--max-request-usd",
        "0.1",
        "--max-calls",
        "10",
    )

    # --------------------------------------------------------
    # STEP 5: Execute the complete evaluation.
    # --------------------------------------------------------

    output_path = live.run(args)

    # The run should produce a predictions file.
    assert output_path is not None
    assert output_path.exists()

    # --------------------------------------------------------
    # STEP 6: Load saved predictions.
    # --------------------------------------------------------

    predictions = pd.read_parquet(
        output_path
    )

    terminal = predictions.loc[
        predictions["terminal"]
    ]

    # One episode was evaluated by two policies:
    # always-cheapest and the fake live LLM router.
    assert len(terminal) == 2

    assert set(
        predictions["policy"]
    ) == {
        "always-cheapest",
        "llm-router-openai",
    }

    # The fake OpenAI API must have received requests.
    assert len(fake_responses.calls) > 0

    # --------------------------------------------------------
    # STEP 7: Verify provider token accounting.
    # --------------------------------------------------------

    live_rows = predictions.loc[
        predictions["policy"].eq(
            "llm-router-openai"
        )
    ]

    # Router usage must be labeled as provider usage.
    assert live_rows[
        "routing_usage_source"
    ].eq("provider").all()

    # Routing cost must be recorded.
    assert live_rows[
        "routing_cost_usd"
    ].gt(0).all()

    # --------------------------------------------------------
    # STEP 8: Verify total episode cost.
    # --------------------------------------------------------

    live_terminal = live_rows.loc[
        live_rows["terminal"]
    ]

    difference = (
        live_terminal["episode_cost_usd"]
        - live_terminal["episode_routing_cost_usd"]
        - live_terminal["episode_execution_cost_usd"]
    )

    # Total cost must equal routing + execution cost.
    assert difference.abs().le(1e-8).all()

    # --------------------------------------------------------
    # STEP 9: Verify run status and spending record.
    # --------------------------------------------------------

    status_path = (
        tmp_path
        / "results"
        / args.eval_id
        / "run_status.json"
    )

    assert status_path.exists()

    with status_path.open(
        encoding="utf-8",
    ) as file:
        status = json.load(file)

    # The evaluation must finish successfully.
    assert status["status"] == "completed"

    # One episode evaluated by two policies.
    assert status[
        "completed_policy_evaluations"
    ] == 2

    # Provider usage must be fully reconciled.
    assert status["usage_uncertain"] is False

    # The reported routing cost must be positive.
    assert status[
        "reported_routing_spend_usd"
    ] > 0
