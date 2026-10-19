
"""C3-10: Tests for the context-free prompt router.

Tests:
- Training from binary prompt outcomes
- Predicted success probabilities
- Train-only text feature fitting
- Budget-aware and cost-aware model selection
- Independence from episode execution history
- Deterministic fitting
- Integration with the existing evaluation harness
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from measurement.context_free_router import ContextFreeRouter
from measurement.evaluate import evaluate_episode
from measurement.scorer import CompositeScorer

from tests.test_measurement_baselines import make_environment


MODEL_IDS = ("m0", "m1")


@pytest.fixture(scope="module")
def training_data():
    rows = []
    text_lookup = {}

    for i in range(12):
        prompt_id = f"train-{i}"

        if i % 2 == 0:
            text = (
                "Astronomy science telescope planet orbit "
                f"observation example number {i}"
            )
        else:
            text = (
                "Cooking recipe kitchen dinner ingredients "
                f"preparation example number {i}"
            )

        text_lookup[prompt_id] = text

        rows.append(
            {
                "prompt_id": prompt_id,
                "m0_correct": int(i % 2 == 0),
                "m0_cost": 0.10 + i * 0.001,
                "m1_correct": int(i % 3 != 0),
                "m1_cost": 0.40 + i * 0.002,
            }
        )

    # These prompts are available for inference only.
    # Their correctness labels are NOT in training_data.
    text_lookup.update(
        {
            "heldout": (
                "heldoutspecialtokenz astronomy "
                "planet telescope observation"
            ),
            "p0": "Astronomy science planet orbit",
            "p1": "Cooking recipe kitchen dinner",
            "p2": "Astronomy telescope planet",
        }
    )

    return pd.DataFrame(rows), text_lookup


@pytest.fixture(scope="module")
def router(training_data):
    frame, text_lookup = training_data

    return ContextFreeRouter.fit(
        train_frame=frame,
        prompt_text_by_id=text_lookup,
        model_ids=MODEL_IDS,
        cost_ref_usd=0.4,
        n_components=4,
        max_features=100,
        random_state=1234,
    )


def make_task(
    prompt_id="heldout",
    budget=10.0,
    lam=0.25,
    trace=None,
):
    return SimpleNamespace(
        episode_id="example-episode",
        step_id="example-episode:step:0",
        round=0,
        step_emb_key=prompt_id,
        task_emb_key="example-episode",
        step_type=0,
        trace=[] if trace is None else trace,
        budget_left_usd=budget,
        lam=lam,
    )


def test_training_creates_both_predictors(router):
    assert router.model_ids == MODEL_IDS
    assert set(router.predictors) == set(MODEL_IDS)


def test_model_costs_are_learned_from_training_data(
    router, training_data
):
    frame, _ = training_data

    for model_id in MODEL_IDS:
        expected = frame[f"{model_id}_cost"].mean()

        assert router.model_costs[model_id] == pytest.approx(
            expected
        )


def test_probability_predictions_are_valid(router):
    probabilities = router.predict_probabilities(
        "heldout"
    )

    assert set(probabilities) == set(MODEL_IDS)

    for probability in probabilities.values():
        assert 0.0 <= probability <= 1.0


def test_unseen_prompt_can_be_predicted(router):
    # This prompt was not included in the training rows.
    probabilities = router.predict_probabilities(
        "heldout"
    )

    assert len(probabilities) == 2


def test_heldout_text_not_used_to_fit_vocabulary(router):
    # The unusual token appears only in held-out text.
    # It must not enter the training vocabulary.
    assert (
        "heldoutspecialtokenz"
        not in router.vectorizer.vocabulary_
    )


def test_decide_returns_valid_strategy(router):
    task = make_task()

    strategy = router.decide(task)

    assert strategy.model_id in MODEL_IDS
    assert strategy.step_id == task.step_id
    assert strategy.round == task.round
    assert strategy.halt == "stop"


def test_confidence_is_raw_prediction(router):
    task = make_task()

    strategy = router.decide(task)

    probabilities = router.predict_probabilities(
        task.step_emb_key
    )

    assert strategy.confidence == pytest.approx(
        probabilities[strategy.model_id]
    )


def test_trace_history_does_not_change_decision(router):
    first = make_task(
        trace=[],
    )

    second = make_task(
        trace=[
            "previous execution",
            "failed model",
            "unrelated history",
        ],
    )

    first_strategy = router.decide(first)
    second_strategy = router.decide(second)

    assert (
        first_strategy.model_id
        == second_strategy.model_id
    )

    assert (
        first_strategy.confidence
        == pytest.approx(second_strategy.confidence)
    )


def test_episode_identifier_does_not_change_choice(router):
    first = make_task()
    second = make_task()

    second.episode_id = "different-episode"
    second.task_emb_key = "different-episode"
    second.step_id = "different-episode:step:4"

    assert (
        router.decide(first).model_id
        == router.decide(second).model_id
    )


def test_high_cost_penalty_selects_cheapest(router):
    task = make_task(
        budget=10.0,
        lam=100.0,
    )

    strategy = router.decide(task)

    cheapest = min(
        MODEL_IDS,
        key=lambda model_id: router.model_costs[model_id],
    )

    assert strategy.model_id == cheapest


def test_zero_budget_returns_cheapest_attempt(router):
    task = make_task(
        budget=0.0,
    )

    strategy = router.decide(task)

    cheapest = min(
        MODEL_IDS,
        key=lambda model_id: router.model_costs[model_id],
    )

    # The simulator is responsible for rejecting
    # this unaffordable attempt.
    assert strategy.model_id == cheapest


def test_missing_prompt_is_rejected(router):
    with pytest.raises(KeyError, match="Unknown prompt"):
        router.predict_probabilities(
            "does-not-exist"
        )


def test_duplicate_training_prompts_are_rejected(
    training_data,
):
    frame, text_lookup = training_data

    duplicate = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate training prompt",
    ):
        ContextFreeRouter.fit(
            duplicate,
            text_lookup,
            MODEL_IDS,
            cost_ref_usd=0.4,
            n_components=4,
        )


def test_nonbinary_training_outcomes_are_rejected(
    training_data,
):
    frame, text_lookup = training_data

    invalid = frame.copy()
    invalid.loc[0, "m0_correct"] = 2

    with pytest.raises(
        ValueError,
        match="Invalid binary outcomes",
    ):
        ContextFreeRouter.fit(
            invalid,
            text_lookup,
            MODEL_IDS,
            cost_ref_usd=0.4,
            n_components=4,
        )


def test_negative_training_cost_is_rejected(
    training_data,
):
    frame, text_lookup = training_data

    invalid = frame.copy()
    invalid.loc[0, "m0_cost"] = -1.0

    with pytest.raises(
        ValueError,
        match="Invalid training costs",
    ):
        ContextFreeRouter.fit(
            invalid,
            text_lookup,
            MODEL_IDS,
            cost_ref_usd=0.4,
            n_components=4,
        )


def test_missing_training_text_is_rejected(
    training_data,
):
    frame, text_lookup = training_data

    incomplete_lookup = dict(text_lookup)
    del incomplete_lookup["train-0"]

    with pytest.raises(
        ValueError,
        match="Missing prompt text",
    ):
        ContextFreeRouter.fit(
            frame,
            incomplete_lookup,
            MODEL_IDS,
            cost_ref_usd=0.4,
            n_components=4,
        )


def test_single_class_outcomes_are_supported(
    training_data,
):
    frame, text_lookup = training_data

    constant = frame.copy()
    constant["m0_correct"] = 1

    trained = ContextFreeRouter.fit(
        constant,
        text_lookup,
        MODEL_IDS,
        cost_ref_usd=0.4,
        n_components=4,
    )

    probabilities = trained.predict_probabilities(
        "heldout"
    )

    assert probabilities["m0"] == 1.0


def test_training_is_reproducible(
    router,
    training_data,
):
    frame, text_lookup = training_data

    second = ContextFreeRouter.fit(
        frame,
        text_lookup,
        MODEL_IDS,
        cost_ref_usd=0.4,
        n_components=4,
        max_features=100,
        random_state=1234,
    )

    original = router.predict_probabilities(
        "heldout"
    )

    repeated = second.predict_probabilities(
        "heldout"
    )

    for model_id in MODEL_IDS:
        assert original[model_id] == pytest.approx(
            repeated[model_id],
            abs=1e-8,
        )


def test_integration_with_existing_evaluator(router):
    episode, _ = make_environment()

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    predictions = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        router,
        policy_name="context-free-router",
        scorer=scorer,
        flip_p=0.0,
    )

    assert not predictions.empty

    assert set(predictions["model_id"]).issubset(
        set(MODEL_IDS)
    )

    assert int(predictions["terminal"].sum()) == 1

    assert predictions.iloc[-1][
        "episode_quality"
    ] >= 0.0
