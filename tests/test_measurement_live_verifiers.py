
"""C3-14: Offline tests for live verifier interfaces.

Tests:
- Exact-match QA verification
- Normalization behavior
- Code unit-test verification using a fake runner
- Open-text judge verification using a fake judge
- Compatibility with existing C3-07 Score
- Execution and verification cost accounting
- Measured and missing latency handling
- Input and backend validation

No generated code is executed.
No external API calls are made.
"""

from __future__ import annotations

import math

import pytest

from measurement.live_verifiers import (
    CodeTestResult,
    ExactMatchVerifier,
    JudgeModelVerifier,
    JudgeResult,
    UnitTestVerifier,
    normalize_qa_answer,
)
from measurement.scorer import (
    CompositeScorer,
    Score,
)


@pytest.fixture
def scorer():
    """Reuse the existing project scoring configuration."""

    return CompositeScorer.from_yaml(
        "conf/score.yaml"
    )


class FakeCodeRunner:
    """Deterministic code runner with no code execution."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, *, code, tests):
        self.calls.append(
            {
                "code": code,
                "tests": tests,
            }
        )

        return self.result


class FakeJudge:
    """Deterministic stand-in for an external judge model."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def judge(
        self,
        *,
        question,
        answer,
        rubric,
        reference,
    ):
        self.calls.append(
            {
                "question": question,
                "answer": answer,
                "rubric": rubric,
                "reference": reference,
            }
        )

        return self.result


# -----------------------------------------------------
# Exact-match QA verifier
# -----------------------------------------------------


def test_exact_match_correct_answer(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Colombo",
        reference="Colombo",
    )

    assert isinstance(result, Score)
    assert result.quality == 1.0
    assert result.reward == 1.0


def test_exact_match_incorrect_answer(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Kandy",
        reference="Colombo",
    )

    assert isinstance(result, Score)
    assert result.quality == 0.0
    assert result.reward == 0.0


def test_exact_match_ignores_case(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="COLOMBO",
        reference="colombo",
    )

    assert result.quality == 1.0


def test_exact_match_normalizes_whitespace(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="  Sri   Lanka  ",
        reference="Sri Lanka",
    )

    assert result.quality == 1.0


def test_exact_match_preserves_punctuation(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Colombo!",
        reference="Colombo",
    )

    assert result.quality == 0.0


def test_exact_match_preserves_numeric_difference(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="42",
        reference="43",
    )

    assert result.quality == 0.0


def test_normalization_returns_expected_text():

    assert normalize_qa_answer(
        "  HELLO \n World "
    ) == "hello world"


def test_exact_match_returns_score_with_cost(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Correct",
        reference="correct",
        cost_usd=0.001,
        lam=0.25,
    )

    expected_reward = (
        1.0
        - 0.25 * (
            0.001 / scorer.cfg.cost_ref_usd
        )
    )

    assert result.quality == 1.0
    assert result.cost_usd == 0.001
    assert result.reward == pytest.approx(
        expected_reward
    )


def test_exact_match_preserves_measured_latency(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Yes",
        reference="yes",
        latency_ms=120.0,
    )

    assert result.latency_ms == 120.0
    assert result.latency_included is False


def test_exact_match_missing_latency_is_none(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    result = verifier.verify(
        answer="Yes",
        reference="yes",
    )

    assert result.latency_ms is None


@pytest.mark.parametrize(
    "answer,reference",
    [
        ("yes", ""),
        ("yes", "   "),
        (None, "yes"),
        (123, "123"),
    ],
)
def test_exact_match_rejects_invalid_text(
    scorer,
    answer,
    reference,
):

    verifier = ExactMatchVerifier(
        scorer
    )

    with pytest.raises(ValueError):
        verifier.verify(
            answer=answer,
            reference=reference,
        )


@pytest.mark.parametrize(
    "cost",
    [
        -0.01,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_exact_match_rejects_invalid_cost(
    scorer,
    cost,
):

    verifier = ExactMatchVerifier(
        scorer
    )

    with pytest.raises(ValueError):
        verifier.verify(
            answer="yes",
            reference="yes",
            cost_usd=cost,
        )


# -----------------------------------------------------
# Unit-test verifier
# -----------------------------------------------------


def test_unit_test_verifier_all_pass(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=5,
            total=5,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="def add(a, b): return a + b",
        tests="assert add(2, 3) == 5",
    )

    assert isinstance(result, Score)
    assert result.quality == 1.0
    assert result.reward == 1.0
    assert len(runner.calls) == 1


def test_unit_test_verifier_partial_pass(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=3,
            total=5,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="def answer(): return 42",
        tests="predefined unit tests",
    )

    assert result.quality == pytest.approx(
        0.6
    )


def test_unit_test_verifier_all_fail(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=0,
            total=4,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="def answer(): return None",
        tests="predefined unit tests",
    )

    assert result.quality == 0.0


def test_unit_test_costs_are_combined(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=2,
            total=2,
            cost_usd=0.0002,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="print('hello')",
        tests="predefined unit tests",
        generation_cost_usd=0.001,
    )

    assert result.cost_usd == pytest.approx(
        0.0012
    )


def test_unit_test_latencies_are_combined(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=2,
            total=2,
            latency_ms=60.0,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="print('hello')",
        tests="predefined unit tests",
        generation_latency_ms=150.0,
    )

    assert result.latency_ms == 210.0


def test_unit_test_missing_latency_remains_none(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=1,
            total=1,
            latency_ms=None,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    result = verifier.verify(
        code="print(1)",
        tests="assert True",
        generation_latency_ms=100.0,
    )

    assert result.latency_ms is None


def test_unit_test_runner_receives_inputs(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=1,
            total=1,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    verifier.verify(
        code="def f(): return 1",
        tests="assert f() == 1",
    )

    assert runner.calls == [
        {
            "code": "def f(): return 1",
            "tests": "assert f() == 1",
        }
    ]


@pytest.mark.parametrize(
    "passed,total",
    [
        (-1, 5),
        (6, 5),
        (0, 0),
        (True, 5),
        (1, True),
    ],
)
def test_unit_test_rejects_invalid_counts(
    scorer,
    passed,
    total,
):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=passed,
            total=total,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    with pytest.raises(ValueError):
        verifier.verify(
            code="print(1)",
            tests="assert True",
        )


def test_unit_test_rejects_wrong_backend_result(scorer):

    runner = FakeCodeRunner(
        {
            "passed": 2,
            "total": 2,
        }
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    with pytest.raises(TypeError):
        verifier.verify(
            code="print(1)",
            tests="assert True",
        )


def test_unit_test_rejects_empty_code(scorer):

    runner = FakeCodeRunner(
        CodeTestResult(
            passed=1,
            total=1,
        )
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    with pytest.raises(ValueError):
        verifier.verify(
            code="",
            tests="assert True",
        )

    assert not runner.calls


def test_unit_test_rejects_invalid_runner(scorer):

    with pytest.raises(TypeError):
        UnitTestVerifier(
            scorer,
            object(),
        )


# -----------------------------------------------------
# Judge-model verifier
# -----------------------------------------------------


def test_judge_model_verifier_returns_score(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=0.8,
            cost_usd=0.0005,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    result = verifier.verify(
        question="Explain cloud computing",
        answer="Computing resources delivered over a network.",
        rubric="Evaluate factual correctness",
        reference=None,
    )

    assert isinstance(result, Score)
    assert result.quality == pytest.approx(
        0.8
    )


def test_judge_model_receives_rubric(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=1.0,
            cost_usd=0.001,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    verifier.verify(
        question="What is a VPC?",
        answer="An isolated virtual network.",
        rubric="Assess technical accuracy.",
        reference="A logically isolated virtual network.",
    )

    assert judge.calls == [
        {
            "question": "What is a VPC?",
            "answer": "An isolated virtual network.",
            "rubric": "Assess technical accuracy.",
            "reference": "A logically isolated virtual network.",
        }
    ]


def test_judge_model_costs_are_combined(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=0.9,
            cost_usd=0.0004,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    result = verifier.verify(
        question="Explain an API",
        answer="An interface for software communication.",
        rubric="Assess correctness",
        generation_cost_usd=0.001,
    )

    assert result.cost_usd == pytest.approx(
        0.0014
    )


def test_judge_model_latencies_are_combined(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=0.75,
            cost_usd=0.0,
            latency_ms=200.0,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    result = verifier.verify(
        question="Explain a database",
        answer="A system for storing and managing data.",
        rubric="Assess correctness",
        generation_latency_ms=100.0,
    )

    assert result.latency_ms == 300.0


def test_judge_model_missing_latency_is_none(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=0.5,
            cost_usd=0.0,
            latency_ms=None,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    result = verifier.verify(
        question="Explain networking",
        answer="Connecting devices.",
        rubric="Assess correctness",
        generation_latency_ms=120.0,
    )

    assert result.latency_ms is None


@pytest.mark.parametrize(
    "quality",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_judge_model_rejects_invalid_quality(
    scorer,
    quality,
):

    judge = FakeJudge(
        JudgeResult(
            quality=quality,
            cost_usd=0.0,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    with pytest.raises(ValueError):
        verifier.verify(
            question="Question",
            answer="Answer",
            rubric="Rubric",
        )


def test_judge_model_rejects_wrong_result(scorer):

    judge = FakeJudge(
        {
            "quality": 0.8,
            "cost_usd": 0.0,
        }
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    with pytest.raises(TypeError):
        verifier.verify(
            question="Question",
            answer="Answer",
            rubric="Rubric",
        )


def test_judge_model_rejects_empty_rubric(scorer):

    judge = FakeJudge(
        JudgeResult(
            quality=1.0,
            cost_usd=0.0,
        )
    )

    verifier = JudgeModelVerifier(
        scorer,
        judge,
    )

    with pytest.raises(ValueError):
        verifier.verify(
            question="Question",
            answer="Answer",
            rubric="",
        )

    assert not judge.calls


def test_judge_model_rejects_invalid_backend(scorer):

    with pytest.raises(TypeError):
        JudgeModelVerifier(
            scorer,
            object(),
        )


# -----------------------------------------------------
# Existing C3-07 scoring contract
# -----------------------------------------------------


def test_all_verifiers_return_same_score_type(scorer):

    qa = ExactMatchVerifier(
        scorer
    ).verify(
        answer="A",
        reference="A",
    )

    code = UnitTestVerifier(
        scorer,
        FakeCodeRunner(
            CodeTestResult(
                passed=1,
                total=1,
            )
        ),
    ).verify(
        code="print(1)",
        tests="assert True",
    )

    open_text = JudgeModelVerifier(
        scorer,
        FakeJudge(
            JudgeResult(
                quality=1.0,
                cost_usd=0.0,
            )
        ),
    ).verify(
        question="Question",
        answer="Answer",
        rubric="Rubric",
    )

    assert isinstance(qa, Score)
    assert isinstance(code, Score)
    assert isinstance(open_text, Score)

    assert qa.quality == 1.0
    assert code.quality == 1.0
    assert open_text.quality == 1.0


def test_verifiers_do_not_fabricate_latency(scorer):

    qa = ExactMatchVerifier(
        scorer
    ).verify(
        answer="A",
        reference="A",
    )

    code = UnitTestVerifier(
        scorer,
        FakeCodeRunner(
            CodeTestResult(
                passed=1,
                total=1,
            )
        ),
    ).verify(
        code="print(1)",
        tests="assert True",
    )

    open_text = JudgeModelVerifier(
        scorer,
        FakeJudge(
            JudgeResult(
                quality=1.0,
                cost_usd=0.0,
            )
        ),
    ).verify(
        question="Question",
        answer="Answer",
        rubric="Rubric",
    )

    assert qa.latency_ms is None
    assert code.latency_ms is None
    assert open_text.latency_ms is None


def test_latency_weight_requires_latency_reference(scorer):

    verifier = ExactMatchVerifier(
        scorer
    )

    with pytest.raises(
        ValueError,
        match="latency reference",
    ):
        verifier.verify(
            answer="Yes",
            reference="yes",
            latency_ms=100.0,
            mu=0.5,
        )


def test_exact_match_normalization_is_not_semantic():

    assert normalize_qa_answer(
        "four"
    ) != normalize_qa_answer(
        "4"
    )
