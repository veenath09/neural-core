
"""C3-14: Live answer verifiers using the existing C3-07 Score.

Implements:
    ExactMatchVerifier  - normalized exact match for QA
    UnitTestVerifier    - code results from an isolated runner
    JudgeModelVerifier  - open-text evaluation from a judge backend

All verifiers return measurement.scorer.Score.

Security:
    This module NEVER executes generated code directly.
    Code execution must be supplied by an external,
    appropriately isolated and resource-limited runner.

The judge backend is injected, allowing deterministic
offline tests without real API calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from measurement.scorer import CompositeScorer, Score


def _nonnegative(
    value: float,
    name: str,
) -> float:
    """Require a finite, nonnegative numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{name} must be a finite number"
        )

    number = float(value)

    if (
        not math.isfinite(number)
        or number < 0
    ):
        raise ValueError(
            f"{name} must be finite and nonnegative"
        )

    return number


def _quality(
    value: float,
    name: str = "quality",
) -> float:
    """Require a measured quality between zero and one."""

    number = _nonnegative(
        value,
        name,
    )

    if number > 1:
        raise ValueError(
            f"{name} must be between zero and one"
        )

    return number


def _optional_latency(
    value: float | None,
) -> float | None:
    if value is None:
        return None

    return _nonnegative(
        value,
        "latency_ms",
    )


def _required_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{name} must not be empty"
        )

    return value


def normalize_qa_answer(
    answer: str,
) -> str:
    """Normalize case and whitespace, not meaning.

    Deliberately preserves punctuation and numbers.
    This is normalized exact matching, not semantic
    matching or answer extraction.
    """

    if not isinstance(answer, str):
        raise ValueError(
            "answer must be a string"
        )

    return re.sub(
        r"\s+",
        " ",
        answer.strip(),
    ).casefold()


@dataclass(frozen=True)
class CodeTestResult:
    """Measured result returned by an isolated code runner."""

    passed: int
    total: int
    cost_usd: float = 0.0
    latency_ms: float | None = None


@dataclass(frozen=True)
class JudgeResult:
    """Measured result returned by an open-text judge backend."""

    quality: float
    cost_usd: float
    latency_ms: float | None = None


class IsolatedCodeRunner(Protocol):
    """Contract for an external, isolated code execution service.

    Production implementations must enforce isolation,
    timeouts, and resource limits.
    """

    def run(
        self,
        *,
        code: str,
        tests: str,
    ) -> CodeTestResult:
        ...


class OpenTextJudge(Protocol):
    """Contract for a judge-model adapter."""

    def judge(
        self,
        *,
        question: str,
        answer: str,
        rubric: str,
        reference: str | None,
    ) -> JudgeResult:
        ...


class ExactMatchVerifier:
    """Verify short QA answers with normalized exact matching."""

    def __init__(
        self,
        scorer: CompositeScorer,
    ) -> None:
        if not isinstance(
            scorer,
            CompositeScorer,
        ):
            raise TypeError(
                "scorer must be a CompositeScorer"
            )

        self.scorer = scorer

    def verify(
        self,
        *,
        answer: str,
        reference: str,
        cost_usd: float = 0.0,
        latency_ms: float | None = None,
        lam: float = 0.0,
        mu: float = 0.0,
    ) -> Score:
        """Return Score with quality 1 for match, else 0."""

        _required_text(
            reference,
            "reference",
        )

        if not isinstance(answer, str):
            raise ValueError(
                "answer must be a string"
            )

        matched = (
            normalize_qa_answer(answer)
            == normalize_qa_answer(reference)
        )

        return self.scorer.score(
            quality=float(matched),
            cost_usd=_nonnegative(
                cost_usd,
                "cost_usd",
            ),
            latency_ms=_optional_latency(
                latency_ms
            ),
            lam=lam,
            mu=mu,
        )


class UnitTestVerifier:
    """Score code using outcomes from an isolated test runner.

    This verifier never executes code itself.

    The injected runner must run tests outside the host
    research process with appropriate sandbox controls.
    """

    def __init__(
        self,
        scorer: CompositeScorer,
        runner: IsolatedCodeRunner,
    ) -> None:
        if not isinstance(
            scorer,
            CompositeScorer,
        ):
            raise TypeError(
                "scorer must be a CompositeScorer"
            )

        if not callable(
            getattr(runner, "run", None)
        ):
            raise TypeError(
                "runner must provide run()"
            )

        self.scorer = scorer
        self.runner = runner

    def verify(
        self,
        *,
        code: str,
        tests: str,
        generation_cost_usd: float = 0.0,
        generation_latency_ms: float | None = None,
        lam: float = 0.0,
        mu: float = 0.0,
    ) -> Score:
        """Score the fraction of predefined tests passed."""

        _required_text(
            code,
            "code",
        )

        _required_text(
            tests,
            "tests",
        )

        generation_cost = _nonnegative(
            generation_cost_usd,
            "generation_cost_usd",
        )

        generation_latency = _optional_latency(
            generation_latency_ms
        )

        result = self.runner.run(
            code=code,
            tests=tests,
        )

        if not isinstance(
            result,
            CodeTestResult,
        ):
            raise TypeError(
                "runner must return CodeTestResult"
            )

        if (
            isinstance(result.passed, bool)
            or isinstance(result.total, bool)
            or not isinstance(result.passed, int)
            or not isinstance(result.total, int)
            or result.total <= 0
            or result.passed < 0
            or result.passed > result.total
        ):
            raise ValueError(
                "Invalid code test counts"
            )

        runner_cost = _nonnegative(
            result.cost_usd,
            "runner_cost_usd",
        )

        runner_latency = _optional_latency(
            result.latency_ms
        )

        # Total latency is available only when both
        # generation and test-runner latencies were measured.
        total_latency = None

        if (
            generation_latency is not None
            and runner_latency is not None
        ):
            total_latency = (
                generation_latency
                + runner_latency
            )

        return self.scorer.score(
            quality=result.passed / result.total,
            cost_usd=(
                generation_cost
                + runner_cost
            ),
            latency_ms=total_latency,
            lam=lam,
            mu=mu,
        )


class JudgeModelVerifier:
    """Score open-text responses using an injected judge model.

    The judge must provide measured quality and cost.
    No scores or usage measurements are fabricated here.
    """

    def __init__(
        self,
        scorer: CompositeScorer,
        judge: OpenTextJudge,
    ) -> None:
        if not isinstance(
            scorer,
            CompositeScorer,
        ):
            raise TypeError(
                "scorer must be a CompositeScorer"
            )

        if not callable(
            getattr(judge, "judge", None)
        ):
            raise TypeError(
                "judge must provide judge()"
            )

        self.scorer = scorer
        self.judge = judge

    def verify(
        self,
        *,
        question: str,
        answer: str,
        rubric: str,
        reference: str | None = None,
        generation_cost_usd: float = 0.0,
        generation_latency_ms: float | None = None,
        lam: float = 0.0,
        mu: float = 0.0,
    ) -> Score:
        """Evaluate an open-text answer with a judge backend."""

        _required_text(
            question,
            "question",
        )

        _required_text(
            answer,
            "answer",
        )

        _required_text(
            rubric,
            "rubric",
        )

        if reference is not None:
            _required_text(
                reference,
                "reference",
            )

        generation_cost = _nonnegative(
            generation_cost_usd,
            "generation_cost_usd",
        )

        generation_latency = _optional_latency(
            generation_latency_ms
        )

        result = self.judge.judge(
            question=question,
            answer=answer,
            rubric=rubric,
            reference=reference,
        )

        if not isinstance(
            result,
            JudgeResult,
        ):
            raise TypeError(
                "judge must return JudgeResult"
            )

        measured_quality = _quality(
            result.quality,
            "judge_quality",
        )

        judge_cost = _nonnegative(
            result.cost_usd,
            "judge_cost_usd",
        )

        judge_latency = _optional_latency(
            result.latency_ms
        )

        # Do not turn unavailable timing into an
        # invented zero-millisecond measurement.
        total_latency = None

        if (
            generation_latency is not None
            and judge_latency is not None
        ):
            total_latency = (
                generation_latency
                + judge_latency
            )

        return self.scorer.score(
            quality=measured_quality,
            cost_usd=(
                generation_cost
                + judge_cost
            ),
            latency_ms=total_latency,
            lam=lam,
            mu=mu,
        )
