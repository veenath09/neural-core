
"""C3-14: Offline tests for the Docker code runner.

Tests:
- Docker command security options
- No execution during construction
- Structured test result parsing
- Execution time measurements
- Missing Docker handling
- Timeout handling and cleanup
- Invalid result rejection
- Input validation
- Integration with C3-07 Score

All tests use a fake Docker command runner.
No containers are started.
No generated code is executed.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from measurement.docker_code_runner import (
    CONTAINER_HARNESS,
    RESULT_PREFIX,
    DockerCodeRunner,
)
from measurement.live_verifiers import (
    CodeTestResult,
    UnitTestVerifier,
)
from measurement.scorer import (
    CompositeScorer,
    Score,
)


CODE = "def add(a, b): return a + b"

TESTS = """
import unittest
from solution import add

class TestAdd(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add(2, 3), 5)
"""


class FakeDocker:
    """Simulate Docker without executing code."""

    def __init__(
        self,
        *,
        passed=2,
        total=3,
        returncode=0,
        stdout=None,
        error=None,
    ):
        self.calls = []
        self.error = error

        if stdout is None:
            stdout = (
                RESULT_PREFIX
                + json.dumps(
                    {
                        "passed": passed,
                        "total": total,
                    }
                )
            )

        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, command, **kwargs):
        self.calls.append(
            {
                "command": command,
                "kwargs": kwargs,
            }
        )

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr="",
        )


def make_runner(fake=None, **kwargs):
    if fake is None:
        fake = FakeDocker()

    return DockerCodeRunner(
        command_runner=fake,
        **kwargs,
    )


def test_runner_construction_does_not_start_docker():
    fake = FakeDocker()

    runner = make_runner(fake)

    assert isinstance(runner, DockerCodeRunner)
    assert fake.calls == []


def test_default_docker_image():
    runner = make_runner()

    assert runner.image == "python:3.12-alpine"


def test_restricted_docker_command():
    runner = make_runner()

    command = runner._command("test-container")

    assert command[0:2] == [
        "docker",
        "run",
    ]

    assert "--rm" in command
    assert "--network" in command
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "--security-opt" in command
    assert "--pids-limit" in command
    assert "--memory" in command
    assert "--cpus" in command
    assert "--user" in command
    assert "--tmpfs" in command

    assert command[
        command.index("--network") + 1
    ] == "none"

    assert command[
        command.index("--cap-drop") + 1
    ] == "ALL"

    assert command[
        command.index("--pull=never")
    ] == "--pull=never"

    assert "no-new-privileges:true" in command


def test_no_host_mounts_in_command():
    runner = make_runner()

    command = runner._command("test-container")

    assert "-v" not in command
    assert "--volume" not in command
    assert "--mount" not in command


def test_container_runs_as_non_root():
    runner = make_runner()

    command = runner._command("test-container")

    index = command.index("--user")

    assert command[index + 1] == "65534:65534"


def test_container_has_no_network():
    runner = make_runner()

    command = runner._command("test-container")

    index = command.index("--network")

    assert command[index + 1] == "none"


def test_container_harness_uses_unittest():
    assert "unittest" in CONTAINER_HARNESS
    assert "discover" in CONTAINER_HARNESS
    assert "test_solution.py" in CONTAINER_HARNESS


def test_successful_docker_result():
    fake = FakeDocker(
        passed=2,
        total=3,
    )

    runner = make_runner(fake)

    result = runner.run(
        code=CODE,
        tests=TESTS,
    )

    assert isinstance(result, CodeTestResult)
    assert result.passed == 2
    assert result.total == 3
    assert result.cost_usd == 0.0

    assert result.latency_ms is not None
    assert result.latency_ms >= 0


def test_all_tests_pass():
    runner = make_runner(
        FakeDocker(
            passed=5,
            total=5,
        )
    )

    result = runner.run(
        code=CODE,
        tests=TESTS,
    )

    assert result.passed == 5
    assert result.total == 5


def test_all_tests_fail():
    runner = make_runner(
        FakeDocker(
            passed=0,
            total=4,
        )
    )

    result = runner.run(
        code=CODE,
        tests=TESTS,
    )

    assert result.passed == 0
    assert result.total == 4


def test_docker_receives_input_payload():
    fake = FakeDocker()

    runner = make_runner(fake)

    runner.run(
        code=CODE,
        tests=TESTS,
    )

    assert len(fake.calls) == 1

    call = fake.calls[0]

    payload = json.loads(
        call["kwargs"]["input"]
    )

    assert payload["code"] == CODE
    assert payload["tests"] == TESTS


def test_docker_receives_timeout():
    fake = FakeDocker()

    runner = make_runner(
        fake,
        timeout_seconds=7.0,
    )

    runner.run(
        code=CODE,
        tests=TESTS,
    )

    assert fake.calls[0]["kwargs"]["timeout"] == 7.0


def test_docker_command_does_not_use_shell():
    fake = FakeDocker()

    runner = make_runner(fake)

    runner.run(
        code=CODE,
        tests=TESTS,
    )

    kwargs = fake.calls[0]["kwargs"]

    assert kwargs.get("shell", False) is False


def test_missing_docker_is_reported():
    fake = FakeDocker(
        error=FileNotFoundError(
            "docker not found"
        )
    )

    runner = make_runner(fake)

    with pytest.raises(
        RuntimeError,
        match="Docker is unavailable",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


def test_timeout_is_reported():
    class TimeoutDocker:
        def __init__(self):
            self.calls = []

        def __call__(self, command, **kwargs):
            self.calls.append(command)

            if command[1] == "run":
                raise subprocess.TimeoutExpired(
                    command,
                    1,
                )

            return SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            )

    fake = TimeoutDocker()

    runner = make_runner(
        fake,
        timeout_seconds=1.0,
    )

    with pytest.raises(
        TimeoutError,
        match="timed out",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )

    assert len(fake.calls) == 2

    # A timeout must trigger a force-removal attempt.
    assert fake.calls[1][1:3] == [
        "rm",
        "--force",
    ]


def test_nonzero_exit_is_rejected():
    runner = make_runner(
        FakeDocker(returncode=1)
    )

    with pytest.raises(
        RuntimeError,
        match="Docker execution failed",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


def test_missing_result_is_rejected():
    runner = make_runner(
        FakeDocker(
            stdout="No result was produced"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="exactly one",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


def test_multiple_results_are_rejected():
    line = (
        RESULT_PREFIX
        + '{"passed": 1, "total": 1}'
    )

    runner = make_runner(
        FakeDocker(
            stdout=line + "\n" + line
        )
    )

    with pytest.raises(
        RuntimeError,
        match="exactly one",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


def test_invalid_result_json_is_rejected():
    runner = make_runner(
        FakeDocker(
            stdout=RESULT_PREFIX + "not-json"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid isolated test result JSON",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


@pytest.mark.parametrize(
    "result",
    [
        {"passed": -1, "total": 3},
        {"passed": 4, "total": 3},
        {"passed": 0, "total": 0},
        {"passed": True, "total": 3},
        {"passed": 1, "total": True},
        {"passed": 1.5, "total": 3},
        {"passed": 1, "total": "3"},
    ],
)
def test_invalid_test_counts_rejected(result):
    fake = FakeDocker(
        stdout=(
            RESULT_PREFIX
            + json.dumps(result)
        )
    )

    runner = make_runner(fake)

    with pytest.raises(
        RuntimeError,
        match="Invalid isolated test counts",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


def test_unexpected_result_fields_rejected():
    fake = FakeDocker(
        stdout=(
            RESULT_PREFIX
            + json.dumps(
                {
                    "passed": 1,
                    "total": 1,
                    "extra": "unexpected",
                }
            )
        )
    )

    runner = make_runner(fake)

    with pytest.raises(
        RuntimeError,
        match="Unexpected isolated test result structure",
    ):
        runner.run(
            code=CODE,
            tests=TESTS,
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_code_rejected_before_docker(value):
    fake = FakeDocker()

    runner = make_runner(fake)

    with pytest.raises(ValueError):
        runner.run(
            code=value,
            tests=TESTS,
        )

    assert fake.calls == []


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_tests_rejected_before_docker(value):
    fake = FakeDocker()

    runner = make_runner(fake)

    with pytest.raises(ValueError):
        runner.run(
            code=CODE,
            tests=value,
        )

    assert fake.calls == []


def test_oversized_code_is_rejected():
    fake = FakeDocker()

    runner = make_runner(fake)

    with pytest.raises(ValueError):
        runner.run(
            code="x" * 100001,
            tests=TESTS,
        )

    assert fake.calls == []


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        make_runner(
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize(
    "memory",
    [
        0,
        16,
        -1,
        True,
        32.5,
    ],
)
def test_invalid_memory_rejected(memory):
    with pytest.raises(ValueError):
        make_runner(
            memory_mb=memory,
        )


@pytest.mark.parametrize(
    "pids",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_pids_limit_rejected(pids):
    with pytest.raises(ValueError):
        make_runner(
            pids_limit=pids,
        )


def test_invalid_command_runner_rejected():
    with pytest.raises(TypeError):
        DockerCodeRunner(
            command_runner=None,
        )


def test_integration_with_existing_score():
    fake = FakeDocker(
        passed=3,
        total=4,
    )

    runner = make_runner(fake)

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    verifier = UnitTestVerifier(
        scorer,
        runner,
    )

    score = verifier.verify(
        code=CODE,
        tests=TESTS,
        generation_cost_usd=0.001,
        lam=0.25,
    )

    assert isinstance(score, Score)

    assert score.quality == pytest.approx(
        0.75
    )

    assert score.cost_usd == pytest.approx(
        0.001
    )

    expected_reward = (
        0.75
        - 0.25
        * (
            0.001
            / scorer.cfg.cost_ref_usd
        )
    )

    assert score.reward == pytest.approx(
        expected_reward
    )


def test_fake_runner_never_executes_code():
    fake = FakeDocker()

    runner = make_runner(fake)

    # This is supplied as data only.
    # The fake runner never invokes Docker or Python.
    candidate = (
        "raise RuntimeError('must not execute locally')"
    )

    result = runner.run(
        code=candidate,
        tests=TESTS,
    )

    assert result.total == 3
    assert len(fake.calls) == 1
