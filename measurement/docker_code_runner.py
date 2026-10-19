
"""C3-14: Docker-based isolated Python unit-test runner.

Implements the IsolatedCodeRunner interface used by
measurement.live_verifiers.UnitTestVerifier.

The candidate code and unittest definitions run inside
a Docker container, never directly in the host Python
interpreter.

Required test format:
    Python unittest.TestCase classes and test methods.

Security controls:
    - No container network access
    - Read-only root filesystem
    - No host directory mounts
    - No Linux capabilities
    - Non-root container user
    - CPU, memory, process, and time limits
    - Temporary writable directory inside the container
    - No automatic image downloads

Important:
    Docker is not available in the current development
    environment. Offline tests can mock subprocess.run.

    Docker isolation reduces risk but is not a complete
    security guarantee against malicious generated code.
    Production deployment requires additional hardening.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from typing import Callable

from measurement.live_verifiers import CodeTestResult


RESULT_PREFIX = "C314_UNIT_TEST_RESULT:"


CONTAINER_HARNESS = r'''
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

PREFIX = "C314_UNIT_TEST_RESULT:"

payload = json.load(sys.stdin)

with tempfile.TemporaryDirectory(dir="/tmp") as folder:
    root = pathlib.Path(folder)

    (root / "solution.py").write_text(
        payload["code"],
        encoding="utf-8",
    )

    (root / "test_solution.py").write_text(
        payload["tests"],
        encoding="utf-8",
    )

    sys.path.insert(0, str(root))

    # Capture output from candidate code and tests.
    with contextlib.redirect_stdout(io.StringIO()):
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(root),
            pattern="test_solution.py",
            top_level_dir=str(root),
        )

        total = suite.countTestCases()

        if total < 1:
            raise ValueError(
                "No unittest test cases discovered"
            )

        result = unittest.TextTestRunner(
            stream=io.StringIO(),
            verbosity=0,
        ).run(suite)

    passed = (
        total
        - len(result.failures)
        - len(result.errors)
        - len(result.skipped)
    )

    print(
        PREFIX
        + json.dumps(
            {
                "passed": passed,
                "total": total,
            }
        ),
        flush=True,
    )
'''


def _positive_number(
    value: float,
    name: str,
) -> float:
    """Validate a positive finite numeric setting."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{name} must be a positive number"
        )

    number = float(value)

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{name} must be positive and finite"
        )

    return number


def _required_text(
    value: str,
    name: str,
    maximum_length: int,
) -> str:
    """Validate candidate code and test definitions."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a nonempty string"
        )

    if len(value) > maximum_length:
        raise ValueError(
            f"{name} exceeds the size limit"
        )

    return value


class DockerCodeRunner:
    """Run Python unittest cases inside a restricted container.

    A custom command_runner can be injected for offline tests.

    Real execution requires:
        Docker Engine running
        A locally available Python container image

    Docker is never invoked during object construction.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12-alpine",
        docker_executable: str = "docker",
        timeout_seconds: float = 20.0,
        memory_mb: int = 128,
        cpus: float = 1.0,
        pids_limit: int = 64,
        command_runner: Callable = subprocess.run,
    ) -> None:

        self.image = _required_text(
            image,
            "image",
            256,
        )

        self.docker_executable = _required_text(
            docker_executable,
            "docker_executable",
            1024,
        )

        self.timeout_seconds = _positive_number(
            timeout_seconds,
            "timeout_seconds",
        )

        self.cpus = _positive_number(
            cpus,
            "cpus",
        )

        if (
            isinstance(memory_mb, bool)
            or not isinstance(memory_mb, int)
            or memory_mb < 32
        ):
            raise ValueError(
                "memory_mb must be at least 32"
            )

        if (
            isinstance(pids_limit, bool)
            or not isinstance(pids_limit, int)
            or pids_limit < 1
        ):
            raise ValueError(
                "pids_limit must be positive"
            )

        if not callable(command_runner):
            raise TypeError(
                "command_runner must be callable"
            )

        self.memory_mb = memory_mb
        self.pids_limit = pids_limit
        self.command_runner = command_runner

    def _command(
        self,
        container_name: str,
    ) -> list[str]:
        """Construct a restricted Docker execution command."""

        return [
            self.docker_executable,
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            f"{self.memory_mb}m",
            "--memory-swap",
            f"{self.memory_mb}m",
            "--cpus",
            str(self.cpus),
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "python",
            "-I",
            "-c",
            CONTAINER_HARNESS,
        ]

    def _cleanup(
        self,
        container_name: str,
    ) -> None:
        """Best-effort cleanup after a Docker client timeout."""

        try:
            self.command_runner(
                [
                    self.docker_executable,
                    "rm",
                    "--force",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def run(
        self,
        *,
        code: str,
        tests: str,
    ) -> CodeTestResult:
        """Run supplied unittest cases inside a Docker container.

        Test failures produce passed/total counts.
        Docker failures, timeouts, and missing result
        metadata raise explicit exceptions.
        """

        code = _required_text(
            code,
            "code",
            100000,
        )

        tests = _required_text(
            tests,
            "tests",
            100000,
        )

        payload = json.dumps(
            {
                "code": code,
                "tests": tests,
            }
        )

        container_name = (
            "c314-verifier-"
            + uuid.uuid4().hex
        )

        command = self._command(
            container_name
        )

        started = time.perf_counter()

        try:
            completed = self.command_runner(
                command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker is unavailable. Install and start "
                "Docker Engine before live code verification."
            ) from exc

        except subprocess.TimeoutExpired as exc:
            self._cleanup(container_name)

            raise TimeoutError(
                "Isolated code verification timed out"
            ) from exc

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if completed.returncode != 0:
            raise RuntimeError(
                "Docker execution failed or the test "
                "harness could not complete"
            )

        stdout = completed.stdout

        if not isinstance(stdout, str):
            raise RuntimeError(
                "Docker returned invalid output"
            )

        results = [
            line[len(RESULT_PREFIX):]
            for line in stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]

        if len(results) != 1:
            raise RuntimeError(
                "Expected exactly one isolated "
                "unit-test result"
            )

        try:
            result = json.loads(
                results[0]
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid isolated test result JSON"
            ) from exc

        if (
            not isinstance(result, dict)
            or set(result) != {"passed", "total"}
        ):
            raise RuntimeError(
                "Unexpected isolated test result structure"
            )

        passed = result["passed"]
        total = result["total"]

        if (
            isinstance(passed, bool)
            or isinstance(total, bool)
            or not isinstance(passed, int)
            or not isinstance(total, int)
            or total <= 0
            or passed < 0
            or passed > total
        ):
            raise RuntimeError(
                "Invalid isolated test counts"
            )

        return CodeTestResult(
            passed=passed,
            total=total,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
