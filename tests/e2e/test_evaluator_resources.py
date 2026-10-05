"""Evaluation CPU quotas must fit the Docker daemon that runs the tests."""

import subprocess

import pytest

from harness.e2e.evaluator import PatchEvaluator


@pytest.mark.parametrize("requested,available,expected", [(16, 8, 8), (4, 8, 4), (16, 32, 16)])
def test_cpu_limit_respects_daemon_and_requested_quota(monkeypatch, requested, available, expected):
    evaluator = object.__new__(PatchEvaluator)
    evaluator.docker_cpus = requested
    evaluator._eval_meta = {}
    monkeypatch.setattr(
        "harness.e2e.evaluator.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, f"{available}\n", ""),
    )

    evaluator._configure_cpu_limit()

    assert evaluator.docker_cpus == expected
    assert evaluator._eval_meta["docker_cpus"] == expected
    assert evaluator._eval_meta["docker_cpus_requested"] == requested


@pytest.mark.parametrize("returncode,stdout,stderr", [(0, "0", ""), (0, "invalid", ""), (1, "", "daemon unavailable")])
def test_missing_cpu_capacity_reports_infrastructure_error(monkeypatch, returncode, stdout, stderr):
    evaluator = object.__new__(PatchEvaluator)
    evaluator.docker_cpus = 16
    evaluator._eval_meta = {}
    monkeypatch.setattr(
        "harness.e2e.evaluator.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, returncode, stdout, stderr),
    )

    with pytest.raises(RuntimeError, match="Cannot determine Docker CPU capacity"):
        evaluator._configure_cpu_limit()
