"""ArtifactNet accounting stays scoped to a trial and separate from Codex usage."""

from __future__ import annotations

import json

import pytest

from workflows.metrics import collect_metrics


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "trial" / "workflow"
    (path / "usage").mkdir(parents=True)
    (path / "runtime.json").write_text(json.dumps({"project": "trial-identity"}))
    return path


def request(**changes):
    record = {
        "schema_version": 1,
        "timestamp": "2026-10-05T10:00:00Z",
        "trial_id": "trial-identity",
        "trial_name": "trial",
        "milestone": "m1",
        "role": "llm",
        "model": "fixture-model",
        "status": 200,
        "duration_ms": 100,
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "cost_usd": 0.1,
    }
    return {**record, **changes}


def write_requests(root, records):
    (root / "usage" / "requests.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def test_rollups_include_failed_requests_and_keep_unknown_costs(root):
    write_requests(root, [
        request(),
        request(role="embedding", usage={"prompt_tokens": 4, "total_tokens": 4}, cost_usd=0.02),
        request(milestone="m2", usage={"input_tokens": 2, "output_tokens": 1}, cost_usd=None),
        request(milestone=None, status=502, usage=None, cost_usd=None),
    ])

    result = collect_metrics(root)
    assert result["schema_version"] == 3
    assert result["trial_id"] == "trial-identity"
    assert result["trial_name"] == "trial"
    total = result["artifactnet_usage"]
    assert total["requests"] == 4
    assert total["failed_requests"] == 1
    assert (total["input_tokens"], total["output_tokens"], total["total_tokens"]) == (16, 4, 20)
    assert total["usage_missing_requests"] == 1
    assert total["usage_complete"] is False
    assert total["known_cost_usd"] == pytest.approx(0.12)
    assert total["cost_missing_requests"] == 2
    assert total["cost_usd"] is None
    assert total["cost_complete"] is False
    assert total["by_role"]["llm"]["requests"] == 3
    assert total["by_role"]["embedding"]["cost_usd"] == 0.02
    milestone = result["milestones"]["m1"]["artifactnet_usage"]
    assert milestone["requests"] == 2
    assert milestone["cost_usd"] == pytest.approx(0.12)
    assert milestone["cost_complete"] is True
    assert result["milestones"]["m2"]["artifactnet_usage"]["total_tokens"] == 3
    assert result["artifactnet_unattributed_usage"]["requests"] == 1
    assert result["artifactnet_unattributed_usage"]["failed_requests"] == 1
    assert result["artifactnet_log"]["records_complete"] is True


def test_failed_request_with_reported_usage_and_cost_is_counted(root):
    write_requests(root, [request(status=429, cost_usd=0)])
    total = collect_metrics(root)["artifactnet_usage"]
    assert total["failed_requests"] == 1
    assert total["input_tokens"] == 10
    assert total["cost_usd"] == 0
    assert total["cost_complete"] is True


def test_logs_for_another_trial_are_excluded_even_with_same_trial_name(root):
    write_requests(root, [request(), request(trial_id="other-identity", cost_usd=100)])
    result = collect_metrics(root)
    assert result["artifactnet_usage"]["requests"] == 1
    assert result["artifactnet_usage"]["cost_usd"] == 0.1
    assert result["artifactnet_log"]["mismatched_trial_records"] == 1
    assert result["artifactnet_log"]["records_complete"] is True


@pytest.mark.parametrize("changes", [{"status": 0}, {"status": 600}, {"milestone": "../m1"}, {"milestone": ""}])
def test_invalid_http_status_or_milestone_is_excluded(root, changes):
    write_requests(root, [request(**changes)])
    result = collect_metrics(root)
    assert result["artifactnet_usage"]["requests"] == 0
    assert result["artifactnet_log"]["invalid_records"] == 1
    assert result["artifactnet_usage"]["cost_usd"] is None


def test_legacy_and_malformed_records_are_reported_without_guessing_identity(root):
    legacy = root / "usage" / "artifactnet.jsonl"
    legacy.write_text('legacy row without identity\n{"usage": null}\n')
    write_requests(root, [request(), request(trial_id=None), [], request(role="unknown"), request(status=True)])
    raw = root / "usage" / "requests.jsonl"
    raw.write_text(raw.read_text() + '{"trial_id":')
    before = raw.read_text()

    result = collect_metrics(root)
    assert result["artifactnet_usage"]["requests"] == 1
    assert result["artifactnet_usage"]["known_cost_usd"] == 0.1
    assert result["artifactnet_usage"]["cost_usd"] is None
    assert result["artifactnet_usage"]["usage_complete"] is False
    assert result["artifactnet_log"]["invalid_records"] == 4
    assert result["artifactnet_log"]["legacy_records"] == 3
    assert result["artifactnet_log"]["records_complete"] is False
    assert raw.read_text() == before
    assert legacy.read_text() == 'legacy row without identity\n{"usage": null}\n'


@pytest.mark.parametrize("runtime_contents", [None, "invalid json", "{}", "[]"])
def test_missing_or_invalid_runtime_identity_never_assigns_records(root, runtime_contents):
    if runtime_contents is None:
        (root / "runtime.json").unlink()
    else:
        (root / "runtime.json").write_text(runtime_contents)
    write_requests(root, [request()])
    result = collect_metrics(root)
    assert result["trial_id"] is None
    assert result["artifactnet_usage"]["requests"] == 0
    assert result["artifactnet_usage"]["cost_usd"] is None
    assert result["artifactnet_log"]["identity_available"] is False
    assert result["artifactnet_log"]["unidentified_trial_records"] == 1


def test_absent_log_is_distinct_from_recorded_zero_requests(root):
    missing = collect_metrics(root)
    assert missing["artifactnet_log"]["present"] is False
    assert missing["artifactnet_usage"]["cost_usd"] is None
    assert missing["artifactnet_usage"]["usage_complete"] is False
    write_requests(root, [])
    empty = collect_metrics(root)
    assert empty["artifactnet_log"]["present"] is True
    assert empty["artifactnet_usage"]["requests"] == 0
    assert empty["artifactnet_usage"]["cost_usd"] == 0
    assert empty["artifactnet_usage"]["usage_complete"] is True


@pytest.mark.parametrize("cost", [None, True, -1, float("nan"), float("inf"), "0.5"])
def test_invalid_costs_do_not_become_zero_or_poison_known_subtotal(root, cost):
    write_requests(root, [request(), request(cost_usd=cost)])
    total = collect_metrics(root)["artifactnet_usage"]
    assert total["known_cost_usd"] == 0.1
    assert total["cost_missing_requests"] == 1
    assert total["cost_usd"] is None
    assert total["cost_complete"] is False


@pytest.mark.parametrize("usage", [None, {}, [], {"prompt_tokens": True}, {"prompt_tokens": -1}])
def test_missing_or_invalid_usage_is_flagged(root, usage):
    write_requests(root, [request(usage=usage)])
    total = collect_metrics(root)["artifactnet_usage"]
    assert total["usage_missing_requests"] == 1
    assert total["usage_complete"] is False
    assert total["input_tokens"] == 0  # Reported subtotal, explicitly incomplete.
    assert total["cost_usd"] == 0.1  # Cost and token completeness are independent.


def test_embedding_zero_output_is_known_but_missing_llm_output_is_not(root):
    write_requests(root, [
        request(role="embedding", usage={"prompt_tokens": 4}),
        request(role="llm", usage={"prompt_tokens": 4}),
    ])
    roles = collect_metrics(root)["artifactnet_usage"]["by_role"]
    assert roles["embedding"]["total_tokens"] == 4
    assert roles["embedding"]["output_tokens"] == 0
    assert roles["embedding"]["usage_complete"] is True
    assert roles["llm"]["usage_complete"] is False


def test_collection_is_idempotent_and_keeps_codex_usage_separate(root, monkeypatch):
    stage = {"status": "complete", "attempts": 1, "elapsed_seconds": 2, "artifactnet_usage": {"obsolete": True}}
    state = {"milestones": {"m1": {"stages": {"openspec-propose": stage}}}}
    (root / "state.json").write_text(json.dumps(state))
    agent_stats = root.parent / "agent_stats.json"
    agent_stats.write_text('{"summary":{"total_cost_usd":7}}')
    monkeypatch.setattr("workflows.metrics.collect_stage_usage", lambda *args: {"total_cost_usd": 7})
    write_requests(root, [request()])

    first = collect_metrics(root)
    assert collect_metrics(root) == first
    assert first["agent_usage_source"] == "../agent_stats.json"
    assert first["artifactnet_usage"]["cost_usd"] == 0.1
    recorded_stage = first["milestones"]["m1"]["stages"]["openspec-propose"]
    assert recorded_stage["agent_usage"] == {"total_cost_usd": 7}
    assert "artifactnet_usage" not in recorded_stage
    assert json.loads((root / "state.json").read_text()) == state
    assert json.loads((root / "metrics.json").read_text()) == first
    assert agent_stats.read_text() == '{"summary":{"total_cost_usd":7}}'
