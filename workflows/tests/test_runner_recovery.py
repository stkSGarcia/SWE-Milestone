"""Recovery continues unfinished agent calls and respects invocation limits."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from harness.e2e.agent_runner import AgentRunner
from harness.e2e.config import E2EConfig
from harness.e2e.run_e2e import E2ETrialRunner
from workflows.metrics import collect_stage_usage
from workflows.providers.artifactnet import ArtifactNetProvider
from workflows.providers.openspec import OpenSpecProvider
from workflows.runner import WorkflowRunner
from workflows.state import WorkflowState

SYNC = "openspec-sync-specs"
ARCHIVE = "openspec-archive-change"


@pytest.fixture
def agent(tmp_path):
    root = tmp_path / "workflow"
    workflow = SimpleNamespace(
        root=root,
        state=WorkflowState(root),
        config={
            "stages": [{"name": SYNC}, {"name": ARCHIVE}],
            "budgets": {"stage_seconds": 7200, "milestone_seconds": 28800, "max_attempts": 2},
        },
        runtime=SimpleNamespace(api_key_env_args=list),
        orchestrator=SimpleNamespace(container_setup=SimpleNamespace(prepare_agent_invocation=Mock())),
        next_milestone=lambda: "m1",
        task=lambda mid: "Update the specs.",
        tag_commit=lambda mid: None,
        submit=Mock(),
    )
    workflow.provider = OpenSpecProvider(workflow.config, None)
    workflow.provider.prepare_runtime_config = Mock()
    workflow.provider.prompt = lambda stage, *args: stage
    return WorkflowRunner(
        integration=workflow,
        container_name="unused-test-container",
        output_dir=str(tmp_path / "log"),
        workdir="/testbed",
        repo_src_dirs=["src"],
        agent_name="codex",
        model="test-model",
        timeout_ms=60000,
    )


@pytest.mark.parametrize("status", ["failed", "interrupted"])
@pytest.mark.parametrize("provider_cls", [OpenSpecProvider, ArtifactNetProvider])
def test_incomplete_stage_reexecutes_agent(agent, monkeypatch, status, provider_cls):
    record = agent.workflow.state.milestone("m1")
    record["stages"][SYNC] = {"status": status, "attempts": 1, "elapsed_seconds": 0}
    provider = provider_cls(agent.workflow.config, None)
    for method in ("prepare_runtime_config", "prompt"):
        setattr(provider, method, getattr(agent.workflow.provider, method))
    agent.workflow.provider = provider
    calls = []

    def invoke(self, prompt):
        calls.append(prompt.splitlines()[0])
        return True, "thread-" + prompt.splitlines()[0]

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.run()
    assert calls == [SYNC, ARCHIVE]
    assert record["stages"][SYNC]["attempts"] == 2
    agent.workflow.submit.assert_called_once()


@pytest.mark.parametrize("override,expected", [(None, 60000), (10000, 10000), (120000, 60000)])
def test_recovery_timeout_applies_per_invocation_and_restores_default(agent, monkeypatch, override, expected):
    clock = [0.0]
    monkeypatch.setattr("workflows.runner.time.monotonic", lambda: clock[0])
    calls = []

    def invoke(self, prompt):
        calls.append(self.timeout_ms)
        clock[0] += 6
        return True, "thread-" + prompt

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.send_recover_message(timeout_ms=override)
    assert calls == [expected, expected]  # The second stage does not share a deadline.
    assert agent.timeout_ms == 60000
    record = agent.workflow.state.milestone("m1")
    assert record["elapsed_seconds"] == 12
    assert all(entry["allocated_seconds"] == expected / 1000 for entry in record["stages"].values())
    assert agent.workflow.orchestrator.container_setup.prepare_agent_invocation.call_count == 2


def test_recovery_override_limits_resume_and_later_fresh_stage(agent, monkeypatch):
    agent.workflow.state.milestone("m1")["stages"][SYNC] = {
        "status": "failed", "attempts": 1, "elapsed_seconds": 0, "session_id": "sync-thread",
    }
    resumed, fresh = [], []

    def resume(self, session_id, prompt, timeout_ms=None):
        resumed.append((session_id, timeout_ms))
        return True

    def invoke(self, prompt):
        fresh.append(self.timeout_ms)
        return True, "archive-thread"

    monkeypatch.setattr(AgentRunner, "resume_session", resume)
    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.send_recover_message(timeout_ms=5000)
    assert resumed == [("sync-thread", 5000)]
    assert fresh == [5000]
    assert agent.timeout_ms == 60000


@pytest.mark.parametrize("stage_remaining,milestone_remaining,expected", [(2, 10, 2000), (10, 1, 1000)])
def test_recovery_keeps_cumulative_budget_limits(agent, monkeypatch, stage_remaining, milestone_remaining, expected):
    agent.workflow.config["stages"] = [{"name": SYNC}]
    record = agent.workflow.state.milestone("m1")
    record["elapsed_seconds"] = 28800 - milestone_remaining
    record["stages"][SYNC] = {
        "status": "failed", "attempts": 1, "elapsed_seconds": 7200 - stage_remaining,
    }
    calls = []

    def invoke(self, prompt):
        calls.append(self.timeout_ms)
        return True, "sync-thread"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.send_recover_message(timeout_ms=10000)
    assert calls == [expected]
    assert agent.timeout_ms == 60000


def test_budget_exhaustion_does_not_skip_sync_or_prepare_an_agent(agent, monkeypatch):
    agent.workflow.state.milestone("m1")["stages"][SYNC] = {
        "status": "failed", "attempts": 2, "elapsed_seconds": 0,
    }
    monkeypatch.setattr(AgentRunner, "run", lambda *args: pytest.fail("budget is exhausted"))
    assert not agent.send_recover_message(timeout_ms=5000)
    assert "budget exhausted" in agent._last_fatal_error
    assert agent.timeout_ms == 60000
    agent.workflow.orchestrator.container_setup.prepare_agent_invocation.assert_not_called()
    agent.workflow.submit.assert_not_called()


def test_interrupted_process_stops_before_environment_preparation(agent, monkeypatch):
    agent.workflow.state.milestone("m1")["stages"][SYNC] = {
        "status": "running", "attempts": 1, "elapsed_seconds": 0,
        "started_at": time.time() - 5, "allocated_seconds": 10,
        "invocation_id": "a" * 12,
    }
    order = []
    monkeypatch.setattr(agent, "_kill_container_invocation", lambda reason: order.append("kill") or True)
    agent.workflow.orchestrator.container_setup.prepare_agent_invocation.side_effect = lambda: order.append("prepare")

    def invoke(self, prompt):
        order.append(prompt.splitlines()[0])
        return True, "thread-" + prompt.splitlines()[0]

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.send_recover_message(timeout_ms=5000)
    assert order == ["kill", "prepare", SYNC, "prepare", ARCHIVE]


@pytest.mark.parametrize("has_thread", [False, True])
def test_interrupted_session_is_recovered_only_from_its_raw_stdout_range(agent, monkeypatch, has_thread):
    agent.workflow.config["stages"] = [{"name": SYNC}]
    stdout = agent.log_dir / "agent_stdout.txt"
    earlier = 'Earlier stage: 中文\n{"type":"thread.started","thread_id":"earlier-thread"}\n'.encode()
    current = b'not JSON\n'
    if has_thread:
        current += b'{"type":"thread.started","thread_id":"current-thread"}\n'
    stdout.write_bytes(earlier + current)
    record = agent.workflow.state.milestone("m1")
    entry = record["stages"][SYNC] = {
        "status": "running", "attempts": 1, "elapsed_seconds": 0,
        "started_at": time.time() - 1, "allocated_seconds": 60,
        "stdout_offset": len(earlier),
    }
    fresh = Mock(return_value=(True, "new-thread"))
    resume = Mock(return_value=True)
    monkeypatch.setattr(AgentRunner, "run", fresh)
    monkeypatch.setattr(agent, "resume_session", resume)

    assert agent.run(), agent._last_fatal_error
    if has_thread:
        assert resume.call_args.args[0] == "current-thread"
        fresh.assert_not_called()
    else:
        fresh.assert_called_once()
        resume.assert_not_called()
    assert "earlier-thread" not in entry.get("session_ids", [])
    interrupted = json.loads((agent.workflow.root / f"milestones/m1/stages/{SYNC}/attempt-1/result.json").read_text())
    assert interrupted["stdout_offset"] == len(earlier)
    assert interrupted["stdout_end"] == len(earlier + current)
    assert stdout.read_bytes() == earlier + current
    assert list(agent.workflow.root.rglob("agent_stdout.txt")) == []


@pytest.mark.parametrize("success", [False, True])
def test_attempt_records_raw_stdout_bounds_without_log_copies(agent, monkeypatch, success):
    agent.workflow.config["stages"] = [{"name": SYNC}]
    stdout = agent.log_dir / "agent_stdout.txt"
    earlier = "Earlier stage: 中文\n".encode()
    current = b'{"type":"thread.started","thread_id":"current-thread"}\n'
    stdout.write_bytes(earlier)

    def invoke(self, prompt):
        with stdout.open("ab") as stream:
            stream.write(current)
        return success, "current-thread"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.run() is success
    entry = agent.workflow.state.milestone("m1")["stages"][SYNC]
    assert entry["stdout_offset"] == len(earlier)
    assert entry["stdout_end"] == len(earlier + current)
    attempt = agent.workflow.root / f"milestones/m1/stages/{SYNC}/attempt-1"
    result = json.loads((attempt / "result.json").read_text())
    assert result["stdout_offset"] == len(earlier)
    assert result["stdout_end"] == len(earlier + current)
    assert {path.name for path in attempt.iterdir()} == {"prompt.md", "result.json"}
    assert stdout.read_bytes() == earlier + current


@pytest.mark.parametrize("returned_id", ["earlier-stage-thread", "unstarted-placeholder"])
def test_failed_fresh_stage_without_thread_retries_fresh(agent, monkeypatch, returned_id):
    agent.workflow.config["stages"] = [{"name": SYNC}]
    stdout = agent.log_dir / "agent_stdout.txt"
    stdout.write_text('{"type":"thread.started","thread_id":"earlier-stage-thread"}\n')
    calls = []

    def invoke(self, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            # Both a stale global extraction and an early copy failure must not
            # provide a session to resume for this fresh stage.
            return False, returned_id
        with stdout.open("a") as stream:
            stream.write('{"type":"thread.started","thread_id":"new-stage-thread"}\n')
        return True, "new-stage-thread"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    monkeypatch.setattr(agent, "resume_session", lambda *a, **kw: pytest.fail("must retry with a fresh session"))
    assert not agent.run()
    entry = agent.workflow.state.milestone("m1")["stages"][SYNC]
    assert entry["session_id"] is None
    assert entry.get("session_ids", []) == []
    assert not (agent.log_dir / "session_id.txt").exists()

    assert agent.run(), agent._last_fatal_error
    assert len(calls) == 2
    assert entry["session_id"] == "new-stage-thread"
    assert entry["session_ids"] == ["new-stage-thread"]


@pytest.mark.parametrize("known_id", [None, "this-stage-thread"])
@pytest.mark.parametrize("emitted_id", [None, "new-thread-from-this-invocation"])
def test_session_extraction_uses_only_current_invocation_output(agent, monkeypatch, known_id, emitted_id):
    stdout = agent.log_dir / "agent_stdout.txt"
    prior = '旧日志\n{"type":"thread.started","thread_id":"another-stage-thread"}\n'.encode()
    current = b"non-JSON output\n"
    if emitted_id:
        current += (json.dumps({"type": "thread.started", "thread_id": emitted_id}) + "\n").encode()
    stdout.write_bytes(prior + current)
    agent.current_stage = {"stdout_offset": len(prior), "session_id": known_id}
    agent.session_id = "base-runner-placeholder"
    monkeypatch.setattr(
        agent._framework, "extract_session_id_from_container",
        lambda *a: pytest.fail("latest rollout may belong to another stage or a subagent"),
    )

    agent._update_session_id_from_output()

    assert agent.session_id == (emitted_id or known_id)
    if agent.session_id:
        assert (agent.log_dir / "session_id.txt").read_text() == agent.session_id


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_retry_preserves_saved_attempt_bounds_when_result_write_was_interrupted(agent, monkeypatch, status):
    agent.workflow.config["stages"] = [{"name": SYNC}]
    stdout = agent.log_dir / "agent_stdout.txt"
    raw = agent.log_dir / "codex"
    raw.mkdir()

    def append_turn(thread, tokens):
        rows = [
            {"type": "thread.started", "thread_id": thread},
            {"type": "turn.completed", "usage": {"input_tokens": tokens, "output_tokens": 2}},
        ]
        with stdout.open("a") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in rows))
        (raw / f"rollout-{thread}.jsonl").write_text(
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.2-codex"}}) + "\n"
        )

    append_turn("thread-before-crash", 10)
    original_end = stdout.stat().st_size
    entry = agent.workflow.state.milestone("m1")["stages"][SYNC] = {
        "status": status, "attempts": 1, "elapsed_seconds": 1,
        "stdout_offset": 0, "stdout_end": original_end,
    }
    agent.workflow.state.save()  # result.json was never written before the crash.
    previous_result = agent.workflow.root / f"milestones/m1/stages/{SYNC}/attempt-1/result.json"

    def invoke(self, prompt):
        saved = json.loads(previous_result.read_text())
        assert saved["status"] == status
        assert saved["stdout_offset"] == 0
        assert saved["stdout_end"] == original_end
        append_turn("thread-after-crash", 7)
        return True, "thread-after-crash"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.run(), agent._last_fatal_error
    assert entry["attempts"] == 2
    assert entry["stdout_offset"] == original_end
    stats = collect_stage_usage(agent.workflow.root, "m1", SYNC, entry)
    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 17
    assert stats["unique_session_count"] == 2


def test_recovery_timeout_restored_on_interruption(agent, monkeypatch):
    def interrupt(self, prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(AgentRunner, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        agent.send_recover_message(timeout_ms=5000)
    assert agent.timeout_ms == 60000
    assert agent.workflow.state.milestone("m1")["stages"][SYNC]["allocated_seconds"] == 5


def test_agent_env_preserves_go_path_and_adds_local_proxy_token(agent, monkeypatch):
    monkeypatch.setattr(AgentRunner, "_get_exec_env_vars", lambda self: ["-e", "PATH=/sealed/go/bin:/usr/bin"])
    agent.workflow.runtime.api_key_env_args = lambda: ["-e", "WORKFLOW_API_TOKEN=local"]
    args = agent._get_exec_env_vars()
    assert "PATH=/opt/workflow/bin:/sealed/go/bin:/usr/bin" in args
    assert args[-2:] == ["-e", "WORKFLOW_API_TOKEN=local"]


def test_proxy_milestone_binding_precedes_agent_calls_and_refreshes_on_resume(agent, monkeypatch):
    current = ["m1"]
    agent.workflow.next_milestone = lambda: current[0]

    def invoke(self, prompt):
        self.workflow.provider.prepare_runtime_config.assert_called_with(current[0])
        return False, "fixture-thread"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    monkeypatch.setattr(agent, "resume_session", lambda *args, **kwargs: invoke(agent, "")[0])
    assert not agent.run()
    assert not agent.send_recover_message()
    current[0] = "m2"
    assert not agent.run()
    assert [call.args for call in agent.workflow.provider.prepare_runtime_config.call_args_list] == [
        ("m1",), ("m1",), ("m2",),
    ]


@pytest.mark.parametrize("with_workflow", [False, True])
def test_outer_runner_prepares_only_baseline_invocations(tmp_path, monkeypatch, with_workflow):
    completed = set()
    config = E2EConfig()
    config.config["retry_and_timing"].update(max_no_progress_attempts=1, recovery_wait_seconds=0)
    dag = SimpleNamespace(
        all_milestones={"m1"}, completed_milestones=completed, failed_milestones=set(), skipped_milestones=set(),
        is_done=lambda: bool(completed),
        get_state_snapshot=lambda: {"completed": set(completed), "submitted": set(), "failed": set()},
    )
    prepare = Mock()
    orchestrator = SimpleNamespace(
        config=config, dag=dag, container_name="unused-test-container", trial_root=tmp_path,
        container_setup=SimpleNamespace(prepare_agent_invocation=prepare), _update_task_queue_file=lambda *a: None,
        _load_summary_or_init=lambda: {"results": {}},
    )
    fake_agent = SimpleNamespace(run=lambda: completed.add("m1") or True, _last_fatal_error=None)
    workflow = SimpleNamespace(create_runner=lambda **kwargs: fake_agent, progress_snapshot=lambda: set())
    monkeypatch.setattr("harness.e2e.run_e2e.E2EAgentRunner", lambda **kwargs: fake_agent)
    trial = E2ETrialRunner(
        orchestrator=orchestrator, agent_output_dir=tmp_path, workdir="/testbed", repo_src_dirs=["src"],
        agent_name="codex", model="test-model", timeout_ms=60000, prompt_version="v1",
        workflow=workflow if with_workflow else None,
    )
    monkeypatch.setattr(trial, "_wait_for_evaluations", lambda: "all_done")
    assert trial.run_agent_with_recovery()
    assert prepare.call_count == (0 if with_workflow else 1)
