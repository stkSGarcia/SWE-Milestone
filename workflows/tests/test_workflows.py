from __future__ import annotations

import copy
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.e2e.agent_runner import AgentRunner
from harness.utils.src_filter import SrcFileFilter
from workflows.config import freeze, load, restore, stage_names, validate
from workflows.integration import WorkflowIntegration
from workflows.metrics import collect_metrics
from workflows.providers.openspec import OpenSpecProvider
from workflows.runner import WorkflowRunner
from workflows.state import WorkflowState

CONFIGS = Path(__file__).parents[1] / "configs"


def git(repo, *args, check=True, input=None):
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=check, input=input)


@pytest.fixture
def config():
    return load(CONFIGS / "openspec.yaml")


def test_freeze_resume_and_tampering(tmp_path, config):
    source = CONFIGS / "openspec.yaml"
    binding = freeze(source, tmp_path, "codex")
    assert restore(binding, tmp_path) == config
    assert restore(binding, tmp_path, source) == config
    frozen = tmp_path / binding["path"]
    changed = copy.deepcopy(config)
    changed["budgets"]["stage_seconds"] += 1
    frozen.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest mismatch"):
        restore(binding, tmp_path)


def test_cannot_switch_workflow_on_resume(tmp_path):
    with pytest.raises(ValueError, match="baseline"):
        restore(None, tmp_path, CONFIGS / "openspec.yaml")
    binding = freeze(CONFIGS / "openspec.yaml", tmp_path, "codex")
    with pytest.raises(ValueError, match="differs"):
        restore(binding, tmp_path, CONFIGS / "openspec_artifactnet.yaml")
    with pytest.raises(ValueError, match="codex"):
        freeze(CONFIGS / "openspec.yaml", tmp_path, "claude-code")


@pytest.mark.parametrize(
    "change",
    [
        {"stages": []},
        {"stages": [{"name": "inspect"}, {"name": "inspect"}]},
        {"budgets": {"max_attempts": 0}},
        {"versions": {"openspec": "^1.7.0"}},
        {"typo": True},
    ],
)
def test_invalid_workflows_rejected(config, change):
    config.update(change)
    with pytest.raises(ValueError):
        validate(config)


def test_no_inline_credentials():
    cfg = load(CONFIGS / "openspec_artifactnet.yaml")
    cfg["artifactnet"]["llm"]["api_key"] = "not-a-real-key"
    with pytest.raises(ValueError, match="Unknown"):
        validate(cfg)


class LocalRuntime:
    def __init__(self, repo):
        self.repo = repo
        self.container = "fixture"

    def exec(self, argv, *, input=None, **kwargs):
        assert argv[0] == "git", argv
        return git(self.repo, *argv[1:], input=input).stdout


class Provider(OpenSpecProvider):
    def prompt(self, stage, mid, change, task, source_dirs):
        return f"{stage}\n{mid}\n{change}\n{task}"


@pytest.fixture
def workflow(tmp_path, config):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@local")
    (repo / "src").mkdir()
    (repo / "src" / "answer.py").write_text("value = 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "baseline")
    srs = tmp_path / "srs" / "m1"
    srs.mkdir(parents=True)
    (srs / "SRS.md").write_text("Set value to 1.")
    orchestrator = SimpleNamespace(
        trial_root=tmp_path / "trial",
        container_name="fixture",
        srs_root=srs.parent,
        dag=SimpleNamespace(get_next_runnable=lambda: ["m1"]),
        src_filter=SrcFileFilter(["src"], ["tests/**"]),
        container_setup=SimpleNamespace(prepare_agent_invocation=lambda: None),
        _docker_exec_git=lambda *args: git(repo, *args, check=False),
    )
    instance = WorkflowIntegration(orchestrator, config)
    instance.runtime = LocalRuntime(repo)
    instance.provider = Provider(config, None)
    return instance


def runner(workflow):
    return WorkflowRunner(
        integration=workflow,
        container_name="fixture",
        output_dir=str(workflow.root.parent / "log"),
        workdir="/testbed",
        repo_src_dirs=["src"],
        agent_name="codex",
        model="test-model",
        timeout_ms=60000,
        prompt_version="v2",
    )


def complete_stage(workflow, text):
    stage, _mid, _change, _ = text.split("\n", 3)
    if stage.endswith("apply-change"):
        (workflow.runtime.repo / "src" / "answer.py").write_text("value = 1\n")
    return stage


def test_all_stages_then_one_source_submission(workflow, monkeypatch):
    calls = []

    def invoke(self, text, session_id=None):
        calls.append(complete_stage(workflow, text))
        assert workflow.tag_commit("m1") is None
        return True, f"session-{len(calls)}"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    agent = runner(workflow)
    assert agent.run(), agent._last_fatal_error
    assert calls == stage_names(workflow.config)
    commit = workflow.tag_commit("m1")
    assert git(workflow.runtime.repo, "show", f"{commit}:src/answer.py").stdout == "value = 1\n"
    assert workflow.state.milestone("m1")["status"] == "submitted"
    assert agent.run()  # tag is waiting for the watcher; don't rerun the milestone
    assert len(calls) == 4


def test_progress_counts_completed_stages_and_submissions_but_not_retries(workflow):
    record = workflow.state.milestone("m1")
    record["stages"]["openspec-propose"] = {"status": "failed", "attempts": 1}
    assert workflow.progress_snapshot() == set()
    record["stages"]["openspec-propose"]["attempts"] = 2
    assert workflow.progress_snapshot() == set()

    record["stages"]["openspec-propose"]["status"] = "complete"
    completed = workflow.progress_snapshot()
    assert completed == {("stage", "m1", "openspec-propose")}
    record["stages"]["openspec-apply-change"] = {"status": "failed", "attempts": 2}
    assert workflow.progress_snapshot() == completed

    record.update(status="submitted", submission_commit="a" * 40)
    assert workflow.progress_snapshot() - completed == {("submission", "m1", "a" * 40)}
    workflow.state.save()
    workflow.state = WorkflowState(workflow.root)
    assert completed < workflow.progress_snapshot()


def test_pending_submission_excludes_already_completed_milestones(workflow, monkeypatch):
    snapshot = {"completed": set(), "failed": set(), "skipped": set()}
    monkeypatch.setattr(workflow.orchestrator.dag, "get_state_snapshot", lambda: snapshot, raising=False)
    record = workflow.state.milestone("m1")
    record.update(status="submitted", submission_commit="a" * 40)
    assert workflow.pending_submissions() == {"m1"}
    snapshot["completed"].add("m1")
    assert workflow.pending_submissions() == set()


def test_failure_resumes_only_its_stage_session(workflow, monkeypatch):
    fresh, resumed = [], []

    def invoke(self, text, session_id=None):
        stage = text.splitlines()[0]
        fresh.append(stage)
        thread = "apply-thread" if stage.endswith("apply-change") else f"thread-{stage}"
        with (self.log_dir / "agent_stdout.txt").open("a") as stream:
            stream.write(json.dumps({"type": "thread.started", "thread_id": thread}) + "\n")
        if stage.endswith("apply-change"):
            return False, thread
        complete_stage(workflow, text)
        return True, thread

    def resume(self, session_id, text, timeout_ms=None):
        resumed.append(session_id)
        complete_stage(workflow, text)
        return True

    monkeypatch.setattr(AgentRunner, "run", invoke)
    monkeypatch.setattr(AgentRunner, "resume_session", resume)
    assert not runner(workflow).run()
    # Reconstruct state and runner just as a process restart would.
    workflow.state = WorkflowState(workflow.root)
    assert runner(workflow).send_recover_message()
    assert fresh.count("openspec-propose") == 1
    assert resumed == ["apply-thread"]
    assert fresh[-2:] == ["openspec-sync-specs", "openspec-archive-change"]


def test_tag_creation_crash_is_reconciled_without_new_commit(workflow, monkeypatch):
    record = workflow.state.milestone("m1")
    original = workflow.runtime.exec

    def crash(argv, **kwargs):
        if argv[:2] == ["git", "tag"]:
            raise RuntimeError("simulated interruption before tag")
        return original(argv, **kwargs)

    monkeypatch.setattr(workflow.runtime, "exec", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        workflow.submit("m1", record)
    intended = record["submission_commit"]
    count = git(workflow.runtime.repo, "rev-list", "--count", "HEAD").stdout
    workflow.state = WorkflowState(workflow.root)
    monkeypatch.setattr(workflow.runtime, "exec", original)
    workflow.submit("m1", workflow.state.milestone("m1"))
    assert workflow.tag_commit("m1") == intended
    assert git(workflow.runtime.repo, "rev-list", "--count", "HEAD").stdout == count


def test_submission_leaves_source_filtering_to_benchmark(workflow, monkeypatch):
    from harness.e2e.orchestrator import E2EOrchestrator

    repo = workflow.runtime.repo
    files = [
        "openspec/specs/example.md", ".artnet/config.yaml", ".codex/skills/example/SKILL.md",
        "tests/test_answer.py", "src/tests/test_answer.py", "go.mod", "README.md",
    ]
    for path in files:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")
    git(repo, "add", ".")
    with monkeypatch.context() as patch:
        patch.setattr(
            workflow.orchestrator.src_filter, "should_include_in_snapshot",
            lambda *a: pytest.fail("submission must leave source filtering to the benchmark"),
        )
        workflow.submit("m1", workflow.state.milestone("m1"))
    submitted = git(repo, "ls-tree", "-r", "--name-only", "agent-impl-m1").stdout.splitlines()
    assert set(submitted) == {"src/answer.py", "src/tests/test_answer.py", "tests/test_answer.py", "go.mod", "README.md"}
    assert (repo / "openspec/specs/example.md").exists()  # retained for subsequent milestones

    # Use the real benchmark filter: committed tests are not graded source.
    snapshot = repo.parent / "source_snapshot.tar"
    git(repo, "archive", "--format=tar", f"--output={snapshot}", "agent-impl-m1", "src", "go.mod")
    workflow.orchestrator.src_filter = SrcFileFilter(["src"], ["tests/**", "src/tests/**"])
    E2EOrchestrator._filter_tar_archive(workflow.orchestrator, snapshot, extra_build_manifests={"go.mod"})
    with tarfile.open(snapshot) as archive:
        assert {member.name for member in archive if member.isfile()} == {"src/answer.py", "go.mod"}


@pytest.mark.parametrize("staged", [False, True])
def test_submission_preserves_source_and_manifest_deletions(workflow, staged):
    repo = workflow.runtime.repo
    deleted = ["src/answer.py", "go.mod", "src/module/go.mod", "src/space [x]\nname.py"]
    for name in deleted:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("baseline\n")
    (repo / "src/retained.py").write_text("value = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "baseline with manifests")
    for name in deleted:
        (repo / name).unlink()
    if staged:
        git(repo, "add", "-A")

    workflow.submit("m1", workflow.state.milestone("m1"))
    files = git(repo, "ls-tree", "-rz", "--name-only", "agent-impl-m1").stdout.split("\0")
    assert set(files) - {""} == {"src/retained.py"}
    assert not git(repo, "status", "--porcelain").stdout


def test_submission_preserves_staged_rename(workflow):
    repo = workflow.runtime.repo
    renamed = "src/renamed [x]\nfile.py"
    git(repo, "mv", "src/answer.py", renamed)
    workflow.submit("m1", workflow.state.milestone("m1"))
    files = git(repo, "ls-tree", "-rz", "--name-only", "agent-impl-m1").stdout.split("\0")
    assert set(files) - {""} == {renamed}
    assert git(repo, "show", f"agent-impl-m1:{renamed}").stdout == "value = 0\n"


@pytest.mark.parametrize("initialized", [False, True])
def test_runtime_config_is_prepared_on_fresh_and_resume(workflow, monkeypatch, initialized):
    calls = []
    workflow.state.data["initialized"] = initialized
    workflow.provider = SimpleNamespace(
        initialize=lambda: calls.append("initialize"),
        prepare_runtime_config=lambda: calls.append("configure"),
        verify_tools=dict,
    )
    monkeypatch.setattr(workflow.runtime, "install_cli_launchers", lambda: None, raising=False)
    monkeypatch.setattr(workflow.runtime, "verify_services", lambda: None, raising=False)
    workflow.initialize()
    assert calls == (["configure"] if initialized else ["initialize", "configure"])


def test_budget_stops_repeated_failed_stages(workflow, monkeypatch):
    workflow.config["budgets"]["max_attempts"] = 1
    monkeypatch.setattr(AgentRunner, "run", lambda *a, **kw: (False, "failed-thread"))
    agent = runner(workflow)
    assert not agent.run()
    assert not agent.run()
    assert "budget exhausted" in agent._last_fatal_error
    assert workflow.tag_commit("m1") is None


def test_metrics_ignore_legacy_external_usage(tmp_path):
    usage = tmp_path / "usage"
    usage.mkdir()
    legacy = usage / "artifactnet.jsonl"
    legacy.write_text("obsolete partial usage record\n")
    data = collect_metrics(tmp_path)
    assert data["schema_version"] == 3
    assert "artifactnet" not in data
    assert data["artifactnet_log"]["legacy_records"] == 1
    assert data["artifactnet_usage"]["cost_complete"] is False
    assert data["agent_usage_source"] == "../agent_stats.json"
    assert legacy.read_text() == "obsolete partial usage record\n"


def test_cleanup_stops_proxy_before_collecting_final_usage(workflow, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow.runtime, "close", lambda: calls.append("stop-proxy"), raising=False)
    monkeypatch.setattr("workflows.metrics.collect_metrics", lambda root: calls.append("collect"))
    workflow.close()
    assert calls == ["stop-proxy", "collect"]


def test_submission_guard_requires_all_stages_and_exact_commit(workflow):
    record = workflow.state.milestone("m1")
    record["submission_commit"] = "a" * 40
    workflow.state.save()
    with pytest.raises(RuntimeError, match="not authorized"):
        workflow.assert_submission("m1", "a" * 40)
    record["stages"] = {stage: {"status": "complete"} for stage in stage_names(workflow.config)}
    workflow.state.save()
    workflow.assert_submission("m1", "a" * 40)
    with pytest.raises(RuntimeError, match="not authorized"):
        workflow.assert_submission("m1", "b" * 40)


def test_surviving_old_process_stops_recovery(workflow, monkeypatch):
    import time

    record = workflow.state.milestone("m1")
    record["stages"]["openspec-propose"] = {
        "status": "running",
        "attempts": 1,
        "elapsed_seconds": 0,
        "started_at": time.time(),
        "allocated_seconds": 60,
        "invocation_id": "a" * 12,
    }
    workflow.state.save()
    agent = runner(workflow)
    monkeypatch.setattr(agent, "_kill_container_invocation", lambda *args: False)
    monkeypatch.setattr(AgentRunner, "run", lambda *args: pytest.fail("must not start a concurrent agent"))
    assert not agent.run()
    assert "still alive" in agent._last_fatal_error
    assert record["stages"]["openspec-propose"]["status"] == "running"


def test_no_resume_session_is_respected_after_interruption(workflow, monkeypatch):
    import time

    agent = runner(workflow)
    stdout = agent.log_dir / "agent_stdout.txt"
    stdout.write_text(json.dumps({"type": "thread.started", "thread_id": "old-thread"}) + "\n")
    record = workflow.state.milestone("m1")
    record["stages"]["openspec-propose"] = {
        "status": "running",
        "attempts": 1,
        "elapsed_seconds": 0,
        "started_at": time.time(),
        "allocated_seconds": 60,
        "session_id": "old-thread",
        "stdout_offset": 0,
    }
    workflow.state.save()
    workflow.invalidate_sessions()
    calls = []

    def invoke(self, text, session_id=None):
        calls.append(complete_stage(workflow, text))
        return True, "new-" + str(len(calls))

    monkeypatch.setattr(AgentRunner, "run", invoke)
    monkeypatch.setattr(AgentRunner, "resume_session", lambda *a, **kw: pytest.fail("old session must not resume"))
    assert agent.run(), agent._last_fatal_error
    assert calls == stage_names(workflow.config)


def test_usage_sums_fresh_threads_without_double_counting_resumes(tmp_path):
    from harness.e2e.log_parser.codex import CodexLogParser
    from workflows.metrics import parse_workflow_stdout

    stdout = tmp_path / "agent_stdout.txt"
    rows = []
    for sid, count in [("thread-a", 10), ("thread-b", 7), ("thread-a", 16)]:
        rows += [
            {"type": "thread.started", "thread_id": sid},
            {"type": "turn.completed", "usage": {"input_tokens": count, "output_tokens": 2, "cached_input_tokens": 0}},
        ]
    stdout.write_text("".join(json.dumps(row) + "\n" for row in rows))
    logs = tmp_path / "codex"
    logs.mkdir()
    for sid in ("thread-a", "thread-b"):
        (logs / f"rollout-{sid}.jsonl").write_text(
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.2-codex"}}) + "\n"
        )
    stats = parse_workflow_stdout(CodexLogParser(), stdout, logs)
    usage = stats["modelUsage"]["gpt-5.2-codex"]
    assert usage["inputTokens"] == 23  # latest a=16 plus b=7, never 10+16+7 or only 16
    assert usage["outputTokens"] == 4
    assert stats["session_count"] == 3
    assert stats["unique_session_count"] == 2


def test_launcher_forwards_workflow_in_fresh_and_resume(tmp_path, monkeypatch):
    from scripts import run_all

    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    config_path = CONFIGS / "openspec.yaml"
    policy = SimpleNamespace(repo_name=repo.name, mode="absent", sha256="a" * 64)
    monkeypatch.setattr(run_all, "image_for_runtime_policy", lambda policy: "fixture:base")
    args = {
        "repo": repo,
        "agent": "codex",
        "model": "model",
        "timeout": 60,
        "trial_name": "trial",
        "reasoning_effort": None,
        "agent_version": None,
        "force": False,
        "runtime_policy": policy,
        "workflow_config": config_path,
    }
    command, mode = run_all.build_cmd(**args)
    assert mode == "fresh"
    assert command[command.index("--workflow-config") + 1] == str(config_path)
    target = repo / "e2e_trial" / "trial"
    target.mkdir(parents=True)
    (target / "trial_metadata.json").write_text("{}")
    command, mode = run_all.build_cmd(**args)
    assert mode == "resume"
    assert command[command.index("--workflow-config") + 1] == str(config_path)


def test_force_cleanup_uses_recorded_trial_resources_without_api_keys(tmp_path, monkeypatch):
    from workflows import runtime as runtime_module

    binding = freeze(CONFIGS / "openspec_artifactnet.yaml", tmp_path, "codex")
    (tmp_path / "trial_metadata.json").write_text(json.dumps({"workflow": binding}))
    root = tmp_path / "workflow"
    identity = {
        "schema_version": 1,
        "project": "swe-workflow-a0123456789bcdef",
        "agent_container": "fixture-original-agent",
    }
    (root / "runtime.json").write_text(json.dumps(identity))
    (root / "compose.resolved.yaml").write_text("services: {}")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    compose_calls = []
    docker_calls = []

    def run_compose(argv, **kwargs):
        assert kwargs["env"]["OPENROUTER_API_KEY"] == "unused-for-cleanup"
        compose_calls.append(argv)
        return ""

    monkeypatch.setattr(runtime_module, "command", run_compose)
    monkeypatch.setattr(runtime_module.subprocess, "run", lambda argv, **kwargs: docker_calls.append(argv))
    runtime_module.discard_trial_runtime(tmp_path)
    assert docker_calls == [
        ["docker", "network", "disconnect", identity["project"] + "_default", identity["agent_container"]]
    ]
    assert len(compose_calls) == 2
    for argv in compose_calls:
        assert argv[argv.index("-p") + 1] == identity["project"]
        assert argv[argv.index("-f") + 1] == str(root / "compose.resolved.yaml")
    assert "-v" not in compose_calls[0]
    assert compose_calls[1][-3:] == ["down", "-v", "--remove-orphans"]
