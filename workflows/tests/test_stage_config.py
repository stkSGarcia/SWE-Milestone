"""Configured skills are agent stages, without controller artifact validation."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest

from harness.e2e.agent_runner import AgentRunner
from workflows.config import digest, load, restore, validate
from workflows.providers import create_provider
from workflows.runner import WorkflowRunner
from workflows.state import WorkflowState

CONFIGS = Path(__file__).parents[1] / "configs"
CUSTOM_STAGES = [
    {"name": "write-proposal", "arguments": ["{task}"]},
    {"name": "implement-tasks", "arguments": ["{change_id}"]},
    {"name": "publish-specs"},
    {"name": "finish-change", "arguments": ["{change_id}"]},
]


@pytest.fixture
def config():
    value = load(CONFIGS / "openspec.yaml")
    value["stages"] = copy.deepcopy(CUSTOM_STAGES)
    return value


def test_custom_names_select_skills_and_render_arguments(config):
    task, change, milestone = "Keep the literal {change_id} in the feature.", "m-one-a1b2", "m1"
    config["stages"][0]["arguments"] = [
        "Specification: {task}", "change={change_id}", "milestone={milestone}", '{"literal": "{other}"}',
    ]

    def execute(argv):
        if argv == ["openspec", "--version"]:
            return "1.7.0\n"
        if argv == ["node", "--version"]:
            return "v26.0.0\n"
        assert argv[:2] == ["test", "-s"], argv
        return ""

    runtime = SimpleNamespace(exec=Mock(side_effect=execute))
    provider = create_provider(validate(config), runtime)
    assert provider.verify_tools()["openspec"] == "1.7.0"
    checked = [call.args[0][-1] for call in runtime.exec.call_args_list if call.args[0][:2] == ["test", "-s"]]
    assert checked == [f".codex/skills/{stage['name']}/SKILL.md" for stage in CUSTOM_STAGES]
    prompt = provider.prompt("write-proposal", milestone, change, task, ["src"])
    assert prompt.startswith("$write-proposal\n")
    for expected in (f"Specification: {task}", f"change={change}", f"milestone={milestone}", '{"literal": "{other}"}'):
        assert expected in prompt


@pytest.mark.parametrize("arguments", [None, [], ["Use the attached specification."]])
def test_arguments_are_explicit_but_controller_identity_is_always_present(config, arguments):
    stage = {"name": "custom-check"}
    if arguments is not None:
        stage["arguments"] = arguments
    config["stages"] = [stage]
    provider = create_provider(validate(config), SimpleNamespace())
    task = "This SRS is not implicitly an argument."
    prompt = provider.prompt("custom-check", "m7", "m-seven-controller-id", task, ["src"])
    assert prompt.startswith("$custom-check\n")
    assert "m-seven-controller-id" in prompt
    assert "m7" in prompt
    assert task not in prompt
    if arguments:
        assert arguments[0] in prompt


@pytest.mark.parametrize("provider", ["openspec", "openspec_artifactnet"])
@pytest.mark.parametrize("stages", [
    [{"name": "single-check"}],
    list(reversed(CUSTOM_STAGES)),
    [*CUSTOM_STAGES, {"name": "audit-results", "arguments": ["{milestone}"]}],
])
def test_any_nonempty_unique_skill_sequence_is_accepted(provider, stages):
    config = load(CONFIGS / f"{provider}.yaml")
    config["stages"] = copy.deepcopy(stages)
    assert validate(config)["stages"] == stages


@pytest.mark.parametrize("stage", [
    {"name": "../outside"},
    {"name": "WriteProposal"},
    {"name": "write-proposal", "role": "propose"},
    {"name": "write-proposal", "arguments": "a string"},
    {"name": "write-proposal", "arguments": [{}]},
    {"name": "write-proposal", "arguments": [None]},
    {"name": "write-proposal", "argument": []},
    {"arguments": []},
    None,
    [],
])
def test_invalid_stage_values_raise_config_errors(config, stage):
    config["stages"][0] = stage
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("stages", [[], [{"name": "repeat"}, {"name": "repeat", "arguments": ["another task"]}]])
def test_empty_sequences_and_duplicate_skill_names_are_rejected(config, stages):
    config["stages"] = stages
    with pytest.raises(ValueError):
        validate(config)


def test_structured_frozen_config_restores_unchanged_and_rejects_tampering(tmp_path, config):
    config["stages"][0]["arguments"] = ["{task}", "{change_id}"]
    config = validate(config)
    original = json.dumps(config, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(original.encode()).hexdigest()
    target = tmp_path / "workflow" / "config.resolved.json"
    target.parent.mkdir()
    target.write_text(original)
    binding = {"path": "workflow/config.resolved.json", "sha256": checksum, "provider": "openspec"}
    assert validate(config) == config
    assert digest(validate(config)) == checksum
    assert restore(binding, tmp_path) == config
    assert target.read_text() == original
    config["stages"][0]["arguments"] = ["Different instructions"]
    target.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="digest mismatch"):
        restore(binding, tmp_path)


def test_unsupported_schema_and_string_stages_are_rejected(config):
    unsupported_schema = copy.deepcopy(config)
    unsupported_schema["schema_version"] = 2
    with pytest.raises(ValueError):
        validate(unsupported_schema)
    config["stages"] = ["openspec-propose", "openspec-apply-change", "openspec-archive-change"]
    with pytest.raises(ValueError):
        validate(config)


def test_provider_registry_controls_configuration_and_factory(monkeypatch):
    from workflows import providers
    from workflows.config import stage_names

    class ChecklistProvider:
        config_keys = ("checks",)
        default_versions: ClassVar[dict[str, str]] = {"checker": "latest"}

        def __init__(self, config, runtime):
            self.config, self.runtime = config, runtime

        @classmethod
        def validate_config(cls, config):
            if config["checks"] != {"format": "json"}:
                raise ValueError("Expected JSON checks")

    monkeypatch.setitem(providers.PROVIDERS, "checklist", ChecklistProvider)
    config = validate({
        "schema_version": 1, "provider": "checklist", "runtime_image": "local/checklist:latest",
        "stages": [{"name": "inspect-code"}, {"name": "publish-report"}],
        "checks": {"format": "json"},
    })
    assert config["versions"] == {"checker": "latest"}
    assert stage_names(config) == ["inspect-code", "publish-report"]
    runtime = object()
    provider = create_provider(config, runtime)
    assert isinstance(provider, ChecklistProvider)
    assert provider.runtime is runtime
    config["checks"] = {"format": "invalid"}
    with pytest.raises(ValueError, match="JSON checks"):
        validate(config)


@pytest.fixture
def agent(tmp_path, config):
    config = validate(config)
    root = tmp_path / "workflow"
    runtime = SimpleNamespace(
        api_key_env_args=list,
        exec=Mock(side_effect=AssertionError("Stage execution must not run artifact checks")),
    )
    provider = create_provider(config, runtime)
    # These old hooks intentionally claim completion: they must never skip the agent.
    provider.validate = Mock(return_value={"artifacts": "complete"})
    provider.recover = Mock(return_value={"artifacts": "complete"})
    provider.active_changes = Mock(return_value=["unrelated-change"])
    workflow = SimpleNamespace(
        config=config, root=root, state=WorkflowState(root), provider=provider, runtime=runtime,
        orchestrator=SimpleNamespace(container_setup=SimpleNamespace(prepare_agent_invocation=Mock())),
        next_milestone=lambda: "m1", task=lambda mid: "Complete the requested work.", tag_commit=lambda mid: None,
        submit=Mock(),
    )
    return WorkflowRunner(
        integration=workflow, container_name="unused-test-container", output_dir=str(tmp_path / "log"),
        workdir="/testbed", repo_src_dirs=["src"], agent_name="codex", model="test-model", timeout_ms=60000,
    )


@pytest.mark.parametrize("status", ["pending", "failed", "interrupted", "running"])
def test_unfinished_stage_runs_agent_even_when_old_artifact_hooks_claim_done(agent, monkeypatch, status):
    workflow = agent.workflow
    record = workflow.state.milestone("m1")
    record["stages"]["write-proposal"] = {"status": "complete", "attempts": 1, "elapsed_seconds": 0}
    previous_attempts = 0 if status == "pending" else 1
    record["stages"]["implement-tasks"] = {
        "status": status, "attempts": previous_attempts, "elapsed_seconds": 0,
        "started_at": time.time() - 1, "allocated_seconds": 60,
    }
    calls = []

    def invoke(self, prompt):
        calls.append(prompt.splitlines()[0])
        return True, f"thread-{len(calls)}"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert agent.run(), agent._last_fatal_error
    assert calls == ["$implement-tasks", "$publish-specs", "$finish-change"]
    assert record["stages"]["implement-tasks"]["attempts"] == previous_attempts + 1
    assert all(stage["status"] == "complete" for stage in record["stages"].values())
    prompt_path = workflow.root / f"milestones/m1/stages/implement-tasks/attempt-{previous_attempts + 1}/prompt.md"
    assert prompt_path.is_file()
    assert list(workflow.root.rglob("validation.json")) == []
    workflow.provider.validate.assert_not_called()
    workflow.provider.recover.assert_not_called()
    workflow.provider.active_changes.assert_not_called()
    workflow.runtime.exec.assert_not_called()
    workflow.submit.assert_called_once_with("m1", record)


def test_agent_failure_keeps_stage_failed_and_blocks_submission(agent, monkeypatch):
    calls = []

    def invoke(self, prompt):
        calls.append(prompt.splitlines()[0])
        with (self.log_dir / "agent_stdout.txt").open("a") as stream:
            stream.write(json.dumps({"type": "thread.started", "thread_id": "failed-thread"}) + "\n")
        return False, "failed-thread"

    monkeypatch.setattr(AgentRunner, "run", invoke)
    assert not agent.run()
    assert calls == ["$write-proposal"]
    entry = agent.workflow.state.milestone("m1")["stages"]["write-proposal"]
    assert entry["status"] == "failed"
    assert entry["session_id"] == "failed-thread"
    agent.workflow.submit.assert_not_called()
    agent.workflow.provider.validate.assert_not_called()
    agent.workflow.provider.recover.assert_not_called()
    agent.workflow.runtime.exec.assert_not_called()
