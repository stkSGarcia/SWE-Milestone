from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from workflows import runtime as runtime_module
from workflows.config import load
from workflows.providers.artifactnet import ArtifactNetProvider
from workflows.runtime import WorkflowRuntime

CONFIGS = Path(__file__).parents[1] / "configs"


def test_upstream_credentials_stay_outside_agent(tmp_path, monkeypatch):
    config = load(CONFIGS / "openspec_artifactnet.yaml")
    config["artifactnet"]["embedding"]["api_key_env"] = "EMBEDDING_FIXTURE_KEY"
    monkeypatch.setenv("OPENROUTER_API_KEY", "llm-fixture-secret")
    monkeypatch.setenv("EMBEDDING_FIXTURE_KEY", "embedding-fixture-secret")
    calls = []
    monkeypatch.setattr(runtime_module, "command", lambda argv, **kwargs: calls.append(argv) or "ok")
    runtime = WorkflowRuntime(config, tmp_path, "fixture-agent")
    assert runtime.exec(["artnet", "retrieve", "--change", "fixture"]) == "ok"
    argv = calls[0]
    values = [argv[i + 1] for i, value in enumerate(argv) if value == "-e"]
    assert "OPENROUTER_API_KEY" not in values
    assert "EMBEDDING_FIXTURE_KEY" not in values
    assert "WORKFLOW_API_TOKEN=local" in values
    assert all("fixture-secret" not in value for value in argv)


def test_missing_proxy_api_key_stops_before_starting_services(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runtime = WorkflowRuntime(load(CONFIGS / "openspec_artifactnet.yaml"), tmp_path, "fixture-agent")
    monkeypatch.setattr(runtime_module, "command", lambda *args, **kwargs: pytest.fail("Docker must not be called"))
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        runtime._prepare_services()
    assert not (tmp_path / "compose.resolved.yaml").exists()
    baseline = WorkflowRuntime(load(CONFIGS / "openspec.yaml"), tmp_path, "fixture-agent")
    assert baseline.api_key_env_args() == []


@pytest.mark.parametrize("endpoint", [None, "http://api:8080/llm/v1", "https://openrouter.ai/api/v1", "opencode"])
def test_runtime_config_refresh_preserves_schema_and_binds_milestone(tmp_path, endpoint):
    config = load(CONFIGS / "openspec_artifactnet.yaml")
    config["artifactnet"]["embedding"]["base_url"] = "https://openrouter.ai/api/v1/"
    path = tmp_path / ".artnet" / "config.yaml"
    path.parent.mkdir()
    existing = {"schema": "custom-schema.yaml", "neo4j": {"maxConnectionPoolSize": 10}}
    if endpoint:
        existing["llm"] = {
            "base_url": endpoint, "api_key_env": "OPENROUTER_API_KEY", "timeout_ms": 45000,
        }
        if endpoint == "opencode":
            existing["llm"] = {"provider": "opencode", "model": "old/model", "timeout_ms": 45000}
    path.write_text(yaml.safe_dump(existing))
    graph_evidence = path.parent / "schema.yaml"
    graph_evidence.write_text("retained schema content")

    class LocalRuntime:
        def __init__(self):
            self.writes = 0

        def exec(self, argv, *, input=None):
            if argv[0] == "cat":
                return (tmp_path / argv[1]).read_text()
            assert argv[:2] == ["python3", "-c"]
            self.writes += 1
            return subprocess.run(
                [sys.executable, *argv[1:]], cwd=tmp_path, input=input, text=True, capture_output=True, check=True,
            ).stdout

    runtime = LocalRuntime()
    provider = ArtifactNetProvider(config, runtime)
    provider.prepare_runtime_config()
    assert yaml.safe_load(path.read_text())["llm"]["base_url"] == "http://api:8080/unassigned/llm/v1"
    provider.prepare_runtime_config("m1")
    provider.prepare_runtime_config("m1")
    result = yaml.safe_load(path.read_text())
    assert result["schema"] == "custom-schema.yaml"
    assert result["neo4j"]["uri"] == "bolt://neo4j:7687"
    assert result["neo4j"]["maxConnectionPoolSize"] == 10
    assert result["llm"]["provider"] == "api"
    for role in ("llm", "embedding"):
        assert result[role]["base_url"] == f"http://api:8080/m/m1/{role}/v1"
        assert result[role]["api_key_env"] == "WORKFLOW_API_TOKEN"
        assert result[role]["model"] == config["artifactnet"][role]["model"]
    if endpoint:
        assert result["llm"]["timeout_ms"] == 45000
    assert graph_evidence.read_text() == "retained schema content"
    assert runtime.writes == 2
    assert not path.with_suffix(".yaml.tmp").exists()
    provider.prepare_runtime_config("m2")
    assert yaml.safe_load(path.read_text())["embedding"]["base_url"] == "http://api:8080/m/m2/embedding/v1"
    with pytest.raises(ValueError, match="milestone identifier"):
        provider.prepare_runtime_config("../m3")


def test_services_upgrade_preserves_graph_identity_and_adds_proxy(tmp_path, monkeypatch):
    config = load(CONFIGS / "openspec_artifactnet.yaml")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")
    config["artifactnet"]["embedding"]["api_key_env"] = "EMBEDDING_FIXTURE_KEY"
    monkeypatch.setenv("EMBEDDING_FIXTURE_KEY", "embedding-secret")
    identity = {"project": "retained-project", "neo4j_image_id": "sha256:graph", "tool_image_id": "sha256:tools"}
    (tmp_path / "runtime.json").write_text(json.dumps(identity))
    (tmp_path / "compose.resolved.yaml").write_text("services:\n  api:\n    image: obsolete\n")
    runtime = WorkflowRuntime(config, tmp_path, "fixture-agent")
    calls = []
    monkeypatch.setattr(runtime, "_compose", lambda *args: calls.append(args))
    runtime._prepare_services()
    compose = yaml.safe_load((tmp_path / "compose.resolved.yaml").read_text())
    assert set(compose["services"]) == {"neo4j", "api"}
    assert compose["services"]["neo4j"]["image"] == "sha256:graph"
    assert compose["services"]["neo4j"]["volumes"] == ["neo4j-data:/data", "neo4j-logs:/logs"]
    assert runtime.meta == identity
    assert calls[0][0] == "up" and "--remove-orphans" in calls[0]
    assert "fixture-secret" not in (tmp_path / "compose.resolved.yaml").read_text()
    proxy = compose["services"]["api"]
    assert proxy["image"] == "sha256:tools"
    assert proxy["environment"] == {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY:?required}",
        "EMBEDDING_FIXTURE_KEY": "${EMBEDDING_FIXTURE_KEY:?required}",
    }
    assert "ports" not in proxy
    assert any(mount.endswith("usage_proxy.mjs:/proxy.mjs:ro") for mount in proxy["volumes"])
    data = json.loads((tmp_path / "proxy.json").read_text())
    assert data["trial_id"] == "retained-project"
    assert data["trial_name"] == tmp_path.parent.name
    assert data["usage_file"] == "/usage/requests.jsonl"
    assert data["upstreams"]["embedding"] == config["artifactnet"]["embedding"]
    assert "fixture-secret" not in json.dumps(data)
    assert "embedding-secret" not in json.dumps(compose)


@pytest.mark.parametrize("key_value", [None, ""])
def test_legacy_compose_cleanup_works_without_api_credentials(tmp_path, monkeypatch, key_value):
    if key_value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key_value)
    runtime = WorkflowRuntime(load(CONFIGS / "openspec_artifactnet.yaml"), tmp_path, "fixture-agent")
    (tmp_path / "compose.resolved.yaml").write_text(
        'services:\n  api:\n    environment:\n      OPENROUTER_API_KEY: "${OPENROUTER_API_KEY:?required}"\n'
    )
    calls = []

    def command(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return ""

    monkeypatch.setattr(runtime_module, "command", command)
    runtime._compose("down", "-v", "--remove-orphans")
    argv, environment = calls[0]
    assert environment["OPENROUTER_API_KEY"] == "unused-for-cleanup"
    assert argv[-3:] == ["down", "-v", "--remove-orphans"]


def test_service_firewall_refreshes_only_trial_service_rules(tmp_path, monkeypatch):
    runtime = WorkflowRuntime(load(CONFIGS / "openspec_artifactnet.yaml"), tmp_path, "fixture-agent")
    network = runtime.meta["project"] + "_default"
    monkeypatch.setattr(runtime, "connect", lambda: None)
    monkeypatch.setattr(runtime, "verify_services", lambda: None)
    monkeypatch.setattr(runtime, "_compose", lambda *args: args[-1])
    scripts = []

    def command(argv, **kwargs):
        if argv[1] == "inspect":
            ip = "172.30.0.2" if argv[2] == "api" else "172.30.0.3"
            return json.dumps({network: {"IPAddress": ip}})
        scripts.append(argv[-1])
        return ""

    monkeypatch.setattr(runtime_module, "command", command)
    runtime.allow_services()
    fixture = '''
iptables() {
    if [ "$1" = "-S" ]; then
        printf '%s\n' \
          '-P OUTPUT DROP' \
          '-A OUTPUT -d 172.30.0.8 -p tcp --dport 8080 -m comment --comment workflow-api -j ACCEPT' \
          '-A OUTPUT -d 172.30.0.9 -p tcp --dport 7687 -m comment --comment workflow-neo4j -j ACCEPT' \
          '-A OUTPUT -d 10.0.0.0/8 -j ACCEPT'
    else
        printf '%s\n' "$*"
    fi
}
'''
    result = subprocess.run(["sh", "-ec", fixture + scripts[0]], capture_output=True, text=True, check=True).stdout
    assert "-D OUTPUT -d 172.30.0.8" in result
    assert "-D OUTPUT -d 172.30.0.9" in result
    assert "-I OUTPUT 1 -p tcp -d 172.30.0.3 --dport 7687" in result
    assert "-I OUTPUT 1 -p tcp -d 172.30.0.2 --dport 8080" in result
    assert "10.0.0.0/8" not in result
