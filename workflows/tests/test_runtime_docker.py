"""Opt-in real CLI/Neo4j tests; no model credentials or paid inference required."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml

from workflows.config import load
from workflows.providers import create_provider
from workflows.runtime import WorkflowRuntime, command

pytestmark = pytest.mark.skipif(os.environ.get("SWE_WORKFLOW_DOCKER_TESTS") != "1", reason="opt-in Docker integration")
CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture(params=["openspec", "openspec_artifactnet"])
def environment(tmp_path, request, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-unused-key")
    config = load(CONFIGS / f"{request.param}.yaml")
    container = "swe-workflow-test-" + uuid.uuid4().hex[:12]
    runtime = WorkflowRuntime(config, tmp_path / "workflow", container)
    try:
        runtime.prepare()
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--cap-add=NET_ADMIN",
                *runtime.mounts(),
                "swe-workflow-testbed:local",
            ]
        )
        # Match the benchmark's default-deny policy before allowing workflow services.
        command(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-ec",
                "iptables -A OUTPUT -o lo -j ACCEPT; iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT; iptables -P OUTPUT DROP",
            ]
        )
        runtime.allow_services()
        runtime.install_cli_launchers()
        provider = create_provider(config, runtime)
        provider.initialize()
        provider.prepare_runtime_config()
        yield runtime, provider
    finally:
        runtime.close()
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
        if runtime.is_artnet:
            # Only this test's unique resources; production close retains volumes.
            runtime._compose("down", "-v", "--remove-orphans")


def test_cli_survives_login_and_go_shell_path_reset(environment):
    runtime, _provider = environment
    assert runtime.exec(["/bin/bash", "-lc", "openspec --version"]).strip() == "1.7.0"
    script = "export PATH=/usr/local/bin:/usr/bin:/bin\n"
    command(["docker", "exec", "-i", runtime.container, "sh", "-c", "cat > /tmp/workflow-path-reset.sh"], input=script)
    assert runtime.exec(
        ["env", "BASH_ENV=/tmp/workflow-path-reset.sh", "/bin/bash", "-c", "openspec --version"]
    ).strip() == "1.7.0"


def test_real_initialization_and_resume(environment):
    runtime, provider = environment
    versions = provider.verify_tools()
    assert versions["openspec"] == "1.7.0"
    assert json.loads(runtime.exec(["openspec", "list", "--json"]))["changes"] == []
    if runtime.is_artnet:
        assert re.fullmatch(r"\d+\.\d+\.\d+", versions["artifactnet"])
        provider.prepare_runtime_config("m1")
        artnet = yaml.safe_load(runtime.exec(["cat", ".artnet/config.yaml"]))
        assert artnet["llm"]["base_url"] == "http://api:8080/m/m1/llm/v1"
        # An invalid model is rejected locally; no upstream request or charge.
        rejected = runtime.exec([
            "node", "-e",
            ("fetch('http://api:8080/m/m1/llm/v1/chat/completions',"
             "{method:'POST',body:JSON.stringify({model:'invalid-local-fixture'})})"
             ".then(r=>console.log(r.status))"),
        ])
        assert rejected.strip() == "400"
        # Store trial-specific evidence, restart services, and confirm persistence.
        runtime._compose(
            "exec",
            "-T",
            "neo4j",
            "cypher-shell",
            "-u",
            "neo4j",
            "-p",
            "workflow-local",
            "CREATE (:WorkflowSmoke {value: 'preserved'})",
        )
        runtime.close()
        resumed = WorkflowRuntime(runtime.config, runtime.root, runtime.container)
        resumed.prepare()
        try:
            resumed.allow_services()
            assert resumed.meta["project"] == runtime.meta["project"]
            value = resumed._compose(
                "exec",
                "-T",
                "neo4j",
                "cypher-shell",
                "-u",
                "neo4j",
                "-p",
                "workflow-local",
                "MATCH (n:WorkflowSmoke) RETURN n.value",
            )
            assert "preserved" in value
            assert runtime.exec(["sh", "-c", "test -z \"$OPENROUTER_API_KEY\" && printenv WORKFLOW_API_TOKEN"]).strip() == "local"
            assert "fixture-unused-key" not in (runtime.root / "compose.resolved.yaml").read_text()
            assert set(yaml.safe_load((runtime.root / "compose.resolved.yaml").read_text())["services"]) == {"neo4j", "api"}
            assert (runtime.root / "usage" / "requests.jsonl").read_text() == ""
        finally:
            resumed.close()


def test_real_cli_workflow_artifacts(environment):
    runtime, _provider = environment
    change = "smoke-add-value"
    runtime.exec(["openspec", "new", "change", change])
    # Exercise the installed CLIs directly, without workflow-controller validation.
    documents = {
        "proposal.md": "# Why\nAdd a value function.\n\n# What Changes\nAdd value capability.\n\n# Capabilities\n## New Capabilities\n- `value`: Return a value.\n\n# Impact\nSource only.\n",
        "design.md": "# Context\nSmall module.\n\n# Decisions\nReturn one.\n",
        "tasks.md": "## Implementation\n- [x] 1.1 Implement value function\n",
        "specs/value/spec.md": "## ADDED Requirements\n### Requirement: Return value\nThe system SHALL return one.\n\n#### Scenario: Invoke value\n- **WHEN** value is called\n- **THEN** the result is one\n",
    }
    runtime.exec(
        [
            "python3",
            "-c",
            (
                "import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); data=json.load(sys.stdin); "
                "[( (root / p).parent.mkdir(parents=True,exist_ok=True), (root / p).write_text(s)) for p,s in data.items()]"
            ),
            f"openspec/changes/{change}",
        ],
        input=json.dumps(documents),
    )
    status = json.loads(runtime.exec(["openspec", "status", "--change", change, "--json"]))
    assert {artifact["id"] for artifact in status["artifacts"] if artifact["status"] == "done"} == {
        "proposal", "specs", "design", "tasks"
    }
    if not runtime.is_artnet:
        runtime.exec(["openspec", "archive", change, "--yes"])
        runtime.exec(
            [
                "python3",
                "-c",
                (
                    "import pathlib,sys; root=pathlib.Path('openspec/changes'); "
                    "assert not (root/sys.argv[1]).exists(); "
                    "assert len([p for p in (root/'archive').glob('*-'+sys.argv[1]) if p.is_dir()]) == 1"
                ),
                change,
            ]
        )

    else:
        # Real graph publication and retrieval; semantic transport is tested separately.
        graph_only = {
            "schema": ".artnet/schema.yaml",
            "neo4j": {"uri": "bolt://neo4j:7687", "user": "neo4j", "password": "workflow-local", "database": "neo4j"},
        }
        runtime.exec(
            ["python3", "-c", "import pathlib,sys;pathlib.Path('.artnet/config.yaml').write_text(sys.stdin.read())"],
            input=yaml.safe_dump(graph_only),
        )
        documents["proposal.md"] = (
            "## Why\nReturn a value.\n\n## What Changes\nAdd a value function.\n\n## Capabilities\n### New Capabilities\n#### Capability: value\nReturn one.\n\n## Impact\nSource only.\n"
        )
        documents["specs/value/spec.md"] = "# Spec: Value\n\n" + documents["specs/value/spec.md"]
        runtime.exec(
            [
                "python3",
                "-c",
                (
                    "import json,pathlib,sys; r=pathlib.Path(sys.argv[1]); "
                    "[(r/p).write_text(v) for p,v in json.load(sys.stdin).items()]"
                ),
                f"openspec/changes/{change}",
            ],
            input=json.dumps(documents),
        )
        for artifact, path in (
            ("proposal", "proposal.md"),
            ("specs", "specs/value/spec.md"),
            ("design", "design.md"),
            ("tasks", "tasks.md"),
        ):
            context = ["--context", f"change_name={change}"]
            if artifact == "specs":
                context += ["--context", "capability=value"]
            runtime.exec(
                ["artnet", "update", "--change", change, "--artifact", artifact, *context],
                input=documents[path],
            )
        commit = json.loads(
            runtime.exec(["artnet", "update", "--change", change, "--commit", "--no-enrichment", "--no-relations"])
        )
        assert commit["changeFound"] and commit["memberNodes"] > 0
        graph = json.loads(runtime.exec(["artnet", "retrieve", "--change", change, "--view", "graph"]))
        assert graph["nodes"]
        source = json.loads(runtime.exec(["artnet", "retrieve", "--change", change, "--view", "source"]))
        assert {artifact["artifactId"]: artifact["content"] for artifact in source["artifacts"]} == {
            "proposal": documents["proposal.md"],
            "specs": documents["specs/value/spec.md"],
            "design": documents["design.md"],
            "tasks": documents["tasks.md"],
        }


@pytest.mark.parametrize("environment", ["openspec_artifactnet"], indirect=True)
def test_concurrent_trials_have_isolated_graphs(environment, tmp_path):
    first, _provider = environment
    second = WorkflowRuntime(first.config, tmp_path / "other-trial", "swe-workflow-test-" + uuid.uuid4().hex[:12])
    try:
        second.prepare()
        assert first.meta["project"] != second.meta["project"]
        first._compose(
            "exec",
            "-T",
            "neo4j",
            "cypher-shell",
            "-u",
            "neo4j",
            "-p",
            "workflow-local",
            "CREATE (:OnlyFirstTrial {value: 'private'})",
        )
        count = second._compose(
            "exec",
            "-T",
            "neo4j",
            "cypher-shell",
            "-u",
            "neo4j",
            "-p",
            "workflow-local",
            "--format",
            "plain",
            "MATCH (n:OnlyFirstTrial) RETURN count(n) AS count",
        )
        assert count.strip().splitlines()[-1] == "0"
        network = second.meta["project"] + "_default"
        for service, port in (("neo4j", 7687), ("api", 8080)):
            cid = second._compose("ps", "-q", service).strip()
            address = json.loads(command(["docker", "inspect", cid, "--format", "{{json .NetworkSettings.Networks}}"]))[
                network
            ]["IPAddress"]
            assert (
                first.exec(
                    [
                        "python3",
                        "-c",
                        (
                            "import socket,sys;s=socket.socket();s.settimeout(1);"
                            "r=s.connect_ex((sys.argv[1],int(sys.argv[2])));print('blocked' if r else 'reachable')"
                        ),
                        address, str(port),
                    ]
                ).strip()
                == "blocked"
            )
    finally:
        second.close()
        if (second.root / "compose.resolved.yaml").exists():
            second._compose("down", "-v", "--remove-orphans")
