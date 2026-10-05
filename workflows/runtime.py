"""Tool bundle and per-trial services. No workflow logic lives in the harness."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import yaml

from .state import atomic_json


def command(argv: list[str], *, env=None, timeout: int = 120, input: str | None = None) -> str:
    result = subprocess.run(argv, input=input, text=True, capture_output=True, timeout=timeout, env=env, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        for key, value in (env or os.environ).items():
            if value and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
                detail = detail.replace(value, "[redacted]")
        raise RuntimeError(
            f"Workflow command {argv[0]} {argv[1] if len(argv) > 1 else ''} "
            f"failed (exit {result.returncode}): {detail[-4000:]}"
        )
    return result.stdout


def bundle_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            h.update(f"link:{relative}:{os.readlink(path)}\0".encode())
        elif path.is_file():
            h.update(f"file:{relative}:{path.stat().st_mode & 0o777}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
    return h.hexdigest()


class WorkflowRuntime:
    def __init__(self, config: dict, root: Path, container: str):
        self.config, self.root, self.container = config, root, container
        self.meta_path = root / "runtime.json"
        self.meta = (
            json.loads(self.meta_path.read_text())
            if self.meta_path.exists()
            else {
                "schema_version": 1,
                "project": f"swe-workflow-{uuid.uuid4().hex[:16]}",
            }
        )
        self.tools = root / "tools"
        self.is_artnet = config["provider"] == "openspec_artifactnet"
        self.services_started = False

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        image = (
            self.meta.get("tool_image_id")
            or command(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.config["runtime_image"],
                    "--format",
                    "{{.Id}}",
                ]
            ).strip()
        )
        command(["docker", "image", "inspect", image])
        self.meta["tool_image_id"] = image
        self.meta["agent_container"] = self.container
        atomic_json(self.meta_path, self.meta)
        if not self.tools.exists():
            scratch = f"{self.meta['project']}-tool-export"
            staging = self.root / "tools.partial"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            subprocess.run(["docker", "rm", "-f", scratch], capture_output=True, timeout=30, check=False)
            command(["docker", "create", "--name", scratch, "--network", "none", image, "true"])
            try:
                command(["docker", "cp", f"{scratch}:/opt/workflow/.", str(staging)], timeout=300)
                staging.rename(self.tools)
            finally:
                command(["docker", "rm", "-f", scratch])
        actual = bundle_digest(self.tools)
        if self.meta.get("tool_bundle_sha256", actual) != actual:
            raise RuntimeError("Workflow tool bundle changed since trial initialization")
        if not (self.tools / "bin" / "node").is_file():
            raise RuntimeError("Incomplete workflow tool bundle; rebuild the tool image")
        self.meta["tool_bundle_sha256"] = actual
        atomic_json(self.meta_path, self.meta)
        if self.is_artnet:
            self._prepare_services()

    def mounts(self) -> list[str]:
        return ["-v", f"{self.tools.resolve()}:/opt/workflow:ro"]

    def api_key_env_args(self) -> list[str]:
        """ArtifactNet uses a local placeholder; upstream keys stay in the proxy."""
        return ["-e", "WORKFLOW_API_TOKEN=local"] if self.is_artnet else []

    def install_cli_launchers(self) -> None:
        """Keep workflow CLIs reachable after login/Go shells reset PATH."""
        tools = ["openspec", "artnet"] if self.is_artnet else ["openspec"]
        script = f'''
import os
from pathlib import Path

tools = {tools!r}
for tool in ['node', *tools]:
    if not os.access('/opt/workflow/bin/' + tool, os.X_OK):
        raise RuntimeError('Missing workflow executable: ' + tool)
directory = Path('/usr/local/bin')
directory.mkdir(parents=True, exist_ok=True)
for tool in tools:
    target = directory / tool
    if target.is_symlink():
        target.unlink()
    target.write_text('#!/bin/sh\\nexport PATH=/opt/workflow/bin:$PATH\\n'
                      + 'exec /opt/workflow/bin/' + tool + ' "$@"\\n')
    os.chown(target, 0, 0)
    os.chmod(target, 0o755)
'''
        command(["docker", "exec", "--user", "root", self.container, "python3", "-c", script])

    def _compose(self, *args: str) -> str:
        env = os.environ.copy()
        if args and args[0] == "down":
            # Compose interpolates service keys even when only removing services.
            for role in ("llm", "embedding"):
                name = self.config["artifactnet"][role]["api_key_env"]
                env[name] = env.get(name) or "unused-for-cleanup"
        return command(
            [
                "docker",
                "compose",
                "-f",
                str(self.root / "compose.resolved.yaml"),
                "-p",
                self.meta["project"],
                *args,
            ],
            env=env,
            timeout=self.config["artifactnet"]["wait_seconds"] + 120,
        )

    def _prepare_services(self) -> None:
        cfg = self.config["artifactnet"]
        key_names = {cfg[role]["api_key_env"] for role in ("llm", "embedding")}
        for name in sorted(key_names):
            if not os.environ.get(name):
                raise ValueError(f"Missing ArtifactNet host environment variable: {name}")
        neo4j = (
            self.meta.get("neo4j_image_id")
            or command(
                [
                    "docker",
                    "image",
                    "inspect",
                    cfg["neo4j_image"],
                    "--format",
                    "{{.Id}}",
                ]
            ).strip()
        )
        self.meta["neo4j_image_id"] = neo4j
        atomic_json(self.meta_path, self.meta)
        usage = self.root / "usage"
        usage.mkdir(exist_ok=True)
        proxy_config = self.root / "proxy.json"
        atomic_json(proxy_config, {
            "trial_id": self.meta["project"],
            "trial_name": self.root.parent.name,
            "port": 8080,
            "usage_file": "/usage/requests.jsonl",
            "upstreams": {role: cfg[role] for role in ("llm", "embedding")},
        })
        proxy_script = Path(__file__).parent / "docker" / "usage_proxy.mjs"
        compose = {
            "services": {
                "api": {
                    "image": self.meta["tool_image_id"],
                    "pull_policy": "never",
                    "init": True,
                    "entrypoint": ["/opt/workflow/bin/node"],
                    "command": ["/proxy.mjs", "/proxy.json"],
                    "environment": {name: "${" + name + ":?required}" for name in sorted(key_names)},
                    "volumes": [
                        f"{proxy_script.resolve()}:/proxy.mjs:ro",
                        f"{proxy_config.resolve()}:/proxy.json:ro",
                        f"{usage.resolve()}:/usage",
                    ],
                    "stop_grace_period": "15s",
                    "healthcheck": {
                        "test": [
                            "CMD", "/opt/workflow/bin/node", "-e",
                            ("fetch('http://127.0.0.1:8080/health').then(r=>process.exit(r.ok?0:1))"
                             ".catch(()=>process.exit(1))"),
                        ],
                        "interval": "5s",
                        "timeout": "5s",
                        "retries": 10,
                    },
                },
                "neo4j": {
                    "image": neo4j,
                    "pull_policy": "never",
                    "environment": {"NEO4J_AUTH": "neo4j/workflow-local"},
                    "volumes": ["neo4j-data:/data", "neo4j-logs:/logs"],
                    "healthcheck": {
                        "test": ["CMD-SHELL", "cypher-shell -u neo4j -p workflow-local 'RETURN 1' >/dev/null 2>&1"],
                        "interval": "5s",
                        "timeout": "30s",
                        "retries": 30,
                        "start_period": "30s",
                    },
                },
            },
            "volumes": {"neo4j-data": {}, "neo4j-logs": {}},
        }
        (self.root / "compose.resolved.yaml").write_text(yaml.safe_dump(compose))
        # Set before up so cleanup also handles a partially successful start.
        self.services_started = True
        self._compose("up", "-d", "--remove-orphans", "--wait", "--wait-timeout", str(cfg["wait_seconds"]))

    def connect(self) -> None:
        if not self.is_artnet:
            return
        network = f"{self.meta['project']}_default"
        networks = json.loads(
            command(["docker", "inspect", self.container, "--format", "{{json .NetworkSettings.Networks}}"])
        )
        if network not in networks:
            command(["docker", "network", "connect", network, self.container])

    def allow_services(self) -> None:
        """Allow only fixed local service IP/ports, preserving existing CIDR rules."""
        if not self.is_artnet:
            return
        self.connect()
        network = f"{self.meta['project']}_default"
        # Service addresses can change when a trial resumes.
        script = (
            "iptables -S OUTPUT | "
            "awk '/--comment workflow-(api|neo4j)( |$)/ {sub(/^-A/, \"-D\"); print}' | "
            "while IFS= read -r rule; do iptables $rule; done\n"
        )
        for service, port in (("api", 8080), ("neo4j", 7687)):
            cid = self._compose("ps", "-q", service).strip()
            networks = json.loads(command(["docker", "inspect", cid, "--format", "{{json .NetworkSettings.Networks}}"]))
            ip = str(ipaddress.ip_address(networks[network]["IPAddress"]))
            script += (
                f"iptables -I OUTPUT 1 -p tcp -d {ip} --dport {port} "
                f"-m comment --comment workflow-{service} -j ACCEPT\n"
            )
        command(["docker", "exec", self.container, "sh", "-ec", script])
        self.verify_services()

    def verify_services(self) -> None:
        if self.is_artnet:
            self.exec(
                [
                    "node",
                    "-e",
                    (
                        "fetch('http://api:8080/health',{signal:AbortSignal.timeout(10000)})"
                        ".then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1));"
                        "const s=require('net').connect(7687,'neo4j',()=>s.end());"
                        "s.setTimeout(10000,()=>{s.destroy();process.exit(1)});s.on('error',()=>process.exit(1))"
                    ),
                ]
            )

    def exec(self, argv: list[str], *, input: str | None = None, timeout: int = 120) -> str:
        return command(
            [
                "docker",
                "exec",
                "-i",
                "--user",
                "fakeroot",
                "-e",
                "HOME=/home/fakeroot",
                "-e",
                "CI=true",
                "-e",
                "OPENSPEC_TELEMETRY=0",
                "-e",
                "NO_COLOR=1",
                *self.api_key_env_args(),
                "-w",
                "/testbed",
                self.container,
                "sh",
                "-ec",
                "export PATH=/opt/workflow/bin:$PATH; exec " + shlex.join(argv),
            ],
            input=input,
            timeout=timeout,
        )

    def close(self) -> None:
        if self.services_started:
            # Disconnect retained agent before down, otherwise the network stays busy.
            subprocess.run(
                ["docker", "network", "disconnect", f"{self.meta['project']}_default", self.container],
                capture_output=True,
                timeout=30,
                check=False,
            )
            self._compose("down", "--remove-orphans")  # deliberately retain volumes for resume
            self.services_started = False


def discard_trial_runtime(trial_root: Path) -> None:
    """Used only by the existing --force path, before it removes trial metadata."""
    root = trial_root / "workflow"
    if not (root / "compose.resolved.yaml").exists():
        return
    from .config import restore

    metadata = json.loads((trial_root / "trial_metadata.json").read_text())
    config = restore(metadata.get("workflow"), trial_root)
    if config is None:
        raise RuntimeError("Cannot identify workflow resources for forced restart")
    runtime_metadata = json.loads((root / "runtime.json").read_text())
    runtime = WorkflowRuntime(config, root, runtime_metadata["agent_container"])
    runtime.services_started = True
    runtime.close()
    runtime._compose("down", "-v", "--remove-orphans")
