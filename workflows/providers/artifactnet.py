from __future__ import annotations

import re
from typing import ClassVar
from urllib.parse import urlsplit

import yaml

from .openspec import OpenSpecProvider


class ArtifactNetProvider(OpenSpecProvider):
    config_keys = ("artifactnet",)
    default_versions: ClassVar[dict[str, str]] = {"openspec": "1.7.0", "artifactnet": "latest"}

    @classmethod
    def validate_config(cls, config: dict) -> None:
        from ..config import _keys

        super().validate_config(config)
        artnet = config.get("artifactnet", {})
        _keys(artnet, {"neo4j_image", "wait_seconds", "llm", "embedding"}, "artifactnet")
        artnet.setdefault("neo4j_image", "neo4j:5-community")
        artnet.setdefault("wait_seconds", 600)
        if not isinstance(artnet["wait_seconds"], int) or artnet["wait_seconds"] <= 0:
            raise ValueError("artifactnet.wait_seconds must be positive")
        for role in ("llm", "embedding"):
            spec = artnet.get(role, {})
            _keys(spec, {"base_url", "model", "api_key_env"}, f"artifactnet.{role}")
            url = urlsplit(spec.get("base_url", ""))
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"artifactnet.{role}.base_url must be an HTTPS API URL without credentials")
            if not isinstance(spec.get("model"), str) or not spec["model"].strip():
                raise ValueError(f"artifactnet.{role}.model is required")
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", spec.get("api_key_env", "")):
                raise ValueError(f"artifactnet.{role}.api_key_env must name a host environment variable")
        config["artifactnet"] = artnet

    def initialize(self) -> None:
        super().initialize()
        self.runtime.exec(["artnet", "init", "--adapter", "openspec"])

    def prepare_runtime_config(self, milestone: str | None = None) -> None:
        """Bind API calls to this milestone without changing the graph or schema."""
        if milestone is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", milestone):
            raise ValueError(f"Unsupported milestone identifier: {milestone!r}")
        prefix = f"m/{milestone}" if milestone is not None else "unassigned"
        config = yaml.safe_load(self.runtime.exec(["cat", ".artnet/config.yaml"])) or {}
        if not isinstance(config, dict):
            raise ValueError("ArtifactNet runtime configuration must be a mapping")  # noqa: TRY004 - config error
        original = yaml.safe_dump(config)
        config.setdefault("schema", ".artnet/schema.yaml")
        config["neo4j"] = {
            **config.get("neo4j", {}),
            "uri": "bolt://neo4j:7687",
            "user": "neo4j",
            "password": "workflow-local",
            "database": "neo4j",
        }
        for role in ("llm", "embedding"):
            config[role] = {
                **config.get(role, {}),
                **self.config["artifactnet"][role],
                "base_url": f"http://api:8080/{prefix}/{role}/v1",
                "api_key_env": "WORKFLOW_API_TOKEN",
            }
        config["llm"]["provider"] = "api"
        updated = yaml.safe_dump(config)
        if updated == original:
            return
        self.runtime.exec(
            [
                "python3", "-c",
                (
                    "import pathlib,sys; p=pathlib.Path('.artnet/config.yaml'); "
                    "tmp=p.with_suffix('.yaml.tmp'); tmp.write_text(sys.stdin.read()); tmp.replace(p)"
                ),
            ],
            input=updated,
        )

    def verify_tools(self) -> dict:
        versions = super().verify_tools()
        actual = self.runtime.exec(["node", "-p", "require('/opt/workflow/artnet/package.json').version"]).strip()
        if not actual:
            raise RuntimeError("ArtifactNet version command returned an empty version")
        requested = self.config["versions"]["artifactnet"]
        if requested != "latest" and actual != requested:
            raise RuntimeError(f"ArtifactNet version mismatch: {actual}")
        versions["artifactnet"] = actual
        return versions
