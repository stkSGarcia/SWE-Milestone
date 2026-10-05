from __future__ import annotations

import re
from typing import ClassVar


class OpenSpecProvider:
    config_keys = ()
    default_versions: ClassVar[dict[str, str]] = {"openspec": "1.7.0"}

    @classmethod
    def validate_config(cls, config: dict) -> None:
        for name in cls.default_versions:
            if name not in config["versions"]:
                raise ValueError(f"versions.{name} is required")

    def __init__(self, config: dict, runtime):
        self.config, self.runtime = config, runtime

    def stage_spec(self, name: str) -> dict:
        for stage in self.config["stages"]:
            if stage["name"] == name:
                return stage
        raise ValueError(f"Workflow stage is not configured: {name}")

    def initialize(self) -> None:
        self.runtime.exec(["openspec", "config", "set", "delivery", "both"])
        self.runtime.exec(
            ["openspec", "init", ".", "--tools", "codex", "--profile", "core", "--force", "--no-animation"]
        )

    def verify_tools(self) -> dict:
        actual = self.runtime.exec(["openspec", "--version"]).strip()
        if not actual:
            raise RuntimeError("OpenSpec version command returned an empty version")
        requested = self.config["versions"]["openspec"]
        if requested != "latest" and actual != requested:
            raise RuntimeError(f"OpenSpec version mismatch: {actual}")
        for stage in self.config["stages"]:
            self.runtime.exec(["test", "-s", f".codex/skills/{stage['name']}/SKILL.md"])
        return {"openspec": actual, "node": self.runtime.exec(["node", "--version"]).strip()}

    def prepare_runtime_config(self, milestone: str | None = None) -> None:
        """Refresh runtime settings on both fresh runs and resume, when needed."""

    def prompt(self, stage: str, mid: str, change: str, task: str, source_dirs: list[str]) -> str:
        spec = self.stage_spec(stage)
        values = {"task": task, "change_id": change, "milestone": mid}
        purpose = "\n\n".join(
            re.sub(r"\{(task|change_id|milestone)\}", lambda match: values[match[1]], argument)
            for argument in spec.get("arguments", [])
        )
        return (
            f"${stage}\n\n{purpose}\n\n"
            f"This is milestone {mid}. Use change ID exactly {change}.\n"
            "Complete only the current skill; do not start later stages.\n"
            "Preserve earlier milestone implementations.\n"
            "Do not create, move or delete agent-impl-* tags; the controller handles submission.\n"
            "Run unattended without asking for confirmation.\n"
        )
