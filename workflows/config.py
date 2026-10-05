"""Validated, trial-frozen workflow configuration (no additional dependencies)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .providers import get_provider_class
from .state import atomic_json

SCHEMA_VERSION = 1
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _keys(value: dict, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")  # noqa: TRY004 - uniform config errors
    if extra := value.keys() - allowed:
        raise ValueError(f"Unknown {label} keys: {sorted(extra)}")


def stage_names(config: dict) -> list[str]:
    """Stage names are the keys used in state, logs and skill invocations."""
    return [stage["name"] for stage in config["stages"]]


def validate(value: dict) -> dict:
    """Return a normalized configuration; secrets are referenced by env name only."""
    value = json.loads(json.dumps(value))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected workflow schema_version: {SCHEMA_VERSION}")
    provider = get_provider_class(value.get("provider"))
    _keys(
        value,
        {"schema_version", "provider", "stages", "runtime_image", "versions", "budgets", *provider.config_keys},
        "workflow",
    )
    image = value.get("runtime_image", "")
    if not isinstance(image, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_./:@-]+", image):
        raise ValueError("runtime_image must name a prebuilt local tool image")
    stages = value.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a nonempty list of skills")
    for stage in stages:
        _keys(stage, {"name", "arguments"}, "stage")
        name = stage.get("name")
        arguments = stage.get("arguments", [])
        if not isinstance(arguments, list) or any(not isinstance(a, str) or not a.strip() for a in arguments):
            raise ValueError("stage.arguments must be a list of nonempty strings")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("stage.name must be a lowercase kebab-case skill name")
    names = stage_names(value)
    if len(names) != len(set(names)):
        raise ValueError("stage names must be unique")
    versions = value.setdefault("versions", dict(provider.default_versions))
    _keys(versions, set(provider.default_versions), "versions")
    for name, version in versions.items():
        if not _NAME.fullmatch(name):
            raise ValueError("version keys must be lowercase kebab-case tool names")
        if not isinstance(version, str) or (version != "latest" and not re.fullmatch(r"\d+\.\d+\.\d+", version)):
            raise ValueError(f"versions.{name} must be latest or an exact semantic version")
    budgets = value.setdefault("budgets", {})
    _keys(budgets, {"stage_seconds", "milestone_seconds", "max_attempts"}, "budgets")
    for key, default in (("stage_seconds", 7200), ("milestone_seconds", 28800), ("max_attempts", 2)):
        n = budgets.setdefault(key, default)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"budgets.{key} must be a positive integer")
    provider.validate_config(value)
    return value


def load(path: Path) -> dict:
    return validate(yaml.safe_load(path.read_text()))


def freeze(path: Path | None, trial_root: Path, agent: str) -> dict | None:
    if path is None:
        return None
    if agent != "codex":
        raise ValueError("Skill workflows currently require --agent codex")
    config = load(path)
    target = trial_root / "workflow" / "config.resolved.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Trial already has a different workflow configuration; create a new trial")
    atomic_json(target, config)
    return {"path": "workflow/config.resolved.json", "sha256": digest(config), "provider": config["provider"]}


def restore(binding: dict | None, trial_root: Path, requested: Path | None = None) -> dict | None:
    if binding is None:
        if requested is not None:
            raise ValueError("Cannot add a workflow when resuming a baseline trial")
        return None
    if binding.get("path") != "workflow/config.resolved.json":
        raise ValueError("Invalid workflow configuration binding")
    config = validate(json.loads((trial_root / binding["path"]).read_text()))
    if digest(config) != binding.get("sha256"):
        raise ValueError("Frozen workflow configuration digest mismatch")
    if requested is not None and load(requested) != config:
        raise ValueError("Resume configuration differs from trial-frozen workflow")
    return config
