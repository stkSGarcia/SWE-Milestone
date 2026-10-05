"""OpenSpec is pinned; ArtifactNet can follow locally built versions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.config import digest, freeze, load, restore, stage_names, validate
from workflows.providers.artifactnet import ArtifactNetProvider
from workflows.providers.openspec import OpenSpecProvider

CONFIGS = Path(__file__).parents[1] / "configs"
PROVIDERS = [
    ("openspec", "openspec", OpenSpecProvider),
    ("openspec_artifactnet", "artifactnet", ArtifactNetProvider),
]


@pytest.mark.parametrize("profile,tool,provider_type", PROVIDERS)
@pytest.mark.parametrize("requested,actual,error", [
    ("latest", "3.2.1", None),
    ("3.2.1", "3.2.1", None),
    ("3.2.1", "3.2.2", "version mismatch"),
    ("latest", " \n", "empty version"),
    ("3.2.1", " \n", "empty version"),
])
def test_verify_reports_actual_versions_and_enforces_pins(profile, tool, provider_type, requested, actual, error):
    config = load(CONFIGS / f"{profile}.yaml")
    config["versions"][tool] = requested
    calls = []

    def execute(argv):
        calls.append(argv)
        if argv == ["openspec", "--version"]:
            return actual if tool == "openspec" else "1.7.0\n"
        if argv == ["node", "-p", "require('/opt/workflow/artnet/package.json').version"]:
            return actual
        if argv == ["node", "--version"]:
            return "v22.21.1\n"
        assert argv[:2] == ["test", "-s"]
        return ""

    provider = provider_type(config, SimpleNamespace(exec=execute))
    if error:
        with pytest.raises(RuntimeError, match=error):
            provider.verify_tools()
    else:
        versions = provider.verify_tools()
        assert versions[tool] == actual.strip()
        assert versions["node"] == "v22.21.1"
        assert all(["test", "-s", f".codex/skills/{stage}/SKILL.md"] in calls for stage in stage_names(config))


@pytest.mark.parametrize("profile,tool,provider_type", PROVIDERS)
def test_default_profiles_pin_openspec_and_observe_artifactnet(profile, tool, provider_type):
    config = load(CONFIGS / f"{profile}.yaml")
    tag = "1.7.0" if tool == "openspec" else "latest"
    assert config["runtime_image"] == f"swe-workflow-{tool}:{tag}"
    assert config["versions"]["openspec"] == "1.7.0"
    if tool == "artifactnet":
        assert config["versions"]["artifactnet"] == "latest"
    expected = config.pop("versions")
    assert validate(config)["versions"] == expected


@pytest.mark.parametrize("profile,tool,provider_type", PROVIDERS)
def test_configuration_freezes_and_restores_without_resolving_tags(tmp_path, profile, tool, provider_type):
    source = CONFIGS / f"{profile}.yaml"
    binding = freeze(source, tmp_path, "codex")
    assert restore(binding, tmp_path, source) == load(source)
    assert restore(binding, tmp_path)["versions"]["openspec"] == "1.7.0"


@pytest.mark.parametrize("profile,tool,provider_type", PROVIDERS)
@pytest.mark.parametrize("openspec_version", ["1.7.0", "latest"])
def test_explicit_version_binding_restores_unchanged(tmp_path, profile, tool, provider_type, openspec_version):
    config = load(CONFIGS / f"{profile}.yaml")
    config["versions"] = {"openspec": openspec_version}
    if tool == "artifactnet":
        config["versions"]["artifactnet"] = "1.4.3"
    config["runtime_image"] = f"swe-workflow-{tool}:{config['versions'][tool]}"
    path = tmp_path / "workflow" / "config.resolved.json"
    path.parent.mkdir()
    original = json.dumps(config)
    path.write_text(original)
    binding = {"path": "workflow/config.resolved.json", "sha256": digest(config), "provider": profile}

    assert restore(binding, tmp_path) == config
    assert path.read_text() == original
    changed = load(CONFIGS / f"{profile}.yaml")
    changed["versions"]["openspec"] = "9.9.9"
    requested = tmp_path / "changed.yaml"
    requested.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs"):
        restore(binding, tmp_path, requested)
