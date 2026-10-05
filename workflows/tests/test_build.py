"""Build the native ArtifactNet CLI before assembling the isolated tool image."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import build


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    repository = tmp_path / "SWE-Milestone"
    module = repository / "workflows" / "build.py"
    shutil.copytree(Path(build.__file__).parent / "docker", module.parent / "docker")
    module.write_text("# Fixture project layout.\n")
    (repository / ".env_private").write_text("FAKE_KEY=do-not-send-to-docker\n")
    (repository / "benchmark-answer.txt").write_text("do not include benchmark data\n")
    monkeypatch.setattr(build, "__file__", str(module))
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return SimpleNamespace(repository=repository, source=tmp_path / "artnet", docker=module.parent / "docker")


def native_source(path, version="1.4.3"):
    path.mkdir(parents=True)
    (path / "Dockerfile").write_text("FROM node:22 AS runtime-base\n")
    (path / "package.json").write_text(json.dumps({"name": "artnet", "version": version}))
    return path


@pytest.fixture
def builds(monkeypatch):
    captured = []

    def run(argv, **kwargs):
        assert argv[:2] == ["docker", "build"]
        assert kwargs.get("check") is True
        context = Path(argv[-1])
        captured.append(SimpleNamespace(
            argv=argv,
            context=context,
            files={p.relative_to(context).as_posix(): p.read_bytes() for p in context.rglob("*") if p.is_file()},
        ))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(build.subprocess, "run", run)
    return captured


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["workflows.build", *args])
    build.main()


def flag(command, name):
    return command[command.index(name) + 1]


def tags(command):
    return [command[i + 1] for i, arg in enumerate(command) if arg == "-t"]


def test_default_openspec_build_is_pinned(workspace, builds, monkeypatch):
    invoke(monkeypatch, "--provider", "openspec")
    assert len(builds) == 1
    assert tags(builds[0].argv) == ["swe-workflow-openspec:1.7.0"]
    assert "OPENSPEC_VERSION=1.7.0" in builds[0].argv


def test_openspec_needs_no_artnet_checkout_and_uses_versioned_default_tag(workspace, builds, monkeypatch):
    assert not workspace.source.exists()
    invoke(monkeypatch, "--provider", "openspec", "--openspec-version", "1.8.2")
    assert len(builds) == 1
    assert flag(builds[0].argv, "--target") == "openspec"
    assert flag(builds[0].argv, "-t") == "swe-workflow-openspec:1.8.2"
    assert "OPENSPEC_VERSION=1.8.2" in builds[0].argv
    assert "NODE_IMAGE=node:22.21.1-bookworm-slim" in builds[0].argv
    assert all("slop-artnet" not in arg for arg in builds[0].argv)


def test_artnet_default_checkout_is_relative_to_project_not_cwd(workspace, builds, monkeypatch):
    native_source(workspace.source, "2.3.4")
    invoke(monkeypatch, "--provider", "openspec_artifactnet")
    assert len(builds) == 2
    native, tools = builds
    assert native.context == workspace.source
    assert flag(native.argv, "--target") == "runtime-base"
    assert flag(native.argv, "-t") == "swe-workflow-artnet-cli:2.3.4"
    assert flag(tools.argv, "--target") == "artifactnet"
    assert flag(tools.argv, "-t") == "swe-workflow-artifactnet:2.3.4"
    assert "ARTNET_IMAGE=swe-workflow-artnet-cli:2.3.4" in tools.argv
    assert "OPENSPEC_VERSION=1.7.0" in tools.argv
    assert tags(native.argv) == ["swe-workflow-artnet-cli:2.3.4", "swe-workflow-artnet-cli:latest"]
    assert tags(tools.argv) == ["swe-workflow-artifactnet:2.3.4", "swe-workflow-artifactnet:latest"]
    assert all("slop-artnet" not in arg for call in builds for arg in call.argv)


def test_custom_source_tags_node_and_openspec_version_are_forwarded(workspace, builds, monkeypatch, tmp_path):
    source = native_source(tmp_path / "custom artnet checkout", "3.2.1")
    invoke(
        monkeypatch, "--provider", "openspec_artifactnet", "--artnet-source", str(source),
        "--artnet-image", "local/artnet:custom", "--node-image", "local/node:custom",
        "--openspec-version", "1.9.3", "--tag", "local/workflow:custom",
    )
    assert len(builds) == 2
    assert builds[0].context == source
    assert flag(builds[0].argv, "-t") == "local/artnet:custom"
    assert flag(builds[1].argv, "-t") == "local/workflow:custom"
    assert "ARTNET_IMAGE=local/artnet:custom" in builds[1].argv
    assert "NODE_IMAGE=local/node:custom" in builds[1].argv
    assert "OPENSPEC_VERSION=1.9.3" in builds[1].argv
    assert tags(builds[0].argv) == ["local/artnet:custom"]
    assert tags(builds[1].argv) == ["local/workflow:custom"]


@pytest.mark.parametrize("missing", ["checkout", "Dockerfile", "package.json"])
def test_missing_native_sources_stop_before_any_docker_build(workspace, builds, monkeypatch, missing):
    if missing != "checkout":
        native_source(workspace.source)
        (workspace.source / missing).unlink()
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        invoke(monkeypatch, "--provider", "openspec_artifactnet")
    assert builds == []


@pytest.mark.parametrize("version", ["latest", "1.4", None, 143, True])
def test_invalid_package_version_stops_before_any_docker_build(workspace, builds, monkeypatch, version):
    native_source(workspace.source, version)
    with pytest.raises((ValueError, SystemExit)):
        invoke(monkeypatch, "--provider", "openspec_artifactnet")
    assert builds == []


def test_malformed_package_json_stops_before_any_docker_build(workspace, builds, monkeypatch):
    native_source(workspace.source)
    (workspace.source / "package.json").write_text('{"version":')
    with pytest.raises((ValueError, SystemExit)):
        invoke(monkeypatch, "--provider", "openspec_artifactnet")
    assert builds == []


def test_failed_native_build_does_not_start_tool_image_build(workspace, monkeypatch):
    native_source(workspace.source)
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        assert kwargs.get("check") is True
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(build.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        invoke(monkeypatch, "--provider", "openspec_artifactnet")
    assert len(calls) == 1
    assert flag(calls[0], "--target") == "runtime-base"
    assert Path(calls[0][-1]) == workspace.source


@pytest.mark.parametrize("provider", ["openspec", "openspec_artifactnet"])
def test_tool_build_context_contains_only_workflow_docker_files(workspace, builds, monkeypatch, provider):
    if provider == "openspec_artifactnet":
        native_source(workspace.source)
    invoke(monkeypatch, "--provider", provider)
    tools = builds[-1]
    expected = {
        "workflows/docker/" + p.relative_to(workspace.docker).as_posix(): p.read_bytes()
        for p in workspace.docker.rglob("*") if p.is_file()
    }
    assert tools.files == expected
    assert "workflows/docker/Dockerfile" in tools.files
    assert all(b"do-not-send-to-docker" not in content for content in tools.files.values())
    assert not tools.context.exists()  # Temporary build context is cleaned after success.
