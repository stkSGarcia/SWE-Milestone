"""Focused tests for the Codex agent adapter."""

import os
import pwd
import subprocess
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from harness.e2e.agents.codex import CodexFramework


def test_codex_oauth_reports_chatgpt_network_endpoint(monkeypatch):
    monkeypatch.delenv("UNIFIED_API_KEY", raising=False)
    monkeypatch.delenv("UNIFIED_BASE_URL", raising=False)

    framework = CodexFramework()

    assert framework.get_network_endpoint_url() == framework.OAUTH_ENDPOINT_URL


def test_codex_explicit_base_url_is_network_endpoint(monkeypatch):
    monkeypatch.setenv("UNIFIED_API_KEY", "test-key")
    monkeypatch.setenv("UNIFIED_BASE_URL", "https://proxy.example/v1")

    framework = CodexFramework()

    assert framework.get_network_endpoint_url() == "https://proxy.example/v1"


def test_codex_disables_server_side_product_tools(monkeypatch):
    monkeypatch.delenv("UNIFIED_API_KEY", raising=False)
    monkeypatch.delenv("UNIFIED_BASE_URL", raising=False)
    framework = CodexFramework(reasoning_effort="max")

    command = framework.build_run_command(
        model="gpt-5.6-sol",
        session_id="ignored",
        prompt_path="/tmp/prompt.txt",
    )
    init_script = framework.get_container_init_script("codex")

    for feature in framework.BENCHMARK_DISABLED_FEATURES:
        assert f"features.{feature}=false" in command
        assert repr(feature) in init_script
    assert "__CODEX_DISABLED_FEATURES__" not in init_script
    assert "*[f'{feature} = false'" in init_script
    assert 'web_search="disabled"' in command
    assert 'web_search = "disabled"' in init_script


def test_codex_uses_standalone_installer_without_npm():
    framework = CodexFramework(api_key="test-key", agent_version="0.145.0")

    init_script = framework.get_container_init_script("codex")

    assert "https://chatgpt.com/codex/install.sh" in init_script
    assert "'CODEX_RELEASE': target_version" in init_script
    assert "'CODEX_NON_INTERACTIVE': '1'" in init_script
    assert "'CODEX_HOME': '/opt/codex'" in init_script
    assert "'CODEX_INSTALL_DIR': '/usr/local/bin'" in init_script
    assert "npm', 'i'" not in init_script
    assert "deb.nodesource.com" not in init_script


@pytest.mark.parametrize(
    "base_url",
    [None, "https://openrouter.ai/api/v1", 'https://proxy.example/v1?label="benchmark"'],
)
def test_codex_init_keeps_endpoint_outside_boolean_features(tmp_path, monkeypatch, base_url):
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    def installed_codex(argv, **kwargs):
        if argv == ["which", "codex"]:
            return subprocess.CompletedProcess(argv, 0, stdout="/usr/local/bin/codex\n", stderr="")
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, stdout="codex-cli 0.145.0\n", stderr="")
        pytest.fail(f"Unexpected external command: {argv}")

    monkeypatch.setattr(subprocess, "run", installed_codex)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()))
    monkeypatch.setattr(os, "chown", lambda *args: None)
    framework = CodexFramework(api_key="fixture-unused-key", agent_version="0.145.0")
    codex_dir = tmp_path / "codex"
    # Execute the generated initialization script with temporary credential and
    # config paths, without installing Codex or changing host account state.
    script = framework.get_container_init_script("codex").replace(
        "/home/fakeroot/.codex", str(codex_dir)
    ).replace("/tmp/host-codex/auth.json", str(tmp_path / "missing-auth.json"))
    exec(compile(script, "<codex-init>", "exec"), {})  # noqa: S102 - exercise our generated container script

    config = tomllib.loads((codex_dir / "config.toml").read_text())
    assert config.get("openai_base_url") == base_url
    assert config["web_search"] == "disabled"
    assert config["features"] == {feature: False for feature in framework.BENCHMARK_DISABLED_FEATURES}
