from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

SCRIPT = Path(__file__).parents[1] / "docker" / "usage_proxy.mjs"


@pytest.fixture
def upstream():
    seen = []
    waiting = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, self.headers.get("authorization"), body))
            scenario = body.get("scenario")
            if scenario == "delayed":
                waiting.set()
                release.wait(timeout=5)
            if scenario == "timeout":
                time.sleep(0.4)
            if scenario == "redirect":
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/other")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if scenario == "partial":
                self.send_response(200)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'{"truncated":')
                self.wfile.flush()
                self.close_connection = True
                return
            response = {"private_response": "response-not-for-logs"}
            if scenario != "no_usage":
                response["usage"] = {
                    "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": 2, "prompt": "private-prompt"},
                    "cost": 0.005, "prompt": "private-prompt", "secret": "fixture-secret",
                }
                if scenario == "no_cost":
                    response["usage"].pop("cost")
            status = 429 if scenario == "rate_limit" else 200
            data = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Timeout/shutdown tests deliberately disconnect the client.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}/v1", seen=seen, waiting=waiting, release=release)
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def proxy_factory(tmp_path, upstream):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the HTTP proxy test")
    processes = []

    def launch(trial="trial-one", timeout=2000):
        root = tmp_path / trial
        root.mkdir(exist_ok=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config = {
            "trial_id": trial + "-id", "trial_name": trial, "port": port,
            "usage_file": str(root / "usage" / "requests.jsonl"), "request_timeout_ms": timeout,
            "upstreams": {
                role: {"base_url": upstream.url, "model": f"test-{role}", "api_key_env": "FIXTURE_KEY"}
                for role in ("llm", "embedding")
            },
        }
        config_path = root / "proxy.json"
        config_path.write_text(json.dumps(config))
        process = subprocess.Popen(
            [node, str(SCRIPT), str(config_path)], env=dict(os.environ, FIXTURE_KEY="fixture-secret"),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        processes.append(process)
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            try:
                with urlopen(base_url + "/health", timeout=0.1):
                    break
            except (URLError, TimeoutError):
                if process.poll() is not None:
                    pytest.fail(process.stderr.read().decode())
                time.sleep(0.01)
        else:
            pytest.fail("Proxy did not start")
        return SimpleNamespace(process=process, url=base_url, log=Path(config["usage_file"]), config=config)

    yield launch
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stderr.close()


def post(proxy, path="/m/m1/llm/v1/chat/completions", *, model="test-llm", scenario=None, **kwargs):
    payload = {"model": model, "messages": [{"role": "user", "content": "private-prompt"}], **kwargs}
    if scenario:
        payload["scenario"] = scenario
    request = Request(
        proxy.url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer client-untrusted-key"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def records(proxy):
    return [json.loads(line) for line in proxy.log.read_text().splitlines()]


def test_concurrent_milestone_attribution_and_private_fields(proxy_factory, upstream):
    proxy = proxy_factory()
    assert proxy.log.read_text() == ""
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(post, proxy, "/m/m1/llm/v1/chat/completions", scenario="delayed")
        assert upstream.waiting.wait(timeout=3)
        assert post(proxy, "/m/m2/embedding/v1/embeddings", model="test-embedding")[0] == 200
        upstream.release.set()
        assert first.result(timeout=3)[0] == 200
    rows = records(proxy)
    assert [(r["milestone"], r["role"]) for r in rows] == [("m2", "embedding"), ("m1", "llm")]
    assert all(r["trial_id"] == "trial-one-id" and r["trial_name"] == "trial-one" for r in rows)
    assert all(r["schema_version"] == 1 and r["status"] == 200 and r["cost_usd"] == 0.005 for r in rows)
    assert rows[0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    assert [item[0] for item in upstream.seen] == ["/v1/chat/completions", "/v1/embeddings"]
    assert all(item[1] == "Bearer fixture-secret" for item in upstream.seen)
    for private in ("private-prompt", "fixture-secret", "client-untrusted-key", "response-not-for-logs"):
        assert private not in proxy.log.read_text()


def test_trials_unassigned_and_resume_append(proxy_factory):
    first, second = proxy_factory("first"), proxy_factory("second")
    assert post(first)[0] == post(second, "/unassigned/llm/v1/chat/completions")[0] == 200
    assert records(first)[0]["trial_id"] == "first-id"
    assert records(second)[0]["trial_id"] == "second-id"
    assert records(second)[0]["milestone"] is None
    first.process.terminate()
    assert first.process.wait(timeout=5) == 0
    resumed = proxy_factory("first")
    assert len(records(resumed)) == 1
    assert post(resumed, "/m/m2/llm/v1/chat/completions")[0] == 200
    assert [row["milestone"] for row in records(resumed)] == ["m1", "m2"]


@pytest.mark.parametrize("scenario,status,error", [
    ("no_usage", 200, None), ("no_cost", 200, None), ("rate_limit", 429, None),
    ("partial", 502, "upstream_error"), ("redirect", 502, "upstream_error"), ("timeout", 504, "timeout"),
])
def test_upstream_failures_and_missing_usage(proxy_factory, upstream, scenario, status, error):
    proxy = proxy_factory(timeout=80 if scenario == "timeout" else 2000)
    assert post(proxy, scenario=scenario)[0] == status
    row, = records(proxy)
    assert row["status"] == status and row["error"] == error
    assert len(upstream.seen) == 1  # No retries or redirects.
    if scenario in {"no_usage", "partial", "redirect", "timeout"}:
        assert row["usage"] is None and row["cost_usd"] is None
    if scenario == "no_cost":
        assert row["usage"]["prompt_tokens"] == 10 and row["cost_usd"] is None


def test_rejects_unsupported_routes_models_streaming_without_upstream_calls(proxy_factory, upstream):
    proxy = proxy_factory()
    for path in (
        "/anything?url=https://example.com", "/m/m1/llm/v1/embeddings", "/m/m1/embedding/v1/chat/completions",
        "/m/../llm/v1/chat/completions", "/m/%2Fm1/llm/v1/chat/completions", "/m/m1/llm/v1/chat/completions?url=x",
    ):
        assert post(proxy, path)[0] == 404
    assert post(proxy, model="wrong")[0] == 400
    assert post(proxy, stream=True)[0] == 400
    assert upstream.seen == [] and records(proxy) == []


def test_shutdown_records_inflight_request_before_exit(proxy_factory, upstream):
    proxy = proxy_factory(timeout=30000)
    with ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(post, proxy, scenario="delayed")
        assert upstream.waiting.wait(timeout=3)
        proxy.process.terminate()
        assert response.result(timeout=3)[0] == 502
        assert proxy.process.wait(timeout=3) == 0
    row, = records(proxy)
    assert row["error"] == "shutdown" and row["status"] == 502 and row["usage"] is None


def test_log_write_failure_is_fatal(proxy_factory):
    proxy = proxy_factory()
    proxy.log.unlink()
    proxy.log.mkdir()
    with pytest.raises((RemoteDisconnected, URLError, ConnectionResetError)):
        post(proxy)
    assert proxy.process.wait(timeout=3) != 0
    assert "usage log append failed" in proxy.process.stderr.read().decode()
