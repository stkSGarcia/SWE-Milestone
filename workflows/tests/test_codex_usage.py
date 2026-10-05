"""Usage attribution reads byte ranges from the original log, without saved copies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.e2e.log_parser.codex import CodexLogParser
from workflows.metrics import collect_stage_usage, parse_workflow_stdout


def record_attempt(root: Path, mid: str, stage: str, number: int, thread: str, tokens: int) -> dict:
    logs = root.parent / "log"
    (logs / "codex").mkdir(parents=True, exist_ok=True)
    rows = [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {"text": "正在处理：中文日志"}},
        {"type": "turn.completed", "usage": {"input_tokens": tokens, "output_tokens": 2}},
    ]
    with (logs / "agent_stdout.txt").open("ab") as stream:
        start = stream.tell()
        stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
        end = stream.tell()
    with (logs / "codex" / f"rollout-{thread}.jsonl").open("a") as stream:
        stream.write(json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.2-codex"}}) + "\n")
    entry = {"status": "complete", "attempts": number, "stdout_offset": start, "stdout_end": end}
    directory = root / "milestones" / mid / "stages" / stage / f"attempt-{number}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(entry))
    return entry


def test_resumed_thread_counts_cumulative_usage_once(tmp_path):
    root = tmp_path / "workflow"
    record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    entry = record_attempt(root, "m1", "propose", 2, "thread-a", 16)

    stats = collect_stage_usage(root, "m1", "propose", entry)

    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 16
    assert stats["modelUsage"]["gpt-5.2-codex"]["outputTokens"] == 2
    assert stats["session_count"] == 2
    assert stats["unique_session_count"] == 1
    assert stats["total_turns"] == 2


def test_byte_ranges_isolate_fresh_threads_stages_and_milestones_without_copies(tmp_path, monkeypatch):
    root = tmp_path / "workflow"
    logs = tmp_path / "log"
    logs.mkdir()
    (logs / "agent_stdout.txt").write_text("既有日志，不属于任何阶段。\n")
    record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    propose = record_attempt(root, "m1", "propose", 2, "thread-b", 7)
    apply = record_attempt(root, "m1", "apply", 1, "thread-c", 101)
    next_milestone = record_attempt(root, "m2", "propose", 1, "thread-d", 1001)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(temporary))
    original_paths = set(tmp_path.rglob("*"))

    first = collect_stage_usage(root, "m1", "propose", propose)
    second = collect_stage_usage(root, "m1", "apply", apply)
    third = collect_stage_usage(root, "m2", "propose", next_milestone)
    total = parse_workflow_stdout(CodexLogParser(), logs / "agent_stdout.txt", logs / "codex")

    assert first["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 17
    assert first["modelUsage"]["gpt-5.2-codex"]["outputTokens"] == 4
    assert first["unique_session_count"] == first["total_turns"] == 2
    assert second["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 101
    assert third["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 1001
    assert total["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 1119
    assert set(tmp_path.rglob("*")) == original_paths
    assert not list(temporary.iterdir())


def test_running_attempt_uses_current_log_end_without_result(tmp_path):
    root = tmp_path / "workflow"
    record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    entry = record_attempt(root, "m1", "propose", 2, "thread-a", 16)
    result = root / "milestones/m1/stages/propose/attempt-2/result.json"
    result.unlink()
    entry["status"] = "running"
    entry.pop("stdout_end")

    stats = collect_stage_usage(root, "m1", "propose", entry)

    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 16
    assert stats["session_count"] == 2
    assert not result.exists()
    assert "stdout_end" not in entry


def test_saved_state_bounds_completed_attempt_when_result_write_was_interrupted(tmp_path):
    root = tmp_path / "workflow"
    entry = record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    (root / "milestones/m1/stages/propose/attempt-1/result.json").unlink()
    record_attempt(root, "m1", "apply", 1, "thread-b", 100)

    stats = collect_stage_usage(root, "m1", "propose", entry)

    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 10
    assert stats["session_count"] == stats["total_turns"] == 1


@pytest.mark.parametrize("noise", ["null", "[]", "17"])
def test_valid_json_noise_does_not_abort_accounting(tmp_path, noise):
    root = tmp_path / "workflow"
    record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    logs = root.parent / "log"
    stdout = logs / "agent_stdout.txt"
    stdout.write_text(noise + "\n" + stdout.read_text())

    stats = parse_workflow_stdout(CodexLogParser(), stdout, logs / "codex")

    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 10
    assert stats["session_count"] == stats["unique_session_count"] == 1


@pytest.mark.parametrize("thread", [None, [], {"id": "thread-a"}])
def test_alternative_thread_identity_does_not_require_nested_object(tmp_path, thread):
    root = tmp_path / "workflow"
    record_attempt(root, "m1", "propose", 1, "thread-a", 10)
    logs = root.parent / "log"
    stdout = logs / "agent_stdout.txt"
    rows = [json.loads(line) for line in stdout.read_text().splitlines()]
    rows[0] = {"type": "thread.started", "thread": thread, "session_id": "thread-a"}
    stdout.write_text("".join(json.dumps(row) + "\n" for row in rows))

    stats = parse_workflow_stdout(CodexLogParser(), stdout, logs / "codex")

    assert stats["modelUsage"]["gpt-5.2-codex"]["inputTokens"] == 10
    assert stats["total_turns"] == stats["unique_session_count"] == 1
