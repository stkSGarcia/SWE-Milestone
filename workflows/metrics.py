"""Workflow reporting remains separate from benchmark correctness scores."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .state import atomic_json


def parse_workflow_stdout(parser, stdout_file: Path, logs_dir: Path) -> dict:
    """Use the upstream cumulative-usage parser separately for each CLI thread."""
    if not stdout_file.exists():
        return parser.parse_stdout_stats(stdout_file, logs_dir)
    groups: dict[str, list[str]] = {}
    current = "unidentified"
    for line in stdout_file.read_text().splitlines(keepends=True):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            thread = event.get("thread")
            current = str(
                event.get("thread_id") or (thread.get("id") if isinstance(thread, dict) else None)
                or event.get("session_id") or "unidentified"
            )
        groups.setdefault(current, []).append(line)
    totals = {"total_cost_usd": 0.0, "total_turns": 0, "session_count": 0, "unique_session_count": 0, "modelUsage": {}}
    raw_files = list(logs_dir.glob("*.jsonl"))
    with tempfile.TemporaryDirectory(prefix="swe-workflow-usage-") as temporary:
        for index, (thread_id, lines) in enumerate(groups.items()):
            directory = Path(temporary) / str(index)
            raw = directory / "codex"
            raw.mkdir(parents=True)
            for source in raw_files:
                if source.name.endswith(f"{thread_id}.jsonl"):
                    (raw / source.name).symlink_to(os.path.relpath(source, raw))
            part = directory / "agent_stdout.txt"
            part.write_text("".join(lines))
            stats = parser.parse_stdout_stats(part, raw)
            for field in ("total_cost_usd", "total_turns", "session_count", "unique_session_count"):
                totals[field] += stats.get(field, 0)
            for model, usage in stats.get("modelUsage", {}).items():
                aggregate = totals["modelUsage"].setdefault(model, {})
                for field, value in usage.items():
                    if field == "contextWindow":
                        aggregate[field] = max(aggregate.get(field, 0), value)
                    elif isinstance(value, (int, float)):
                        aggregate[field] = aggregate.get(field, 0) + value
                    else:
                        aggregate[field] = value
    return totals


def collect_stage_usage(root: Path, mid: str, stage: str, entry: dict) -> dict | None:
    from harness.e2e.log_parser.codex import CodexLogParser

    directory = root / "milestones" / mid / "stages" / stage
    stdout = root.parent / "log" / "agent_stdout.txt"
    if not stdout.exists():
        return None
    attempts = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("attempt-*/result.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    ]
    if not (directory / f"attempt-{entry.get('attempts', 0)}" / "result.json").exists():
        attempts.append(entry)
    parts = []
    with stdout.open("rb") as stream:
        size = stdout.stat().st_size
        for attempt in attempts:
            start, end = attempt.get("stdout_offset"), attempt.get("stdout_end")
            if end is None and attempt is entry and entry.get("status") == "running":
                end = size
            if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                continue
            stream.seek(start)
            parts.append(stream.read(end - start).decode("utf-8", errors="replace"))
    if not parts:
        return None
    combined = "".join(parts)
    if '"turn.completed"' not in combined:
        return None
    with tempfile.TemporaryDirectory(prefix="swe-workflow-stage-usage-") as temporary:
        aggregate = Path(temporary) / "agent_stdout.txt"
        aggregate.write_text(combined)
        return parse_workflow_stdout(CodexLogParser(), aggregate, root.parent / "log" / "codex")


def _read_artifactnet_requests(root: Path) -> tuple[str | None, dict, list[dict]]:
    """Read only records explicitly bound to this trial's persistent identity."""
    try:
        runtime = json.loads((root / "runtime.json").read_text())
    except (OSError, ValueError):
        runtime = {}
    trial_id = runtime.get("project") if isinstance(runtime, dict) else None
    if not isinstance(trial_id, str) or not trial_id:
        trial_id = None
    path = root / "usage" / "requests.jsonl"
    audit = {
        "path": "usage/requests.jsonl",
        "present": path.exists(),
        "identity_available": trial_id is not None,
        "invalid_records": 0,
        "legacy_records": 0,
        "mismatched_trial_records": 0,
        "unidentified_trial_records": 0,
    }
    legacy = root / "usage" / "artifactnet.jsonl"
    if legacy.exists():
        with legacy.open(errors="replace") as stream:
            audit["legacy_records"] = sum(bool(line.strip()) for line in stream)
    records = []
    if path.exists():
        with path.open(errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    audit["invalid_records"] += 1
                    continue
                if not isinstance(record, dict):
                    audit["invalid_records"] += 1
                elif not isinstance(record.get("trial_id"), str) or not record["trial_id"]:
                    audit["legacy_records"] += 1
                elif trial_id is None:
                    audit["unidentified_trial_records"] += 1
                elif record["trial_id"] != trial_id:
                    audit["mismatched_trial_records"] += 1
                elif (
                    type(record.get("schema_version")) is not int or record["schema_version"] != 1
                    or record.get("role") not in ("llm", "embedding")
                    or type(record.get("status")) is not int
                    or not 100 <= record["status"] <= 599
                    or (record.get("milestone") is not None and (
                        not isinstance(record["milestone"], str)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", record["milestone"])
                    ))
                ):
                    audit["invalid_records"] += 1
                else:
                    records.append(record)
    audit["records_complete"] = bool(
        audit["present"] and audit["identity_available"]
        and not (audit["invalid_records"] or audit["legacy_records"] or audit["unidentified_trial_records"])
    )
    return trial_id, audit, records


def _summarize_requests(records: list[dict], *, complete: bool, by_role: bool = True) -> dict:
    """Tokens are reported subtotals; an unknown charge never becomes zero cost."""
    result = {
        "requests": len(records),
        "failed_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_missing_requests": 0,
        "known_cost_usd": 0.0,
        "cost_missing_requests": 0,
    }
    for record in records:
        result["failed_requests"] += not 200 <= record["status"] < 300
        usage = record.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        tokens = {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get(
                "output_tokens", 0 if record["role"] == "embedding" else None,
            )),
            "total_tokens": usage.get("total_tokens"),
        }
        for key, value in tokens.items():
            if type(value) is not int or value < 0:
                tokens[key] = None
        if tokens["total_tokens"] is None and tokens["input_tokens"] is not None and tokens["output_tokens"] is not None:
            tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
        result["usage_missing_requests"] += any(value is None for value in tokens.values())
        for key, value in tokens.items():
            if value is not None:
                result[key] += value
        cost = record.get("cost_usd")
        if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
            result["known_cost_usd"] += cost
        else:
            result["cost_missing_requests"] += 1
    result["usage_complete"] = complete and result["usage_missing_requests"] == 0
    result["cost_complete"] = complete and result["cost_missing_requests"] == 0
    result["cost_usd"] = result["known_cost_usd"] if result["cost_complete"] else None
    if by_role:
        result["by_role"] = {
            role: _summarize_requests([r for r in records if r["role"] == role], complete=complete, by_role=False)
            for role in ("llm", "embedding")
        }
    return result


def collect_metrics(root: Path) -> dict:
    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    milestones = state.get("milestones", {})
    for mid, milestone in milestones.items():
        for stage, entry in milestone["stages"].items():
            entry["agent_usage"] = collect_stage_usage(root, mid, stage, entry)
            entry.pop("artifactnet_usage", None)
    trial_id, audit, records = _read_artifactnet_requests(root)
    grouped: dict[str, list[dict]] = {mid: [] for mid in milestones}
    unattributed = []
    for record in records:
        mid = record.get("milestone")
        if mid is None:
            unattributed.append(record)
        else:
            grouped.setdefault(mid, []).append(record)
    for mid, requests in grouped.items():
        milestones.setdefault(mid, {"stages": {}})["artifactnet_usage"] = _summarize_requests(
            requests, complete=audit["records_complete"],
        )
    result = {
        "schema_version": 3,
        "trial_id": trial_id,
        "trial_name": root.parent.name,
        "milestones": milestones,
        "artifactnet_usage": _summarize_requests(records, complete=audit["records_complete"]),
        "artifactnet_unattributed_usage": _summarize_requests(unattributed, complete=audit["records_complete"]),
        "artifactnet_log": audit,
        "agent_usage_source": "../agent_stats.json",
        "session_policy": "fresh_per_stage_resume_within_stage",
    }
    atomic_json(root / "metrics.json", result)
    return result
