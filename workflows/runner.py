"""Stage execution with independent sessions and durable milestone recovery."""

from __future__ import annotations

import hashlib
import json
import re
import time

from harness.e2e.agent_runner import AgentRunner, E2EAgentRunner

from .config import stage_names
from .state import atomic_json


class WorkflowRunner(E2EAgentRunner):
    def __init__(self, *, integration, **kwargs):
        super().__init__(**kwargs)
        self.workflow = integration
        self.current_stage = None

    def _wrap_with_pidfile(self, agent_cmd: str) -> str:
        wrapped = super()._wrap_with_pidfile(agent_cmd)
        if self.current_stage is not None:
            self.current_stage["invocation_id"] = self._invocation_id
            self.workflow.state.save()
        return wrapped

    def _update_session_id_from_output(self) -> None:
        # Global stdout and the latest container rollout can belong to another
        # stage (or a Codex subagent). Only this invocation can supply a new ID.
        entry = self.current_stage or {}
        self.session_id = entry.get("session_id")
        stdout = self.log_dir / "agent_stdout.txt"
        if stdout.exists() and "stdout_offset" in entry:
            with stdout.open("rb") as stream:
                stream.seek(entry["stdout_offset"])
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "thread.started":
                        sid = event.get("thread_id")
                        if isinstance(sid, str) and sid:
                            self.session_id = sid
        session_file = self.log_dir / "session_id.txt"
        if self.session_id:
            session_file.write_text(self.session_id)
        else:
            session_file.unlink(missing_ok=True)

    def _get_exec_env_vars(self) -> list[str]:
        base = super()._get_exec_env_vars()
        values = {base[i + 1].split("=", 1)[0]: base[i + 1].split("=", 1)[1] for i in range(0, len(base), 2)}
        # Retain the shared Go/offline environment, including its original PATH.
        path = values.get(
            "PATH", "/home/fakeroot/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        values.update(
            PATH=f"/opt/workflow/bin:{path}",
            CI="true",
            OPENSPEC_TELEMETRY="0",
            NO_COLOR="1",
        )
        return [item for key, value in values.items() for item in ("-e", f"{key}={value}")] + (
            self.workflow.runtime.api_key_env_args()
        )

    def invalidate_persistent_session(self, reason: str = "unknown"):
        if self.current_stage is not None:
            self.current_stage["session_id"] = None
            self.workflow.state.save()
        return super().invalidate_persistent_session(reason)

    def send_recover_message(self, has_new_tasks: bool = True, timeout_ms=None) -> bool:
        # The override limits each agent invocation; cumulative stage/milestone
        # budgets still account for all invocations across recovery calls.
        original_timeout = self.timeout_ms
        if timeout_ms is not None:
            self.timeout_ms = min(original_timeout, timeout_ms)
        try:
            return self.run()
        finally:
            self.timeout_ms = original_timeout

    def run(self, prompt=None, session_id=None) -> bool:
        self._last_fatal_error = None
        try:
            mid = self.workflow.next_milestone()
            if mid is None:
                return True  # watcher owns pending submissions and DAG progression
            self.workflow.provider.prepare_runtime_config(mid)
            record = self.workflow.state.milestone(mid)
            change = record.setdefault(
                "change_id",
                "m-"
                + re.sub(r"[^a-z0-9-]", "-", mid.lower())[:65]
                + "-"
                + hashlib.sha256(mid.encode()).hexdigest()[:8],
            )
            self.workflow.state.save()
            for stage in stage_names(self.workflow.config):
                entry = record["stages"].setdefault(stage, {"status": "pending", "attempts": 0, "elapsed_seconds": 0.0})
                self.current_stage = entry
                if entry["status"] == "complete":
                    continue
                if not self._run_stage(mid, change, stage, record, entry):
                    return False
            self.workflow.submit(mid, record)
            return True
        except (RuntimeError, ValueError, OSError) as error:
            self._last_fatal_error = f"Workflow stopped: {error}"
            self.workflow.state.event("workflow_error", error=str(error))
            return False

    def _run_stage(self, mid: str, change: str, stage: str, record: dict, entry: dict) -> bool:
        state, config = self.workflow.state, self.workflow.config
        stdout = self.log_dir / "agent_stdout.txt"
        if entry.get("invocation_id") and entry["status"] in {"running", "failed", "interrupted"}:
            identity = entry["invocation_id"]
            if not re.fullmatch(r"[0-9a-f]{12}", identity):
                raise RuntimeError("Invalid persisted workflow invocation identity")
            self._invocation_id = identity
            self._invocation_pidfile = f"/tmp/evoclaw_invocation_{identity}.pid"
            if not self._kill_container_invocation("workflow recovery"):
                raise RuntimeError("Previous workflow invocation is still alive; refusing concurrent execution")
            entry.pop("invocation_id", None)
            state.save()
        if entry["status"] == "running":
            # An interrupted process has no completion timestamp. Charge at most
            # its allocated attempt budget before letting the agent continue.
            elapsed = min(max(0, time.time() - entry["started_at"]), entry["allocated_seconds"])
            entry["elapsed_seconds"] += elapsed
            record["elapsed_seconds"] += elapsed
            entry["status"] = "interrupted"
            directory = self.workflow.root / "milestones" / mid / "stages" / stage / f"attempt-{entry['attempts']}"
            entry["stdout_end"] = stdout.stat().st_size if stdout.exists() else 0
            if stdout.exists() and "stdout_offset" in entry:
                with stdout.open("rb") as stream:
                    stream.seek(entry["stdout_offset"])
                    lines = stream.read(max(0, entry["stdout_end"] - entry["stdout_offset"])).splitlines()
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "thread.started" and event.get("thread_id"):
                        sid = event["thread_id"]
                        if sid not in entry.setdefault("session_ids", []):
                            entry["session_ids"].append(sid)
                        if not entry.get("force_fresh_session"):
                            entry["session_id"] = sid
            atomic_json(directory / "result.json", {**entry, "duration_seconds": elapsed})
            state.save()
        if entry["attempts"] and entry["status"] in {"failed", "interrupted", "complete"}:
            previous_result = (
                self.workflow.root / "milestones" / mid / "stages" / stage / f"attempt-{entry['attempts']}" / "result.json"
            )
            # A crash after saving state but before writing result.json must not
            # lose the old log range when the next attempt replaces these fields.
            if "stdout_offset" in entry and "stdout_end" in entry and not previous_result.exists():
                atomic_json(previous_result, entry)
        budgets = config["budgets"]
        remaining = min(
            budgets["stage_seconds"] - entry["elapsed_seconds"],
            budgets["milestone_seconds"] - record["elapsed_seconds"],
        )
        if entry["attempts"] >= budgets["max_attempts"] or remaining <= 0:
            raise RuntimeError(f"Workflow budget exhausted for {mid}/{stage}")
        text = self.workflow.provider.prompt(stage, mid, change, self.workflow.task(mid), self.repo_src_dirs)
        if entry["status"] in {"interrupted", "failed"}:
            text += (
                "\nThe previous attempt failed or was interrupted. "
                "Inspect existing work and complete only what remains of this stage.\n"
            )
            if entry.get("error"):
                text += "Previous execution error: " + entry["error"] + "\n"
        remaining = min(remaining, self.timeout_ms / 1000)
        entry.update(
            status="running", attempts=entry["attempts"] + 1, started_at=time.time(), allocated_seconds=remaining
        )
        directory = self.workflow.root / "milestones" / mid / "stages" / stage / f"attempt-{entry['attempts']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.md").write_text(text)
        entry["stdout_offset"] = stdout.stat().st_size if stdout.exists() else 0
        entry.pop("stdout_end", None)
        state.save()
        state.event("stage_started", milestone=mid, stage=stage, attempt=entry["attempts"])
        started = time.monotonic()
        old_timeout = self.timeout_ms
        self.timeout_ms = int(min(old_timeout, remaining * 1000))
        success = False
        try:
            self.workflow.orchestrator.container_setup.prepare_agent_invocation()
            if entry.get("session_id"):
                self.session_id = entry["session_id"]
                success = self.resume_session(self.session_id, text, timeout_ms=self.timeout_ms)
            else:
                success, self.session_id = AgentRunner.run(self, text)
                # Early failures (such as docker cp) return before the base
                # runner extracts a real Codex thread ID, leaving a placeholder.
                self._update_session_id_from_output()
            entry["session_id"] = self.session_id
            entry.pop("force_fresh_session", None)
            if self.session_id and self.session_id not in entry.setdefault("session_ids", []):
                entry["session_ids"].append(self.session_id)
            if self.session_file and self.session_id:
                self.session_file.write_text(self.session_id)
            if success:
                if self.workflow.tag_commit(mid):
                    raise RuntimeError("Agent created a submission tag before workflow finalization")
                entry["status"] = "complete"
                entry.pop("error", None)
        except (RuntimeError, ValueError, OSError) as error:
            entry["error"] = str(error)
            success = False
        finally:
            elapsed = time.monotonic() - started
            entry["elapsed_seconds"] += elapsed
            record["elapsed_seconds"] += elapsed
            if not success:
                entry["status"] = "failed"
            self.timeout_ms = old_timeout
            entry["stdout_end"] = stdout.stat().st_size if stdout.exists() else 0
            state.save()
            atomic_json(directory / "result.json", {**entry, "duration_seconds": elapsed})
            state.event(
                "stage_finished",
                milestone=mid,
                stage=stage,
                success=success,
                session_id=entry.get("session_id"),
                duration_seconds=elapsed,
            )
        return success
