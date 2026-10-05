"""The only module that knows SWE-Milestone's orchestration interfaces."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import restore, stage_names
from .providers import create_provider
from .runtime import WorkflowRuntime
from .state import WorkflowState, atomic_json


class WorkflowIntegration:
    def __init__(self, orchestrator, config: dict):
        self.orchestrator = orchestrator
        self.config = config
        self.root = orchestrator.trial_root / "workflow"
        self.state = WorkflowState(self.root)
        self.runtime = WorkflowRuntime(config, self.root, orchestrator.container_name)
        self.provider = create_provider(config, self.runtime)

    @classmethod
    def from_trial(cls, orchestrator, binding, requested: Path | None = None):
        config = restore(binding, orchestrator.trial_root, requested)
        return cls(orchestrator, config) if config else None

    def prepare(self) -> None:
        self.runtime.prepare()
        self.orchestrator.container_setup.runtime_extension = self.runtime
        self.orchestrator.submission_guard = self.assert_submission

    def initialize(self) -> None:
        self.runtime.install_cli_launchers()
        # The original repository BASE has already been recorded by the harness.
        if not self.state.data["initialized"]:
            self.provider.initialize()
            self.state.data["initialized"] = True
            self.state.save()
        self.provider.prepare_runtime_config()
        versions = self.provider.verify_tools()
        self.runtime.verify_services()
        for mid, record in self.state.data["milestones"].items():
            if record.get("status") == "submitting" and self.tag_commit(mid) == record.get("submission_commit"):
                record["status"] = "submitted"
        self.state.save()
        self.state.event("runtime_ready", versions=versions, provider=self.config["provider"])
        atomic_json(self.root / "versions.json", versions)

    def invalidate_sessions(self) -> None:
        for record in self.state.data["milestones"].values():
            for entry in record["stages"].values():
                entry["session_id"] = None
                entry["force_fresh_session"] = True
        self.state.save()
        self.state.event("sessions_invalidated", reason="no_resume_session")

    def parse_stdout_stats(self, parser, stdout_file, logs_dir):
        from .metrics import parse_workflow_stdout

        return parse_workflow_stdout(parser, stdout_file, logs_dir)

    def create_runner(self, **kwargs):
        from .runner import WorkflowRunner

        return WorkflowRunner(integration=self, **kwargs)

    def progress_snapshot(self) -> set[tuple[str, str, str]]:
        """Count durable completions, rather than retries or elapsed time."""
        progress = set()
        for mid, record in self.state.data["milestones"].items():
            for stage, entry in record["stages"].items():
                if entry.get("status") == "complete":
                    progress.add(("stage", mid, stage))
            if record.get("status") in {"submitting", "submitted"} and record.get("submission_commit"):
                progress.add(("submission", mid, record["submission_commit"]))
        return progress

    def pending_submissions(self) -> set[str]:
        """Include tags the watcher has not yet discovered or debounced."""
        snapshot = self.orchestrator.dag.get_state_snapshot()
        terminal = snapshot["completed"] | snapshot["failed"] | snapshot["skipped"]
        return {
            mid for mid, record in self.state.data["milestones"].items()
            if mid not in terminal and record.get("status") in {"submitting", "submitted"}
            and record.get("submission_commit")
        }

    def assert_submission(self, mid: str, commit: str) -> None:
        # Watcher thread reads an atomic on-disk snapshot, never a half-mutated dict.
        data = json.loads(self.state.path.read_text())
        record = data["milestones"].get(mid, {})
        complete = all(
            record.get("stages", {}).get(stage, {}).get("status") == "complete" for stage in stage_names(self.config)
        )
        if not complete or record.get("submission_commit") != commit:
            raise RuntimeError(f"Workflow has not authorized this submission for {mid}")

    def next_milestone(self) -> str | None:
        candidates = self.orchestrator.dag.get_next_runnable()
        for mid in candidates:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", mid):
                raise ValueError(f"Unsupported milestone identifier: {mid!r}")
            record = self.state.milestone(mid)
            tag = self.tag_commit(mid)
            if tag:
                if record.get("submission_commit") != tag:
                    raise RuntimeError(f"Unexpected or moved submission tag for {mid}")
                record["status"] = "submitted"
                self.state.save()
                continue
            if record["status"] == "submitted":
                raise RuntimeError(f"Submission tag disappeared for {mid}")
            return mid
        return None

    def task(self, mid: str) -> str:
        # Only the selected runnable task is sent to a stage; never preload future SRSs.
        return (self.orchestrator.srs_root / mid / "SRS.md").read_text()

    def tag_commit(self, mid: str) -> str | None:
        result = self.orchestrator._docker_exec_git(
            "rev-parse", "--verify", "--quiet", f"refs/tags/agent-impl-{mid}^{{commit}}"
        )
        if result.returncode == 1:
            return None
        if result.returncode:
            # `show-ref --verify` uses 128 for an absent ref on some Git releases.
            if "not a valid ref" in result.stderr or "not a valid ref" in result.stdout:
                return None
            raise RuntimeError(f"Cannot inspect submission tag for {mid}")
        return result.stdout.strip()

    def submit(self, mid: str, record: dict) -> None:
        existing = self.tag_commit(mid)
        if existing:
            if record.get("submission_commit") != existing:
                raise RuntimeError(f"Refusing to replace existing submission tag for {mid}")
        else:
            if not record.get("submission_commit"):
                # Git handles additions, renames and deletions. The benchmark's
                # existing snapshot pipeline owns source/test/manifest filtering.
                # Leave extension directories out of this staging operation.
                workflow_dirs = ("openspec", ".artnet", ".codex")
                self.runtime.exec(
                    ["git", "add", "-A", "--", ".", *[f":(top,exclude){p}" for p in workflow_dirs]]
                )
                self.runtime.exec(["git", "reset", "-q", "HEAD", "--", *workflow_dirs])
                self.runtime.exec(
                    [
                        "git",
                        "-c",
                        "user.name=Workflow Controller",
                        "-c",
                        "user.email=workflow@local",
                        "commit",
                        "--allow-empty",
                        "-m",
                        f"Implement {mid}",
                    ]
                )
                record["submission_commit"] = self.runtime.exec(["git", "rev-parse", "HEAD"]).strip()
                record["status"] = "submitting"
                self.state.save()  # persist intent BEFORE creating the externally observed tag
            self.runtime.exec(["git", "tag", f"agent-impl-{mid}", record["submission_commit"]])
        record["status"] = "submitted"
        self.state.save()
        self.state.event("milestone_submitted", milestone=mid, commit=record["submission_commit"])

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            from .metrics import collect_metrics

            collect_metrics(self.root)
