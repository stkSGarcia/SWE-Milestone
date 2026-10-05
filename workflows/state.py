"""Durable state, written atomically under the harness's existing trial lock."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class WorkflowState:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "state.json"
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "schema_version": 1,
                "initialized": False,
                "milestones": {},
            }
        )
        if self.data.get("schema_version") != 1:
            raise ValueError("Unsupported workflow state schema")

    def save(self) -> None:
        atomic_json(self.path, self.data)

    def event(self, kind: str, **fields) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a") as out:
            out.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}) + "\n")
            out.flush()
            os.fsync(out.fileno())

    def milestone(self, mid: str) -> dict:
        if mid not in self.data["milestones"]:
            self.data["milestones"][mid] = {"stages": {}, "status": "pending", "elapsed_seconds": 0.0}
            self.save()
        return self.data["milestones"][mid]
