"""The run ledger: what an agent was given and what it spent.

A run directory holds:

    run.json            agent, model, budget, task, start/end, outcome
    evaluations.jsonl   one line per evaluation the agent requested
    llm.jsonl           one line per LLM call, where the agent exposes it

Appends are single ``write`` calls of one line, so the agent's own worker
processes (KernelAgent, GEAK's unittest) can append concurrently.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
EVALUATIONS_FILE = "evaluations.jsonl"
LLM_FILE = "llm.jsonl"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def append_line(path: Path, record: dict[str, Any]) -> None:
    line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RunLedger:
    """Create or reopen ``run_dir`` for one agent run on one task."""

    def __init__(self, run_dir: str | Path, *, agent: str | None = None,
                 task=None, config: dict[str, Any] | None = None):
        self.root = Path(run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / RUN_FILE
        if path.is_file():
            self.record = json.loads(path.read_text(encoding="utf-8"))
        else:
            if agent is None or task is None:
                raise ValueError(f"{self.root} has no {RUN_FILE}; agent and task are required")
            self.record = {
                "agent": agent,
                "task_id": task.task_id,
                "task_dir": str(task.root),
                "statement_sha256": task.spec.get("statement_sha256"),
                "config": config or {},
                "started": _now(),
                "started_unix": time.time(),
                "status": "running",
            }
            self._save()

    def _save(self) -> None:
        path = self.root / RUN_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        temporary.replace(path)

    def log_evaluation(self, record: dict[str, Any]) -> None:
        append_line(self.root / EVALUATIONS_FILE, {"time": _now(), **record})

    def log_llm(self, record: dict[str, Any]) -> None:
        append_line(self.root / LLM_FILE, {"time": _now(), **record})

    def finish(self, *, status: str, best_program: str | Path | None = None,
               **extra: Any) -> None:
        self.record.update({
            "status": status,
            "finished": _now(),
            "wall_seconds": round(time.time() - float(self.record.get("started_unix", time.time())), 1),
            "best_program": str(best_program) if best_program else None,
            **extra,
        })
        self._save()

    def summary(self) -> dict[str, Any]:
        evaluations = read_lines(self.root / EVALUATIONS_FILE)
        llm = read_lines(self.root / LLM_FILE)
        tokens: dict[str, int] = {}
        for entry in llm:
            for key, value in (entry.get("usage") or {}).items():
                if isinstance(value, int):
                    tokens[key] = tokens.get(key, 0) + value
        return {
            "evaluations": len(evaluations),
            "correct_evaluations": sum(1 for e in evaluations if e.get("correct")),
            "evaluation_kinds": sorted({e.get("kind", "full") for e in evaluations}),
            "llm_calls_recorded": len(llm),
            "llm_tokens_recorded": tokens,
            "wall_seconds": self.record.get("wall_seconds"),
            "status": self.record.get("status"),
        }
