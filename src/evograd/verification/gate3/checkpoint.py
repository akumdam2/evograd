"""Resumable Gate 3 runs: one training trajectory across several processes.

A 1000-step run of a billion-parameter model outlasts one GPU allocation. With
a :class:`RunCheckpoint`, :func:`~evograd.verification.gate3.runner.measure_training`
writes the run's complete state every ``every`` steps and when ``deadline``
passes, and a later call with the same checkpoint continues it. The state is
everything the next step depends on: model and optimizer state, torch CPU and
CUDA generators, Python's ``random``, the step records, the site-call counts
and the run id. A resumed run therefore has the same records, identity and
digest as one that never stopped; only its ``segments`` (provenance) differ.

A checkpoint is used only when it belongs to this run: same run identity
(plan, seed, adapter, data, optimizer, environment, context, initial state)
and same deployment. Anything else is moved aside and the run starts over.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA = "evograd-verification-gate3-checkpoint/1"


@dataclass(frozen=True)
class RunCheckpoint:
    """Where a run keeps its state, how often, and until when it may run."""

    path: Path
    #: Save after every ``every`` completed steps (0: only at the deadline).
    every: int = 100
    #: Absolute ``time.time()`` after which the run saves and suspends.
    deadline: float | None = None

    def expired(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline


def rng_state(device: str) -> dict[str, Any]:
    state = {"torch_cpu": torch.get_rng_state(), "python": random.getstate()}
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return state


def restore_rng(state: dict[str, Any], device: str) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], torch.device(device))


def save(checkpoint: RunCheckpoint, payload: dict[str, Any], *, attempts: int = 3) -> None:
    """Write atomically: a crash mid-write leaves the previous checkpoint.

    Each attempt writes a fresh temporary name (a name recreated right after
    its rename has left an unreadable entry on a parallel filesystem) and is
    retried; the last error propagates.
    """
    import uuid

    path = Path(checkpoint.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            torch.save({"schema": SCHEMA, **payload}, tmp)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError):
            try:
                tmp.unlink()
            except OSError:
                pass
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0 + attempt)


def load(checkpoint: RunCheckpoint) -> dict[str, Any] | None:
    path = Path(checkpoint.path)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data if data.get("schema") == SCHEMA else None


def discard(checkpoint: RunCheckpoint, reason: str) -> str | None:
    """Move a checkpoint that does not belong to this run aside; returns where."""
    path = Path(checkpoint.path)
    if not path.exists():
        return None
    aside = path.with_name(f"{path.name}.stale-{time.strftime('%Y%m%d%H%M%S')}")
    os.replace(path, aside)
    return str(aside)


def remove(checkpoint: RunCheckpoint) -> None:
    Path(checkpoint.path).unlink(missing_ok=True)
