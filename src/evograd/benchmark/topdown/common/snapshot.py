"""Reading a tracked workload snapshot, and proving it was not edited by hand.

A harvest manifest is a local result: it is a megabyte of transcript, it is not
version controlled, and it exists only on a machine that has run the canonical
workload. A declaration in ``evograd/ops/`` cannot depend on any of that -- it
has to import on a laptop with no GPU, no Transformers and no ``results/``
directory -- yet the whole point of those tasks is that their shapes come from a
real training step rather than from someone's judgement.

The snapshot is the reconciliation: a small, tracked, hashed JSON file carrying
only what a task needs to state its provenance. This module is the half of that
contract every workload shares -- how to read one and how to verify it. Deriving
a snapshot *from* a harvest is the other half, and it stays in the workload
package, because which boundaries collapse into which task is a fact about a
particular model.

The hash is the integrity check that matters: it covers everything except the
hash field, so any hand edit is caught. The schema string is checked only for
shape here; a workload that cares about its exact version asserts it alongside
the extraction code that produces it.

This module imports json, hashlib, pathlib and re. Nothing else -- no torch, no
Transformers -- because everything that reads a snapshot must work without them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: ``evograd-<workload>-snapshot/<n>``. The workload segment and the revision
#: are the workload's business; that the file declares one at all is not.
SCHEMA_PATTERN = re.compile(r"^evograd-[a-z0-9_.-]+-snapshot/\d+$")


class SnapshotError(RuntimeError):
    """The snapshot is missing, malformed, or disagrees with a full harvest."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def snapshot_hash(payload: dict[str, Any]) -> str:
    """Deterministic hash over everything except the hash field itself."""
    body = {key: value for key, value in payload.items() if key != "snapshot_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def load(path: Path, *, schema_version: str | None = None) -> dict[str, Any]:
    """The frozen snapshot at ``path``, with its hash verified.

    ``schema_version`` pins the exact revision when a caller knows which one it
    expects; without it the string is only required to be well formed, so this
    function can read any workload's snapshot without being told about it.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - tracked in the repo
        raise SnapshotError(f"{path} is missing; it is tracked, not generated") from exc
    declared = payload.get("schema_version")
    if schema_version is not None and declared != schema_version:
        raise SnapshotError(
            f"{path}: schema {declared!r}, expected {schema_version!r}"
        )
    if schema_version is None and not (
        isinstance(declared, str) and SCHEMA_PATTERN.match(declared)
    ):
        raise SnapshotError(
            f"{path}: schema_version {declared!r} is not of the form "
            f"'evograd-<workload>-snapshot/<n>'"
        )
    recomputed = snapshot_hash(payload)
    if payload.get("snapshot_hash") != recomputed:
        raise SnapshotError(
            f"{path}: snapshot hash mismatch (stored {payload.get('snapshot_hash')}, "
            f"recomputed {recomputed}). The file was edited by hand; regenerate it "
            "from a harvest with --write instead."
        )
    return payload


def task(name: str, path: Path, *, schema_version: str | None = None) -> dict[str, Any]:
    """One Level-2 task entry from the snapshot at ``path``."""
    payload = load(path, schema_version=schema_version)
    try:
        return payload["tasks"][name]
    except KeyError:
        raise SnapshotError(
            f"the snapshot has no task {name!r}; it carries {sorted(payload['tasks'])}"
        ) from None
