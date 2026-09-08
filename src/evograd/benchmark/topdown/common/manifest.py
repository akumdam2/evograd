"""Turning observed events into a deterministic, deduplicated manifest.

The manifest carries two views of the same run, and both are needed.

``events`` is the transcript: every observed invocation, in execution order,
with the module path and layer index that produced it. It answers "what did this
model actually do, and in what order", and it is what makes the harvest
auditable -- a claim about the workload can be checked against it.

``configurations`` is the working set: events collapsed by *structural
equivalence*, keeping frequency and every provenance that fed into them. It
answers "how many genuinely different kernels are in here", which is the
question the later task-extraction milestone asks. A decoder's layers are
identical, so the transcript is roughly as many times longer than the set of
things worth optimising as the model has layers.

The deduplication key is the configuration -- task, tensor metadata, dtype,
layout, operator parameters -- and nothing about *where* the call happened. A
``gate_proj`` and an ``up_proj`` at the same shape are one Linear configuration
with two roles, not two configurations; the roles survive as provenance, so
nothing is lost, but a kernel written for one serves both.

Determinism is a property the file must have, not a hope. The semantic hash
covers structure only -- schema, capture scope, workload identity, events,
configurations. Environment, diagnostics and measured results stay in the file
and out of the hash, so the same workload observed on a different machine, on a
different day, hashes the same.

Nothing here names an architecture. What a workload must supply is its schema
string and the exact call sites it patched -- both facts about that model and
the library version it runs against, neither about aggregation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; `from __future__ import annotations`
    from .observe import Event, Observation  # noqa: F401  (never imported at runtime)

PROVENANCE_KIND = "observed"

#: Named so a reader of the manifest is never left to infer the scope from what
#: happens to be present.
NOT_OBSERVED = (
    "the softmax and matmuls inside scaled_dot_product_attention -- derived from "
    "the observed attention configuration in a later milestone, not traced here",
    "residual additions",
    "the embedding lookup",
    "view / reshape / transpose / contiguous",
    "every backward kernel: the backward pass runs, and nothing observes it",
    "all ATen operations below the listed boundaries",
    "CUDA kernels",
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _natural_key(text: str) -> list[tuple[int, Any]]:
    """Sort ``layers.2`` before ``layers.10`` while staying total and stable."""
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
        if part
    ]


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


@dataclass
class Configuration:
    """One structurally distinct invocation, plus everywhere it came from."""

    config_id: str
    task: str
    inputs: list[dict[str, Any]]
    input_kwargs: dict[str, dict[str, Any]]
    outputs: list[dict[str, Any]]
    params: dict[str, Any]
    attrs: dict[str, Any]
    frequency: int = 0
    module_paths: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    layer_indices: list[int | None] = field(default_factory=list)
    first_ordinal: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "task": self.task,
            "frequency": self.frequency,
            "inputs": self.inputs,
            "input_kwargs": self.input_kwargs,
            "outputs": self.outputs,
            "params": self.params,
            "attrs": self.attrs,
            "module_paths": self.module_paths,
            "roles": self.roles,
            "layer_indices": self.layer_indices,
            "first_ordinal": self.first_ordinal,
            "provenance": self.provenance,
        }


def deduplicate(events: Iterable[Event], *, workload_id: str, config_hash: str) -> list[Configuration]:
    """Collapse events by structural equivalence, keeping all provenance."""
    by_key: dict[str, Configuration] = {}
    paths: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = {}
    layers: dict[str, set[int | None]] = {}

    for event in events:
        payload = event.semantic_key_payload()
        key = _sha256(payload)[:16]
        record = by_key.get(key)
        if record is None:
            record = Configuration(
                config_id=key,
                task=event.task,
                inputs=payload["inputs"],
                input_kwargs=payload["input_kwargs"],
                outputs=payload["outputs"],
                params=payload["params"],
                attrs=payload["attrs"],
                first_ordinal=event.ordinal,
                provenance={
                    "kind": PROVENANCE_KIND,
                    "workload_id": workload_id,
                    "config_hash": config_hash,
                },
            )
            by_key[key] = record
            paths[key], roles[key], layers[key] = set(), set(), set()
        record.frequency += 1
        record.first_ordinal = min(record.first_ordinal, event.ordinal)
        if event.module_path:
            paths[key].add(event.module_path)
        if event.role:
            roles[key].add(event.role)
        layers[key].add(event.layer_index)

    for key, record in by_key.items():
        record.module_paths = sorted(paths[key], key=_natural_key)
        record.roles = sorted(roles[key])
        # ``None`` is a real value here: the final RMSNorm and the loss belong to
        # no decoder layer, and saying so is more useful than dropping them.
        record.layer_indices = sorted(layers[key], key=lambda v: (v is not None, v))

    return sorted(by_key.values(), key=lambda r: (r.task, r.first_ordinal, r.config_id))


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


def capture_scope(
    observation: "Observation",
    *,
    function_wrappers: Sequence[str],
    not_observed: Sequence[str] = NOT_OBSERVED,
) -> dict[str, Any]:
    """A statement of what was watched, and -- more importantly -- what was not.

    ``function_wrappers`` names the exact call sites the workload patched. It is
    the workload's to supply: which module holds ``apply_rotary_pos_emb`` is a
    fact about an architecture and a Transformers version, and a list written
    here would be a second copy of it, free to drift from the one that is
    actually installed.
    """
    return {
        "phases_observed": ["forward"],
        "backward_executed": True,
        "backward_observed": False,
        "mechanism": {
            "module_hooks": "forward pre-hook assigns the ordinal, forward hook records the result",
            "function_wrappers": list(function_wrappers),
        },
        "boundaries": observation.boundaries,
        "not_observed": list(not_observed),
        "note": (
            "Forward-side semantic invocations only. The full loss and backward pass "
            "execute, and gradient coverage is validated, but no backward operation "
            "and no CUDA kernel is traced."
        ),
    }


def build_manifest(
    observation: "Observation",
    *,
    schema_version: str,
    function_wrappers: Sequence[str],
    workload: dict[str, Any],
    environment: dict[str, Any],
    validation: dict[str, Any],
    diagnostics: dict[str, Any],
    not_observed: Sequence[str] = NOT_OBSERVED,
) -> dict[str, Any]:
    """Assemble the file. ``manifest_hash`` is computed over the semantic part
    only, then attached -- so it can never depend on itself."""
    events = observation.ordered_events()
    configurations = deduplicate(
        events,
        workload_id=observation.workload_id,
        config_hash=observation.config_hash,
    )

    semantic = {
        "schema_version": schema_version,
        "workload_id": observation.workload_id,
        "config_hash": observation.config_hash,
        "capture_scope": capture_scope(
            observation, function_wrappers=function_wrappers,
            not_observed=not_observed,
        ),
        "counts_by_task": observation.counts_by_task(),
        "events": [event.to_dict() for event in events],
        "configurations": [record.to_dict() for record in configurations],
    }
    manifest_hash = _sha256(semantic)

    return {
        **semantic,
        "manifest_hash": manifest_hash,
        # Everything below is outside the hash on purpose.
        "workload": workload,
        "environment": environment,
        "validation": validation,
        "diagnostics": diagnostics,
    }


def semantic_hash(manifest: dict[str, Any]) -> str:
    """Recompute the hash from a manifest, for verifying a file on disk."""
    semantic = {
        key: manifest[key]
        for key in (
            "schema_version",
            "workload_id",
            "config_hash",
            "capture_scope",
            "counts_by_task",
            "events",
            "configurations",
        )
    }
    return _sha256(semantic)


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# human-readable summary
# --------------------------------------------------------------------------


def _shape_of(entry: dict[str, Any]) -> str:
    if entry.get("kind") != "tensor":
        return entry.get("kind", "?")
    return f"{list(entry['shape'])}{entry['dtype'].replace('torch.', ' ')}"


def _io(record: Configuration | dict[str, Any]) -> str:
    data = record if isinstance(record, dict) else record.to_dict()
    ins = " , ".join(_shape_of(i) for i in data["inputs"]) or "-"
    outs = " , ".join(_shape_of(o) for o in data["outputs"]) or "-"
    return f"{ins}  ->  {outs}"


def summarize(manifest: dict[str, Any], *, top_linear: int = 8,
              label: str = "harvest") -> str:
    lines: list[str] = []
    workload = manifest["workload"]
    lines.append(f"{label} -- {manifest['workload_id']}")
    lines.append(
        f"  canonical={workload.get('canonical')}  config_hash={manifest['config_hash']}"
    )
    lines.append(f"  manifest_hash={manifest['manifest_hash']}")
    lines.append("")
    configs = [dict(c) for c in manifest["configurations"]]
    lines.append(
        f"raw events: {len(manifest['events'])}     "
        f"deduplicated configurations: {len(configs)}"
    )
    lines.append("")
    lines.append("counts by semantic task type")
    per_task_configs: dict[str, int] = {}
    for record in configs:
        per_task_configs[record["task"]] = per_task_configs.get(record["task"], 0) + 1
    for task, count in manifest["counts_by_task"].items():
        lines.append(f"  {task:24s} {count:5d} events   {per_task_configs.get(task, 0):3d} configurations")
    lines.append("")

    linear = sorted(
        (c for c in configs if c["task"] == "linear"),
        key=lambda c: (-c["frequency"], c["first_ordinal"]),
    )
    lines.append(f"linear configurations ({len(linear)})")
    for record in linear[:top_linear]:
        roles = ",".join(record["roles"])
        lines.append(f"  x{record['frequency']:<4d} {roles:28s} {_io(record)}")
    lines.append("")

    norms = sorted(
        (c for c in configs if c["task"] == "rms_norm"),
        key=lambda c: (-c["frequency"], c["first_ordinal"]),
    )
    lines.append(f"rms_norm configurations ({len(norms)})")
    for record in norms:
        roles = ",".join(record["roles"])
        width = record["attrs"].get("normalized_size")
        lines.append(f"  x{record['frequency']:<4d} width={width:<5} {roles:44s} {_io(record)}")
    lines.append("")

    for task in ("sdpa", "causal_cross_entropy", "cross_entropy"):
        for record in configs:
            if record["task"] != task:
                continue
            lines.append(f"{task}  x{record['frequency']}")
            lines.append(f"  {_io(record)}")
            lines.append(f"  attrs: {record['attrs']}")
    lines.append("")

    validation = manifest["validation"]
    lines.append(
        "validation: loss={loss} finite={finite}  grads {have}/{want}  all-finite={fin}".format(
            loss=validation.get("loss"),
            finite=validation.get("loss_is_finite"),
            have=validation.get("params_with_grad"),
            want=validation.get("trainable_params"),
            fin=validation.get("grads_all_finite"),
        )
    )
    missing = validation.get("missing_grad_params") or []
    if missing:
        lines.append(f"  missing gradients: {missing[:10]}")
    return "\n".join(lines)
