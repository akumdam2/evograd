"""Deriving a tracked snapshot from a full harvest manifest.

The snapshot is a *frozen extract*, not a second source of truth: ``--validate``
re-derives it from a harvest and fails if the two disagree, so it cannot quietly
drift away from the run it claims to describe.

Reading one is in :mod:`.snapshot`, and every operator declaration reaches that
at import time. This half runs only when a snapshot is written or validated, so
it is kept apart -- nothing that imports ``evograd.ops`` pays for it.

**What a workload supplies.** Three things, and nothing else here names a model:

``level1_sources``
    Which generic Level-1 operator each harvested configuration maps onto, and
    which ``ModelConfig`` component re-derives its dims. :data:`DECODER_LEVEL1`
    is the mapping every HuggingFace decoder shares.
``task_sources``
    Which harvested boundaries collapse into which Level-2 task. This one is
    genuinely per-model: a fused operator is a claim about a particular
    decoder's structure.
``schema_version``
    So a snapshot declares which workload produced it.

An unmapped role is an error rather than a smaller snapshot. A configuration the
model really ran, silently dropped, would leave a task with shapes that do not
add up to the step -- so a role with no component raises and names itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .snapshot import SnapshotError, snapshot_hash

# --------------------------------------------------------------------------
# extraction from a full harvest
# --------------------------------------------------------------------------


def _find(manifest: dict[str, Any], task_name: str, roles: list[str]) -> dict[str, Any]:
    """The one configuration that *contains* the requested roles.

    Containment rather than equality, because deduplication is shape-driven and
    a smaller model can merge roles a larger one keeps apart -- at Qwen3-0.6B's
    widths ``q_proj`` (1024->2048) and ``o_proj`` (2048->1024) are distinct
    configurations, but in a model whose query fan-out equals its hidden size
    they are the same one. Requiring equality would make the extractor fail on
    the smaller model rather than describe it; the record keeps its true role
    list either way, so nothing is hidden.
    """
    wanted = set(roles)
    matches = [
        record
        for record in manifest["configurations"]
        if record["task"] == task_name and wanted <= set(record["roles"])
    ]
    if len(matches) != 1:
        raise SnapshotError(
            f"expected exactly one {task_name} configuration containing roles "
            f"{sorted(wanted)}, found {len(matches)}"
        )
    return matches[0]


#: Which generic Level-1 task each harvested configuration maps onto, and which
#: ``ModelConfig`` component re-derives its dims. Only *names* live here; every
#: dimension comes from the harvest, and ``tests/test_provenance`` proves the
#: two agree by re-deriving each shape from the published configuration.
#:
#: These are the HuggingFace decoder convention rather than one model's.
#: ``q_norm``/``k_norm`` are Qwen3's per-head normalizations and simply never
#: appear in an architecture that lacks them; the rest are spelled identically
#: by Llama, Mistral and Qwen. A workload with a boundary not covered here
#: passes its own mapping to :func:`extract`.
#:
#: Two deliberate absences. Standalone ``softmax`` is not mapped: these models
#: run fused SDPA and never materialize one, so a softmax case would be a shape
#: no step executes. And the observed ``silu`` record maps onto ``swiglu``
#: rather than a bare activation task, because the production pointwise boundary
#: is ``silu(gate) * up`` -- the SiLU record is kept as supporting provenance.
DECODER_LEVEL1: dict[str, dict[str, Any]] = {
    # Biasless: every projection and the lm_head in a modern decoder set
    # bias=False, which the harvest records per configuration. They map onto
    # `linear_no_bias`, not the bias-carrying `linear` task.
    "linear_no_bias": {
        "harvested_task": "linear",
        "component_by_role": {
            "q_proj": ("attn_qkv", "tokens"),
            "k_proj": ("attn_kv_proj", "tokens"),
            "o_proj": ("attn_out_proj", "tokens"),
            "gate_proj": ("mlp_up", "tokens"),
            "down_proj": ("mlp_down", "tokens"),
            "lm_head": ("lm_head", "tokens"),
        },
    },
    "rmsnorm": {
        "harvested_task": "rms_norm",
        "component_by_role": {
            "input_layernorm": ("rmsnorm", "tokens"),
            "q_norm": ("q_head_norm", "batch_seq"),
            "k_norm": ("k_head_norm", "batch_seq"),
        },
    },
    "rope": {"harvested_task": "rope_apply", "component_by_role": {}},
    "swiglu": {
        "harvested_task": "silu",
        "component_by_role": {"act_fn": ("mlp_activation", "tokens")},
        "supporting_tasks": {
            "mlp": {"task": "mlp", "roles": ["mlp"]},
            "gate_up_projection": {"task": "linear", "roles": ["gate_proj", "up_proj"]},
        },
    },
    "causal_gqa_attention": {
        "harvested_task": "sdpa",
        "component_by_role": {"scaled_dot_product_attention": ("causal_gqa_sdpa", "batch_seq")},
    },
    "cross_entropy": {
        "harvested_task": "cross_entropy",
        "component_by_role": {"fixed_cross_entropy": ("logits", "tokens")},
        "supporting_tasks": {
            "causal_wrapper": {"task": "causal_cross_entropy", "roles": ["loss_function"]},
        },
    },
}


def _rows(shape: list[int], width: int) -> int:
    """Flatten every leading axis into rows, given the reduced width."""
    total = 1
    for size in shape:
        total *= size
    if total % width:
        raise SnapshotError(f"shape {shape} does not divide into rows of {width}")
    return total // width


def _free_for(kind: str, batch: int, seq: int) -> dict[str, int]:
    if kind == "tokens":
        return {"tokens": batch * seq}
    if kind == "batch_seq":
        return {"batch": batch, "seq": seq}
    raise SnapshotError(f"unknown provenance free-variable kind {kind!r}")


def _level1_dims(task: str, record: dict[str, Any], entries: list[dict[str, Any]]):
    """Generic dims for one harvested configuration, and the role it keys on.

    Returns a list of ``(dims, role_key, extra)`` -- a list because one harvested
    record can be more than one generic invocation: ``apply_rotary_pos_emb``
    rotates the queries *and* the keys in a single call, at different head
    counts, and they are two RoPE workloads.
    """
    attrs = record["attrs"]
    if task == "linear":
        width = attrs["in_features"]
        dims = {
            "M": _rows(entries[0]["shape"], width),
            "K": width,
            "N": attrs["out_features"],
        }
        return [(dims, sorted(record["roles"])[0], {"bias": attrs["bias"]})]
    if task == "rms_norm":
        width = attrs["normalized_size"]
        dims = {"rows": _rows(entries[0]["shape"], width), "hidden": width}
        return [(dims, sorted(record["roles"])[0], {"eps": attrs["eps"]})]
    if task == "silu":
        width = entries[0]["shape"][-1]
        dims = {"rows": _rows(entries[0]["shape"], width), "cols": width}
        return [(dims, sorted(record["roles"])[0], {})]
    if task == "cross_entropy":
        shape = entries[0]["shape"]
        dims = {"rows": shape[0], "cols": shape[1]}
        return [(dims, sorted(record["roles"])[0], dict(attrs))]
    if task == "sdpa":
        q, k, _v = entries[:3]
        dims = {
            "B": q["shape"][0],
            "HQ": q["shape"][1],
            "HK": k["shape"][1],
            "T": q["shape"][2],
            "D": q["shape"][3],
        }
        return [(dims, sorted(record["roles"])[0], dict(attrs))]
    if task == "rope_apply":
        out = []
        for label, entry in zip(("q", "k"), entries[:2]):
            batch, heads, tokens, head_dim = entry["shape"]
            out.append(
                (
                    {"B": batch, "n_heads": heads, "T": tokens, "head_dim": head_dim},
                    label,
                    {"rotated": label, "unsqueeze_dim": attrs["unsqueeze_dim"]},
                )
            )
        return out
    raise SnapshotError(f"no Level-1 dim rule for harvested task {task!r}")


def _extract_level1(
    manifest: dict[str, Any], arch: dict[str, Any], batch: int, seq: int,
    level1_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Every generic Level-1 configuration the canonical step runs.

    One entry per *deduplicated* harvested configuration: the harvest already
    collapsed structurally identical invocations, and this keeps every role and
    the pooled frequency rather than re-splitting them.
    """
    mapped: dict[str, Any] = {}
    for generic, spec in level1_sources.items():
        harvested = spec["harvested_task"]
        records = [c for c in manifest["configurations"] if c["task"] == harvested]
        if not records:
            raise SnapshotError(f"no {harvested!r} configuration to map onto {generic!r}")
        configurations = []
        for record in sorted(records, key=lambda r: (-r["frequency"], r["config_id"])):
            entries = _tensor_entries(record["inputs"])
            for dims, role_key, extra in _level1_dims(harvested, record, entries):
                component = spec["component_by_role"].get(role_key)
                if generic == "rope":
                    component = (
                        ("rope", "batch_seq") if role_key == "q" else ("rope_kv", "batch_seq")
                    )
                if component is None:
                    raise SnapshotError(
                        f"{generic}: no provenance component for role {role_key!r}; "
                        f"roles were {sorted(record['roles'])}"
                    )
                configurations.append(
                    {
                        "config_id": record["config_id"],
                        "harvested_task": harvested,
                        "roles": sorted(record["roles"]),
                        "frequency": record["frequency"],
                        "layer_indices": list(record["layer_indices"]),
                        "module_paths": list(record["module_paths"]),
                        "dims": dims,
                        "dtype": entries[0]["dtype"],
                        "inputs": entries,
                        "outputs": _tensor_entries(record["outputs"]),
                        "attrs": extra,
                        "provenance": {
                            "component": component[0],
                            "free": _free_for(component[1], batch, seq),
                        },
                    }
                )
        entry: dict[str, Any] = {
            "harvested_task": harvested,
            "configurations": configurations,
            "total_frequency": sum(c["frequency"] for c in records),
        }
        if "supporting_tasks" in spec:
            entry["supporting"] = {
                key: _supporting(manifest, sub)
                for key, sub in spec["supporting_tasks"].items()
            }
        mapped[generic] = entry
    return mapped


def _derive_fusion_sites(record: dict[str, Any], arch: dict[str, Any]) -> dict[str, Any]:
    """How many residual-add-then-RMSNorm sites one step contains.

    Derived from the layer count, then reconciled with what was observed. Each
    decoder layer does two residual adds, and each is followed by an RMSNorm --
    but the norm after the *MLP* add belongs to the next layer, or, for the last
    layer, to ``model.norm``. Layer 0's ``input_layernorm`` therefore has no
    preceding decoder residual add and is not a fusion site, which is exactly
    why this is one fewer than the residual-width RMSNorm invocations the
    harvest counted. The reconciliation is asserted rather than asserted-in-
    prose: if the two ever stop agreeing, extraction fails.
    """
    layers = arch["num_hidden_layers"]
    sites = {
        "attention_residual_then_post_attention_layernorm": layers,
        "mlp_residual_then_next_layer_input_layernorm": layers - 1,
        "final_mlp_residual_then_model_norm": 1,
        "excluded_layer0_input_layernorm": 1,
    }
    sites["total"] = (
        sites["attention_residual_then_post_attention_layernorm"]
        + sites["mlp_residual_then_next_layer_input_layernorm"]
        + sites["final_mlp_residual_then_model_norm"]
    )
    observed = record["frequency"]
    if sites["total"] + sites["excluded_layer0_input_layernorm"] != observed:
        raise SnapshotError(
            f"derived {sites['total']} fusion sites plus 1 excluded "
            f"input_layernorm, which should account for all {observed} observed "
            "residual-width RMSNorm invocations but does not"
        )
    sites["observed_rms_norm_invocations"] = observed
    sites["directly_verified_invocations"] = 1
    sites["directly_verified_note"] = (
        "one canonical invocation is verified against the model, inside layer "
        "14; the other 55 are the same configuration by deduplication, not by "
        "direct comparison"
    )
    return sites


def _tensor_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape, dtype and stride for every tensor in an argument list.

    Stride matters here and nowhere else in the snapshot: attention receives q,
    k and v already transposed into head-major order, and a task that generated
    contiguous substitutes would benchmark a different memory access pattern.
    """
    return [
        {
            "shape": list(entry["shape"]),
            "dtype": entry["dtype"],
            "stride": list(entry["stride"]),
            "contiguous": entry["contiguous"],
        }
        for entry in entries
        if entry.get("kind") == "tensor"
    ]


def _supporting(manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    record = _find(manifest, spec["task"], spec["roles"])
    return {
        "config_id": record["config_id"],
        "task": record["task"],
        "roles": sorted(record["roles"]),
        "frequency": record["frequency"],
        "input_shapes": _tensor_entries(record["inputs"]),
        "output_shapes": _tensor_entries(record["outputs"]),
        "params": {
            name: {"shape": meta["shape"], "dtype": meta["dtype"]}
            for name, meta in sorted(record.get("params", {}).items())
        },
        "attrs": record["attrs"],
    }


def extract(
    manifest: dict[str, Any],
    *,
    layer_index: int,
    task_sources: dict[str, dict[str, Any]],
    level1_sources: dict[str, dict[str, Any]] = DECODER_LEVEL1,
    schema_version: str,
) -> dict[str, Any]:
    """Build the snapshot from a full harvest manifest."""
    arch = manifest["workload"]["config"]
    layer_events = [
        event
        for event in manifest["events"]
        if event["task"] == "decoder_layer" and event["layer_index"] == layer_index
    ]
    if len(layer_events) != 1:
        raise SnapshotError(
            f"expected exactly one decoder_layer event at index {layer_index}, "
            f"found {len(layer_events)}"
        )

    tasks: dict[str, Any] = {}
    for name, spec in task_sources.items():
        record = _find(manifest, spec["task"], spec["roles"])
        tasks[name] = {
            "config_id": record["config_id"],
            "harvested_task": record["task"],
            "roles": sorted(record["roles"]),
            "frequency": record["frequency"],
            "module_paths": list(record["module_paths"]),
            "layer_indices": list(record["layer_indices"]),
            "input_shapes": _tensor_entries(record["inputs"]),
            "output_shapes": _tensor_entries(record["outputs"]),
            "dtype": _tensor_entries(record["inputs"])[0]["dtype"],
            "params": {
                name: {"shape": meta["shape"], "dtype": meta["dtype"]}
                for name, meta in sorted(record.get("params", {}).items())
            },
            "attrs": record["attrs"],
            "supporting": {
                key: _supporting(manifest, sub)
                for key, sub in spec["supporting_tasks"].items()
            },
        }
        if spec.get("derived") == "fusion_sites":
            tasks[name]["fusion_sites"] = _derive_fusion_sites(record, arch)

    payload = {
        "schema_version": schema_version,
        "workload_id": manifest["workload_id"],
        "workload_hash": manifest["workload"]["workload_hash"],
        "config_hash": manifest["config_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_schema": manifest["schema_version"],
        "model": {
            "name": manifest["workload"]["model_name"],
            "hidden_size": arch["hidden_size"],
            "intermediate_size": arch["intermediate_size"],
            "num_hidden_layers": arch["num_hidden_layers"],
            "hidden_act": arch["hidden_act"],
            "num_attention_heads": arch["num_attention_heads"],
            "num_key_value_heads": arch["num_key_value_heads"],
            "head_dim": arch["head_dim"],
        },
        "batch_size": manifest["workload"]["batch_size"],
        "seq_len": manifest["workload"]["seq_len"],
        "dtype": manifest["workload"]["dtype"],
        "level1": _extract_level1(
            manifest,
            arch,
            manifest["workload"]["batch_size"],
            manifest["workload"]["seq_len"],
            level1_sources,
        ),
        "representative_layer": {
            "layer_index": layer_index,
            "module_path": layer_events[0]["module_path"],
            "event_ordinal": layer_events[0]["ordinal"],
        },
        "tasks": tasks,
    }
    payload["snapshot_hash"] = snapshot_hash(payload)
    return payload


def diff(frozen: dict[str, Any], derived: dict[str, Any]) -> list[str]:
    """Field-by-field disagreements, so a failure names what moved."""
    problems: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    problems.append(f"{path}.{key}: only in the derived snapshot")
                elif key not in b:
                    problems.append(f"{path}.{key}: only in the frozen snapshot")
                else:
                    walk(a[key], b[key], f"{path}.{key}")
        elif a != b:
            problems.append(f"{path}: frozen {a!r} != derived {b!r}")

    walk(frozen, derived, "$")
    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
