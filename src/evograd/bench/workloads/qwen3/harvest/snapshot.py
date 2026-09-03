"""The tracked slice of the Qwen3 harvest that repository tasks are allowed to read.

A harvest manifest is a local result: it is a megabyte of transcript, it is not
version controlled, and it exists only on a machine that has run the canonical
workload. A declaration in ``evograd/ops/`` cannot depend on any of that -- it
has to import on a laptop with no GPU, no Transformers and no ``results/``
directory -- yet the whole point of these tasks is that their shapes come from a
real training step rather than from someone's judgement.

The snapshot is the reconciliation: a small, tracked, hashed JSON file carrying
only what a task needs to state its provenance -- the workload and manifest it
came from, the harvested configuration ids, how often each configuration ran,
which modules produced it, and the widths and dtype involved. Everything else
stays in the manifest.

It is a *frozen extract*, not a second source of truth. ``--validate`` re-derives
it from a full harvest and fails if the two disagree, so the snapshot cannot
quietly drift away from the run it claims to describe; ``--write`` regenerates
it. Neither is needed to use it.

This module imports json, hashlib and pathlib. Nothing else -- no torch, no
Transformers -- because everything that reads it must work without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

#: Bumped from /1 when a task whose boundary takes three tensors was added:
#: ``input_shape``/``output_shape`` became ``input_shapes``/``output_shapes``,
#: because an SDPA configuration has q, k and v and recording only the first
#: would have silently described a different operator.
#: Bumped from /3 when the Qwen projections were remapped from the bias-carrying
#: `linear` task onto `linear_no_bias`: every projection and the lm_head are
#: biasless, and a zero-valued bias is not an equivalent benchmark -- it adds a
#: broadcast add, a dbias reduction and a third gradient the model never
#: computes. /2 added the `level1` section itself.
SCHEMA_VERSION = "evograd-qwen3-snapshot/4"

SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")

#: Tasks the snapshot carries, and the harvested configuration each is derived
#: from. Adding a task here and running ``--write`` is how the next Level-2
#: extraction starts.
TASK_SOURCES: dict[str, dict[str, Any]] = {
    "qwen3_swiglu_mlp": {
        "task": "mlp",
        "roles": ["mlp"],
        "supporting_tasks": {
            "gate_up_projection": {"task": "linear", "roles": ["gate_proj", "up_proj"]},
            "down_projection": {"task": "linear", "roles": ["down_proj"]},
            "activation": {"task": "silu", "roles": ["act_fn"]},
        },
    },
    # The residual fusion site. Its primary configuration is the residual-width
    # RMSNorm, which the harvest deduplicated across `input_layernorm`,
    # `post_attention_layernorm` and the final `model.norm` -- 57 invocations.
    # The residual *add* is not an observed boundary (it is a bare `+`), so the
    # number of fusion sites is derived from the architecture and cross-checked
    # against that 57 by `_derive_fusion_sites`.
    "fused_add_rms_norm": {
        "task": "rms_norm",
        "roles": ["post_attention_layernorm"],
        "supporting_tasks": {
            "decoder_layer": {"task": "decoder_layer", "roles": ["decoder_layer"]},
        },
        "derived": "fusion_sites",
    },
    # The QKV task's primary configuration is the RoPE application: it is the
    # last step of the boundary and its two outputs *are* q and k, in the exact
    # layout the next task receives them. v never passes through RoPE, so its
    # observed shape and stride are sourced from where it is consumed -- the
    # SDPA call -- rather than invented.
    "qwen3_qkv_norm_rope": {
        "task": "rope_apply",
        "roles": ["apply_rotary_pos_emb"],
        "supporting_tasks": {
            "q_projection": {"task": "linear", "roles": ["q_proj"]},
            "kv_projection": {"task": "linear", "roles": ["k_proj", "v_proj"]},
            "q_norm": {"task": "rms_norm", "roles": ["q_norm"]},
            "k_norm": {"task": "rms_norm", "roles": ["k_norm"]},
            "consumer": {"task": "sdpa", "roles": ["scaled_dot_product_attention"]},
            "enclosing_attention": {"task": "attention", "roles": ["self_attn"]},
        },
    },
    # The attention task's primary configuration is the SDPA call itself; the
    # output projection is the second half of the same boundary and is carried
    # as a supporting configuration rather than a separate task.
    "qwen3_attention": {
        "task": "sdpa",
        "roles": ["scaled_dot_product_attention"],
        "supporting_tasks": {
            "output_projection": {"task": "linear", "roles": ["o_proj"]},
            "enclosing_attention": {"task": "attention", "roles": ["self_attn"]},
        },
    },
}


class SnapshotError(RuntimeError):
    """The snapshot is missing, malformed, or disagrees with a full harvest."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def snapshot_hash(payload: dict[str, Any]) -> str:
    """Deterministic hash over everything except the hash field itself."""
    body = {key: value for key, value in payload.items() if key != "snapshot_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def load(path: Path | None = None) -> dict[str, Any]:
    """The frozen snapshot, with its hash verified."""
    path = Path(path or SNAPSHOT_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - tracked in the repo
        raise SnapshotError(f"{path} is missing; it is tracked, not generated") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError(
            f"{path}: schema {payload.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    recomputed = snapshot_hash(payload)
    if payload.get("snapshot_hash") != recomputed:
        raise SnapshotError(
            f"{path}: snapshot hash mismatch (stored {payload.get('snapshot_hash')}, "
            f"recomputed {recomputed}). The file was edited by hand; regenerate it "
            "from a harvest with --write instead."
        )
    return payload


def task(name: str, path: Path | None = None) -> dict[str, Any]:
    payload = load(path)
    try:
        return payload["tasks"][name]
    except KeyError:
        raise SnapshotError(
            f"the snapshot has no task {name!r}; it carries {sorted(payload['tasks'])}"
        ) from None


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
#: dimension comes from the harvest, and ``tests/test_qwen3_level1`` proves the
#: two agree by re-deriving each shape from the published Qwen3-0.6B config.
#:
#: Two deliberate absences. Standalone ``softmax`` is not mapped: the model runs
#: fused SDPA and never materializes one, so a Qwen softmax case would be a
#: shape no step executes. And the observed ``silu`` record maps onto ``swiglu``
#: rather than a bare activation task, because the production pointwise boundary
#: is ``silu(gate) * up`` -- the SiLU record is kept as supporting provenance.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = {
    # Biasless: every Qwen3 projection and the lm_head set bias=False, which the
    # harvest records per configuration. They map onto `linear_no_bias`, not the
    # bias-carrying `linear` task.
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
    manifest: dict[str, Any], arch: dict[str, Any], batch: int, seq: int
) -> dict[str, Any]:
    """Every generic Level-1 configuration the canonical step runs.

    One entry per *deduplicated* harvested configuration: the harvest already
    collapsed structurally identical invocations, and this keeps every role and
    the pooled frequency rather than re-splitting them.
    """
    mapped: dict[str, Any] = {}
    for generic, spec in LEVEL1_SOURCES.items():
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


def extract(manifest: dict[str, Any], *, layer_index: int) -> dict[str, Any]:
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
    for name, spec in TASK_SOURCES.items():
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
        "schema_version": SCHEMA_VERSION,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.harvest.snapshot",
        description=(
            "Regenerate or validate the tracked Qwen3 workload snapshot from a "
            "full harvest manifest."
        ),
    )
    parser.add_argument(
        "--harvest",
        type=Path,
        default=Path("results/qwen3-level4/harvest.json"),
        help="the harvest manifest to derive from",
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--out", type=Path, default=SNAPSHOT_PATH)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="regenerate the snapshot")
    action.add_argument(
        "--validate",
        action="store_true",
        help="re-derive and compare against the tracked snapshot; exit 1 if they differ",
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.harvest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"{args.harvest} not found. The snapshot is tracked and usable without "
            "it; a full harvest is only needed to regenerate or validate.",
            file=sys.stderr,
        )
        return 2

    try:
        derived = extract(manifest, layer_index=args.layer)
    except SnapshotError as exc:
        print(f"extraction failed: {exc}", file=sys.stderr)
        return 1

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        print(f"snapshot_hash {derived['snapshot_hash']}")
        return 0

    try:
        frozen = load(args.out)
    except SnapshotError as exc:
        print(f"frozen snapshot unusable: {exc}", file=sys.stderr)
        return 1
    problems = diff(frozen, derived)
    if problems:
        print("the tracked snapshot disagrees with the harvest:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"snapshot matches {args.harvest}")
    print(f"snapshot_hash {frozen['snapshot_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
