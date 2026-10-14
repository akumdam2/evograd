"""The tracked slice of the Qwen3.5 whole-model harvest that tasks may read.

Same contract as every workload's snapshot (:mod:`...common.snapshot`): a small,
hashed JSON file, a frozen extract of a harvest that ``--validate`` re-derives.
What is Qwen3.5's is here:

* :data:`TASK_SOURCES` -- which harvested boundaries each Level-2 site is read
  from;
* :data:`LEVEL1_SOURCES` and :data:`DIM_RULES` -- which generic primitive each
  observed role maps onto, and how the operators the shared rules do not know
  (a partial RoPE, a causal convolution, the GDN gating) turn into dims;
* :data:`REPRESENTATIVE_LAYERS` -- two, one per block kind (14 and 15);
* :func:`check_declared_dims` -- every site's *declared* (config-derived) dims
  against the shapes the harvest saw.

This is a **whole-model** harvest: the
snapshot's ``frequency`` fields and each task's ``observed`` entry count what
one canonical training step actually invoked (24 GDN layers, 8 full-attention
layers, 32 MLPs and post-attention norms), read from the events, never derived.

This module imports json, hashlib and pathlib, and nothing that needs torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...common.snapshot import SnapshotError, snapshot_hash  # noqa: F401  (re-export)
from ...common.snapshot import load as _load
from ...common.snapshot import task as _task
from ...common.snapshot_extract import diff
from ...common import snapshot_extract as _extract
from ..architecture import REPRESENTATIVE_LAYERS

SCHEMA_VERSION = "evograd-qwen3_5-snapshot/1"
SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")

#: The architecture fields the snapshot's ``model`` section records.
MODEL_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "full_attention_interval",
    "hidden_act", "num_attention_heads", "num_key_value_heads", "head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size", "tie_word_embeddings",
)


def _prod(shape) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


# --------------------------------------------------------------------------
# dim rules for the harvested tasks the shared rules do not cover
# --------------------------------------------------------------------------


def _partial_rope_rule(record, entries):
    rotary = entries[2]["shape"][-1]
    out = []
    for label, entry in zip(("q", "k"), entries[:2]):
        batch, heads, tokens, head_dim = entry["shape"]
        out.append(({"B": batch, "H": heads, "T": tokens, "D": head_dim, "R": rotary}, label,
                    {"rotated": label, "unsqueeze_dim": record["attrs"].get("unsqueeze_dim", 1)}))
    return out


def _conv_rule(record, entries):
    batch, channels, tokens = entries[0]["shape"]
    taps = entries[1]["shape"][-1]
    return [({"B": batch, "C": channels, "T": tokens, "K": taps}, sorted(record["roles"])[0],
             dict(record["attrs"]))]


def _l2norm_rule(record, entries):
    shape = entries[0]["shape"]
    return [({"rows": _prod(shape[:-1]), "cols": shape[-1]}, sorted(record["roles"])[0],
             dict(record["attrs"]))]


def _gated_norm_rule(record, entries):
    shape = entries[0]["shape"]
    return [({"rows": _prod(shape[:-1]), "hidden": shape[-1]}, sorted(record["roles"])[0],
             {"eps": record["attrs"]["eps"]})]


def _gating_rule(record, entries):
    """g and beta as the delta rule receives them: ``[B, T, HV]``.

    The gating's ``softplus`` is observed too (supporting provenance), but that
    call sees the float32 promotion, not the operator's inputs; the delta-rule
    call carries the gate tensors, so the case is cut from there.
    """
    batch, tokens, heads = entries[4]["shape"]
    return [({"rows": batch * tokens, "heads": heads}, "gdn_gating",
             {"source": "g and beta as the gated delta rule receives them"})]


DIM_RULES = {
    "rope_apply": _partial_rope_rule,
    "causal_conv1d": _conv_rule,
    "l2norm": _l2norm_rule,
    "rms_norm_gated": _gated_norm_rule,
    "gated_delta_rule": _gating_rule,
}

#: Every observed role, mapped onto a generic primitive. A role missing here
#: fails the extraction by name. Projections of equal shape collapse into one
#: configuration keyed on the first role alphabetically (``in_proj_qkv`` with
#: ``q_proj``, ``o_proj`` with ``out_proj``, ``in_proj_a`` with ``in_proj_b``,
#: ``k_proj`` with ``v_proj``, ``gate_proj`` with ``up_proj``), so every role
#: names the component whose dims it would have alone.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = {
    "linear_no_bias": {
        "harvested_task": "linear",
        "component_by_role": {
            "in_proj_qkv": ("gdn_in_qkv", "tokens"),
            "in_proj_z": ("gdn_in_z", "tokens"),
            "in_proj_a": ("gdn_in_ba", "tokens"),
            "in_proj_b": ("gdn_in_ba", "tokens"),
            "out_proj": ("gdn_out_proj", "tokens"),
            "q_proj": ("attn_q_gate_proj", "tokens"),
            "k_proj": ("attn_kv_proj", "tokens"),
            "v_proj": ("attn_kv_proj", "tokens"),
            "o_proj": ("attn_out_proj", "tokens"),
            "gate_proj": ("mlp_up", "tokens"),
            "up_proj": ("mlp_up", "tokens"),
            "down_proj": ("mlp_down", "tokens"),
            "lm_head": ("lm_head", "tokens"),
        },
    },
    "gemma_rmsnorm": {
        "harvested_task": "rms_norm",
        "component_by_role": {
            "input_layernorm": ("residual_norm", "tokens"),
            "post_attention_layernorm": ("residual_norm", "tokens"),
            "norm": ("residual_norm", "tokens"),
            "q_norm": ("attn_q_norm", "batch_seq"),
            "k_norm": ("attn_k_norm", "batch_seq"),
        },
    },
    "gated_rmsnorm": {
        "harvested_task": "rms_norm_gated",
        "component_by_role": {"norm": ("gdn_gated_norm", "tokens")},
    },
    "causal_conv1d": {
        "harvested_task": "causal_conv1d",
        "component_by_role": {"causal_conv1d_fn": ("gdn_conv1d", "batch_seq")},
    },
    "l2norm": {
        "harvested_task": "l2norm",
        "component_by_role": {"l2norm": ("gdn_qk_l2norm", "batch_seq")},
    },
    "gdn_gating": {
        "harvested_task": "gated_delta_rule",
        "component_by_role": {"gdn_gating": ("gdn_gating", "tokens")},
        "supporting_tasks": {
            "softplus": {"task": "gdn_gating", "roles": ["gdn_gating"]},
        },
    },
    "partial_rope": {
        "harvested_task": "rope_apply",
        "component_by_role": {"q": ("partial_rope", "batch_seq"),
                              "k": ("partial_rope_kv", "batch_seq")},
    },
    "causal_gqa_attention": {
        "harvested_task": "sdpa",
        "component_by_role": {"scaled_dot_product_attention": ("causal_gqa_sdpa", "batch_seq")},
    },
    "swiglu": {
        "harvested_task": "silu",
        "component_by_role": {"act_fn": ("mlp_activation", "tokens")},
        "supporting_tasks": {
            "mlp": {"task": "mlp", "roles": ["mlp"]},
            "gate_up_projection": {"task": "linear", "roles": ["gate_proj", "up_proj"]},
        },
    },
    "cross_entropy": {
        "harvested_task": "cross_entropy",
        "component_by_role": {"fixed_cross_entropy": ("logits", "tokens")},
        "supporting_tasks": {
            "causal_wrapper": {"task": "causal_cross_entropy", "roles": ["loss_function"]},
        },
    },
}

_GDN = {"task": "linear_attention", "roles": ["linear_attn"]}
_ATTN = {"task": "attention", "roles": ["self_attn"]}

#: Each Level-2 task, and the harvested boundaries it is read from.
TASK_SOURCES: dict[str, dict[str, Any]] = {
    "qwen3_5_gdn_in_proj": {
        # The delta rule's arguments *are* this site's q, k, v, beta and g.
        "task": "gated_delta_rule", "roles": ["chunk_gated_delta_rule"],
        "supporting_tasks": {
            "qkv_projection": {"task": "linear", "roles": ["in_proj_qkv"]},
            "z_projection": {"task": "linear", "roles": ["in_proj_z"]},
            "ba_projection": {"task": "linear", "roles": ["in_proj_a", "in_proj_b"]},
            "conv": {"task": "causal_conv1d", "roles": ["causal_conv1d_fn"]},
            "gating": {"task": "gdn_gating", "roles": ["gdn_gating"]},
            "z_consumer": {"task": "rms_norm_gated", "roles": ["norm"]},
            "enclosing": _GDN,
        },
    },
    "qwen3_5_gated_delta_rule": {
        "task": "gated_delta_rule", "roles": ["chunk_gated_delta_rule"],
        "supporting_tasks": {
            "l2norm": {"task": "l2norm", "roles": ["l2norm"]},
            "enclosing": _GDN,
        },
    },
    "qwen3_5_gdn_norm_out": {
        "task": "rms_norm_gated", "roles": ["norm"],
        "supporting_tasks": {
            "output_projection": {"task": "linear", "roles": ["out_proj"]},
            "enclosing": _GDN,
        },
    },
    "qwen3_5_qkv_norm_partial_rope": {
        "task": "rope_apply", "roles": ["apply_rotary_pos_emb"],
        "supporting_tasks": {
            "q_projection": {"task": "linear", "roles": ["q_proj"]},
            "kv_projection": {"task": "linear", "roles": ["k_proj", "v_proj"]},
            "q_norm": {"task": "rms_norm", "roles": ["q_norm"]},
            "k_norm": {"task": "rms_norm", "roles": ["k_norm"]},
            "consumer": {"task": "sdpa", "roles": ["scaled_dot_product_attention"]},
            "enclosing": _ATTN,
        },
    },
    "qwen3_5_gated_attention": {
        "task": "sdpa", "roles": ["scaled_dot_product_attention"],
        "supporting_tasks": {
            "output_projection": {"task": "linear", "roles": ["o_proj"]},
            "enclosing": _ATTN,
        },
    },
    "qwen3_5_residual_rmsnorm": {
        "task": "rms_norm", "roles": ["post_attention_layernorm"],
        "supporting_tasks": {
            "consumer": {"task": "mlp", "roles": ["mlp"]},
        },
    },
    "qwen3_5_swiglu_mlp": {
        "task": "mlp", "roles": ["mlp"],
        "supporting_tasks": {
            "gate_up_projection": {"task": "linear", "roles": ["gate_proj", "up_proj"]},
            "down_projection": {"task": "linear", "roles": ["down_proj"]},
            "activation": {"task": "silu", "roles": ["act_fn"]},
        },
    },
}

#: Which observed events count one invocation of each site: ``(task, role)``.
#: One per call of the site in a training step -- the site is one call of its
#: anchor boundary, wherever in the stack that happened.
SITE_EVENTS: dict[str, tuple[str, str]] = {
    "qwen3_5_gdn_in_proj": ("gated_delta_rule", "chunk_gated_delta_rule"),
    "qwen3_5_gated_delta_rule": ("gated_delta_rule", "chunk_gated_delta_rule"),
    "qwen3_5_gdn_norm_out": ("rms_norm_gated", "norm"),
    "qwen3_5_qkv_norm_partial_rope": ("rope_apply", "apply_rotary_pos_emb"),
    "qwen3_5_gated_attention": ("sdpa", "scaled_dot_product_attention"),
    "qwen3_5_residual_rmsnorm": ("rms_norm", "post_attention_layernorm"),
    "qwen3_5_swiglu_mlp": ("mlp", "mlp"),
}


def observed_counts(manifest: dict[str, Any]):
    """``(task_name, record) -> {...}``: one site's whole-model invocations, from the events."""
    by_layer = {}
    for event in manifest["events"]:
        if event["task"] == "decoder_layer":
            by_layer[event["layer_index"]] = event["module_path"]
    kinds = manifest["workload"]["config"].get("layer_types") or []

    def count(name: str, record: dict[str, Any]) -> dict[str, Any]:
        task, role = SITE_EVENTS[name]
        layers = [e["layer_index"] for e in manifest["events"]
                  if e["task"] == task and e["role"] == role]
        by_kind: dict[str, int] = {}
        for index in layers:
            kind = kinds[index] if index is not None and index < len(kinds) else "unknown"
            by_kind[kind] = by_kind.get(kind, 0) + 1
        return {
            "kind": "observed",
            "whole_model_invocations": len(layers),
            "by_layer_type": dict(sorted(by_kind.items())),
            "layer_indices": sorted(i for i in layers if i is not None),
            "anchor": {"task": task, "role": role},
            "note": "counted from the harvest's events: one canonical training step's forward",
        }

    return count


def load(path: Path | None = None) -> dict[str, Any]:
    return _load(Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


def task(name: str, path: Path | None = None) -> dict[str, Any]:
    return _task(name, Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


def extract(manifest: dict[str, Any], *,
            representative_layers: dict[str, int] | None = None) -> dict[str, Any]:
    return _extract.extract_blocks(
        manifest,
        representative_layers=dict(representative_layers or REPRESENTATIVE_LAYERS),
        task_sources=TASK_SOURCES,
        level1_sources=LEVEL1_SOURCES,
        schema_version=SCHEMA_VERSION,
        model_fields=MODEL_FIELDS,
        dim_rules=DIM_RULES,
        observed=observed_counts(manifest),
    )


# --------------------------------------------------------------------------
# declared dims against harvested shapes
# --------------------------------------------------------------------------


def _shape(entry: dict[str, Any], index: int) -> list[int]:
    return list(entry["input_shapes"][index]["shape"])


def site_dims(snapshot: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Every Level-2 site's dims, as the harvested shapes state them."""
    tasks = snapshot["tasks"]
    out: dict[str, dict[str, int]] = {}

    entry = tasks["qwen3_5_gdn_in_proj"]
    batch, tokens, hv, d = _shape(entry, 0)
    sup = entry["supporting"]
    qkv = sup["qkv_projection"]["params"]["weight"]["shape"]
    z = sup["z_projection"]["params"]["weight"]["shape"]
    conv_w = sup["conv"]["input_shapes"][1]["shape"]
    out["gdn_in_proj"] = {"B": batch, "T": tokens, "H": qkv[1], "C": qkv[0], "VD": z[0],
                          "HV": hv, "K": conv_w[1], "D": d}

    batch, tokens, hv, d = _shape(tasks["qwen3_5_gated_delta_rule"], 0)
    out["gated_delta_rule"] = {"B": batch, "T": tokens, "HV": hv, "D": d}

    entry = tasks["qwen3_5_gdn_norm_out"]
    proj = entry["supporting"]["output_projection"]
    batch, tokens, vd = proj["input_shapes"][0]["shape"]
    hidden, _ = proj["params"]["weight"]["shape"]
    d = _shape(entry, 0)[-1]
    out["gdn_norm_out"] = {"B": batch, "T": tokens, "HV": vd // d, "D": d, "VD": vd, "H": hidden}

    entry = tasks["qwen3_5_qkv_norm_partial_rope"]
    q, k = entry["output_shapes"][0]["shape"], entry["output_shapes"][1]["shape"]
    rotary = _shape(entry, 2)[-1]
    sup = entry["supporting"]
    qg, hidden = sup["q_projection"]["params"]["weight"]["shape"]
    kvo = sup["kv_projection"]["params"]["weight"]["shape"][0]
    out["qkv_norm_partial_rope"] = {"B": q[0], "T": q[2], "H": hidden, "HQ": q[1], "HK": k[1],
                                    "D": q[3], "R": rotary, "QG": qg, "KVO": kvo, "QO": qg // 2}

    entry = tasks["qwen3_5_gated_attention"]
    q, k = _shape(entry, 0), _shape(entry, 1)
    hidden, qo = entry["supporting"]["output_projection"]["params"]["weight"]["shape"]
    out["gated_attention"] = {"B": q[0], "T": q[2], "HQ": q[1], "HK": k[1], "D": q[3],
                              "QO": qo, "H": hidden}

    batch, tokens, hidden = _shape(tasks["qwen3_5_residual_rmsnorm"], 0)
    out["residual_rmsnorm"] = {"B": batch, "T": tokens, "H": hidden}

    entry = tasks["qwen3_5_swiglu_mlp"]
    batch, tokens, hidden = _shape(entry, 0)
    out["swiglu_mlp"] = {"B": batch, "T": tokens, "H": hidden,
                         "I": int(entry["attrs"]["intermediate_size"])}
    return out


def check_declared_dims(snapshot: dict[str, Any], declared: dict[str, dict[str, int]]) -> list[str]:
    """Where a site's declared dims and its harvested shapes disagree."""
    observed = site_dims(snapshot)
    problems = []
    for site, dims in sorted(declared.items()):
        if observed.get(site) != dims:
            problems.append(f"{site}: declared {dims} != harvested {observed.get(site)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot",
        description="Regenerate or validate the tracked Qwen3.5 snapshot.")
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SNAPSHOT_PATH)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.harvest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"{args.harvest} not found", file=sys.stderr)
        return 2
    try:
        derived = extract(manifest)
    except SnapshotError as exc:
        print(f"extraction failed: {exc}", file=sys.stderr)
        return 1
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}\nsnapshot_hash {derived['snapshot_hash']}")
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
    print(f"snapshot matches {args.harvest}\nsnapshot_hash {frozen['snapshot_hash']}")
    return 0


__all__ = ["DIM_RULES", "LEVEL1_SOURCES", "MODEL_FIELDS", "REPRESENTATIVE_LAYERS",
           "SCHEMA_VERSION", "SITE_EVENTS", "SNAPSHOT_PATH", "SnapshotError", "TASK_SOURCES",
           "check_declared_dims", "diff", "extract", "load", "main", "observed_counts",
           "site_dims", "task"]


if __name__ == "__main__":
    raise SystemExit(main())
