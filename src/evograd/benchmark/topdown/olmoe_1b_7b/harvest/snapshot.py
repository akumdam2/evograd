"""The tracked slice of the OLMoE whole-model harvest that tasks may read.

Same contract as every workload's snapshot (:mod:`...common.snapshot`): a small,
hashed JSON file, a frozen extract of a harvest that ``--validate`` re-derives.
What is OLMoE's is here:

* :data:`TASK_SOURCES` -- which harvested boundaries each Level-2 site is read
  from;
* :data:`LEVEL1_SOURCES` -- which generic primitive each observed role maps
  onto. Two things differ from a dense decoder: the query and key norms are
  full-width rows (``attn_q_norm`` / ``attn_k_norm``, one row per token), and
  the SiLU comes from the experts (role ``experts_act_fn``) on the packed
  ``[tokens * top_k, intermediate]`` tensor the grouped GEMM activates once per
  layer. The expert GEMMs are not ``nn.Linear`` modules and the router is not
  one either, so ``linear_no_bias`` sees only the attention projections and
  the head;
* :data:`REPRESENTATIVE_LAYERS` -- one, layer 8;
* :func:`check_declared_dims` -- every site's *declared* (config-derived) dims
  against the shapes the harvest saw.

This is a **whole-model** harvest: the snapshot's ``frequency`` fields and each
task's ``observed`` entry count what one canonical training step actually
invoked (16 of every site), read from the events, never derived.

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
from ..architecture import BLOCK_KIND, REPRESENTATIVE_LAYERS

SCHEMA_VERSION = "evograd-olmoe-snapshot/1"
SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")

#: The architecture fields the snapshot's ``model`` section records.
MODEL_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "hidden_act",
    "num_attention_heads", "num_key_value_heads", "num_experts", "num_experts_per_tok",
    "norm_topk_prob", "vocab_size", "tie_word_embeddings", "rms_norm_eps", "rope_theta",
    "router_aux_loss_coef", "output_router_logits",
)

#: Every observed role, mapped onto a generic primitive. A role missing here
#: fails the extraction by name. Projections of equal shape collapse into one
#: configuration keyed on the first role alphabetically -- with multi-head
#: attention at ``q_out == kv_out == hidden`` all four attention projections
#: are one configuration -- so every role names the component whose dims it
#: would have alone, and the collapsed configurations re-derive either way.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = {
    "linear_no_bias": {
        "harvested_task": "linear",
        "component_by_role": {
            "q_proj": ("attn_qkv", "tokens"),
            "k_proj": ("attn_kv_proj", "tokens"),
            "v_proj": ("attn_kv_proj", "tokens"),
            "o_proj": ("attn_out_proj", "tokens"),
            "lm_head": ("lm_head", "tokens"),
        },
    },
    "rmsnorm": {
        "harvested_task": "rms_norm",
        "component_by_role": {
            "input_layernorm": ("residual_norm", "tokens"),
            "post_attention_layernorm": ("residual_norm", "tokens"),
            "norm": ("residual_norm", "tokens"),
            # Full-width, one row per token: OLMoE normalizes the whole
            # projection before the head split, unlike Qwen3's per-head norms.
            "q_norm": ("attn_q_norm", "tokens"),
            "k_norm": ("attn_k_norm", "tokens"),
        },
    },
    "rope": {"harvested_task": "rope_apply", "component_by_role": {}},
    "causal_gqa_attention": {
        "harvested_task": "sdpa",
        "component_by_role": {"scaled_dot_product_attention": ("causal_gqa_sdpa", "batch_seq")},
    },
    "swiglu": {
        "harvested_task": "silu",
        "component_by_role": {"experts_act_fn": ("moe_activation", "tokens")},
        "supporting_tasks": {
            "experts": {"task": "experts", "roles": ["experts"]},
            "router": {"task": "router", "roles": ["gate"]},
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

_ATTN = {"task": "attention", "roles": ["self_attn"]}
_MOE = {"task": "moe", "roles": ["mlp"]}

#: Each Level-2 task, and the harvested boundaries it is read from.
TASK_SOURCES: dict[str, dict[str, Any]] = {
    "olmoe_qkv_norm_rope": {
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
    "olmoe_attention": {
        "task": "sdpa", "roles": ["scaled_dot_product_attention"],
        "supporting_tasks": {
            "output_projection": {"task": "linear", "roles": ["o_proj"]},
            "enclosing": _ATTN,
        },
    },
    "olmoe_moe_experts": {
        "task": "experts", "roles": ["experts"],
        "supporting_tasks": {
            "router": {"task": "router", "roles": ["gate"]},
            "activation": {"task": "silu", "roles": ["experts_act_fn"]},
            "enclosing": _MOE,
        },
    },
    "olmoe_residual_rmsnorm": {
        "task": "rms_norm", "roles": ["post_attention_layernorm"],
        "supporting_tasks": {
            "consumer": _MOE,
        },
    },
}

#: Which observed events count one invocation of each site: ``(task, role)``.
SITE_EVENTS: dict[str, tuple[str, str]] = {
    "olmoe_qkv_norm_rope": ("rope_apply", "apply_rotary_pos_emb"),
    "olmoe_attention": ("sdpa", "scaled_dot_product_attention"),
    "olmoe_moe_experts": ("experts", "experts"),
    "olmoe_residual_rmsnorm": ("rms_norm", "post_attention_layernorm"),
}


def observed_counts(manifest: dict[str, Any]):
    """``(task_name, record) -> {...}``: one site's whole-model invocations, from the events."""

    def count(name: str, record: dict[str, Any]) -> dict[str, Any]:
        task, role = SITE_EVENTS[name]
        layers = [e["layer_index"] for e in manifest["events"]
                  if e["task"] == task and e["role"] == role]
        return {
            "kind": "observed",
            "whole_model_invocations": len(layers),
            "by_layer_type": {BLOCK_KIND: len(layers)},
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

    entry = tasks["olmoe_qkv_norm_rope"]
    q, k = entry["output_shapes"][0]["shape"], entry["output_shapes"][1]["shape"]
    sup = entry["supporting"]
    qo, hidden = sup["q_projection"]["params"]["weight"]["shape"]
    kvo = sup["kv_projection"]["params"]["weight"]["shape"][0]
    out["qkv_norm_rope"] = {"B": q[0], "T": q[2], "H": hidden, "HQ": q[1], "HK": k[1],
                            "D": q[3], "QO": qo, "KVO": kvo}

    entry = tasks["olmoe_attention"]
    q, k = _shape(entry, 0), _shape(entry, 1)
    hidden, qo = entry["supporting"]["output_projection"]["params"]["weight"]["shape"]
    out["attention"] = {"B": q[0], "T": q[2], "HQ": q[1], "HK": k[1], "D": q[3],
                        "QO": qo, "H": hidden}

    entry = tasks["olmoe_moe_experts"]
    tokens, hidden = _shape(entry, 0)
    top_k = _shape(entry, 1)[1]
    experts, gate_up, _ = entry["params"]["gate_up_proj"]["shape"]
    _, _, inter = entry["params"]["down_proj"]["shape"]
    out["moe_experts"] = {"T": tokens, "H": hidden, "E": experts, "G": gate_up, "I": inter,
                          "K": top_k}

    batch, tokens, hidden = _shape(tasks["olmoe_residual_rmsnorm"], 0)
    out["residual_rmsnorm"] = {"B": batch, "T": tokens, "H": hidden}
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
        prog="python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.snapshot",
        description="Regenerate or validate the tracked OLMoE snapshot.")
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


__all__ = ["LEVEL1_SOURCES", "MODEL_FIELDS", "REPRESENTATIVE_LAYERS", "SCHEMA_VERSION",
           "SITE_EVENTS", "SNAPSHOT_PATH", "SnapshotError", "TASK_SOURCES",
           "check_declared_dims", "diff", "extract", "load", "main", "observed_counts",
           "site_dims", "task"]


if __name__ == "__main__":
    raise SystemExit(main())
