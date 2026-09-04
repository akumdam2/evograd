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

**What is Qwen-specific, and what is not.** Reading a snapshot and verifying its
hash is the same operation for every workload and lives in
:mod:`..common.snapshot`. What stays here is the part that names things: which
harvested boundaries collapse into which task (:data:`TASK_SOURCES`), which
generic Level-1 operator each observed configuration maps onto
(:data:`LEVEL1_SOURCES`), and the extraction that turns a manifest into the
frozen file. Those are facts about Qwen3's decoder, not about snapshots.

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

from ...common.snapshot import SnapshotError, snapshot_hash
from ...common.snapshot import load as _load
from ...common.snapshot import task as _task
from ...common.snapshot_extract import DECODER_LEVEL1, diff
from ...common import snapshot_extract as _extract

__all__ = [
    "LEVEL1_SOURCES",
    "SCHEMA_VERSION",
    "SNAPSHOT_PATH",
    "SnapshotError",
    "TASK_SOURCES",
    "diff",
    "extract",
    "load",
    "main",
    "snapshot_hash",
    "task",
]

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

#: Which generic Level-1 operator each observed configuration maps onto. Qwen3
#: uses the shared decoder mapping unchanged: its roles are the standard
#: HuggingFace spellings, and ``q_norm``/``k_norm`` -- which only Qwen3 has --
#: are already in it.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = DECODER_LEVEL1

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


def load(path: Path | None = None) -> dict[str, Any]:
    """The frozen Qwen3 snapshot, with its hash and exact schema verified."""
    return _load(Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


def task(name: str, path: Path | None = None) -> dict[str, Any]:
    """One Level-2 task entry from the Qwen3 snapshot."""
    return _task(name, Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


# --------------------------------------------------------------------------
# extraction from a full harvest
# --------------------------------------------------------------------------


def extract(manifest: dict[str, Any], *, layer_index: int) -> dict[str, Any]:
    """Build the Qwen3 snapshot from a full harvest manifest.

    The derivation is shared; what is Qwen3's is :data:`TASK_SOURCES` -- which
    harvested boundaries collapse into which Level-2 task -- and the schema
    string the file declares.
    """
    return _extract.extract(
        manifest,
        layer_index=layer_index,
        task_sources=TASK_SOURCES,
        level1_sources=LEVEL1_SOURCES,
        schema_version=SCHEMA_VERSION,
    )


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
