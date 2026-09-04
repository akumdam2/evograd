"""The tracked slice of the Llama-3 harvest that repository tasks may read.

A harvest manifest is a local result: a megabyte of transcript, not version
controlled, existing only on a machine that has run the canonical workload. A
declaration in ``evograd/ops/`` cannot depend on any of that -- it has to import
on a laptop with no GPU -- yet the whole point of those tasks is that their
shapes come from a real training step rather than from someone's judgement.

The snapshot is the reconciliation. Reading and verifying one is shared; what is
here is the part that names things.

**This workload has no tracked snapshot yet.** ``snapshot.json`` is *derived*,
not authored: it comes out of a harvest, and nobody has run one for Llama-3.
Until someone does::

    python -m evograd.bench.workloads.llama3.harvest.harvest \\
        --out results/llama3-level4/harvest.json
    python -m evograd.bench.workloads.llama3.harvest.snapshot \\
        --harvest results/llama3-level4/harvest.json --write

Level-1 mapping is the shared decoder one unchanged: every role Llama presents
-- ``q_proj``, ``o_proj``, ``gate_proj``, ``down_proj``, ``lm_head``,
``input_layernorm``, ``act_fn``, SDPA, cross-entropy -- is a standard
HuggingFace spelling already in it. Llama has no ``q_norm``/``k_norm``, and
those entries simply never match.

:data:`TASK_SOURCES` carries only ``fused_add_rms_norm``, and that is deliberate
rather than unfinished. A Level-2 entry names a *declared operator*, and the
other three Qwen3 tasks have no Llama counterpart yet: ``qwen3_qkv_norm_rope``
bakes in the per-head normalization Llama does not have, so Llama needs its own
declaration before a task can point at one. Adding an entry here and running
``--write`` is how that starts.
"""

from __future__ import annotations

import argparse
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

SCHEMA_VERSION = "evograd-llama3-snapshot/1"

SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")

#: The shared decoder mapping, unchanged. See the module docstring.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = DECODER_LEVEL1

#: Level-2 tasks the snapshot carries. See the module docstring for why this is
#: one entry rather than four.
TASK_SOURCES: dict[str, dict[str, Any]] = {
    # The residual fusion site. Its primary configuration is the residual-width
    # RMSNorm, which the harvest deduplicates across `input_layernorm`,
    # `post_attention_layernorm` and the final `model.norm`. The residual *add*
    # is not an observed boundary (it is a bare `+`), so the number of fusion
    # sites is derived from the architecture and cross-checked against the
    # observed RMSNorm count.
    "fused_add_rms_norm": {
        "task": "rms_norm",
        "roles": ["post_attention_layernorm"],
        "supporting_tasks": {
            "decoder_layer": {"task": "decoder_layer", "roles": ["decoder_layer"]},
        },
        "derived": "fusion_sites",
    },
}


def load(path: Path | None = None) -> dict[str, Any]:
    """The frozen Llama-3 snapshot, with its hash and exact schema verified."""
    return _load(Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


def task(name: str, path: Path | None = None) -> dict[str, Any]:
    """One Level-2 task entry from the Llama-3 snapshot."""
    return _task(name, Path(path or SNAPSHOT_PATH), schema_version=SCHEMA_VERSION)


def extract(manifest: dict[str, Any], *, layer_index: int) -> dict[str, Any]:
    """Build the Llama-3 snapshot from a full harvest manifest."""
    return _extract.extract(
        manifest,
        layer_index=layer_index,
        task_sources=TASK_SOURCES,
        level1_sources=LEVEL1_SOURCES,
        schema_version=SCHEMA_VERSION,
    )


#: The layer a snapshot describes. Mid-stack, like Qwen3's 14 of 28: the first
#: and last layers of a decoder see distributions the rest do not.
REPRESENTATIVE_LAYER = 16


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.llama3.harvest.snapshot",
        description=(
            "Regenerate or validate the tracked Llama-3 workload snapshot from a "
            "full harvest manifest."
        ),
    )
    parser.add_argument("--harvest", type=Path,
                        default=Path("results/llama3-level4/harvest.json"),
                        help="the harvest manifest to derive from")
    parser.add_argument("--layer", type=int, default=REPRESENTATIVE_LAYER,
                        help="the representative decoder layer to describe")
    parser.add_argument("--out", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--write", action="store_true",
                        help="write the derived snapshot")
    parser.add_argument("--validate", action="store_true",
                        help="re-derive and fail if it disagrees with the tracked file")
    args = parser.parse_args(argv)

    if not args.write and not args.validate:
        parser.error("choose --write or --validate")
    try:
        manifest = json.loads(args.harvest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"no harvest at {args.harvest}; run harvest.harvest first", file=sys.stderr)
        return 2

    derived = extract(manifest, layer_index=args.layer)
    derived["snapshot_hash"] = snapshot_hash(derived)

    if args.validate:
        try:
            frozen = load(args.out)
        except SnapshotError as exc:
            print(f"cannot validate: {exc}", file=sys.stderr)
            return 1
        problems = diff(frozen, derived)
        if problems:
            print("the tracked snapshot disagrees with this harvest:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{args.out} agrees with {args.harvest}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}  snapshot_hash={derived['snapshot_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
