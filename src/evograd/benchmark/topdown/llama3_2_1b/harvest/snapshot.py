"""The tracked slice of the Llama-3 harvest that repository tasks may read.

A harvest manifest is a local result: a megabyte of transcript, not version
controlled, existing only on a machine that has run the harvest. A
declaration in ``evograd/ops/`` cannot depend on any of that -- it has to import
on a laptop with no GPU -- yet the whole point of those tasks is that their
shapes come from a real training step rather than from someone's judgement.

The snapshot is the reconciliation. Reading and verifying one is shared; what is
here is the part that names things.

``snapshot.json`` is *derived*, not authored: it comes out of a harvest of the
real-payload capture's first step (pinned checkpoint, order-101 batch)::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python -m evograd.benchmark.topdown.llama3_2_1b.harvest.harvest \\
        --out results/llama3-level3/harvest.json
    python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot \\
        --harvest results/llama3-level3/harvest.json --write

Level-1 mapping is the shared decoder one unchanged: every role Llama presents
-- ``q_proj``, ``o_proj``, ``gate_proj``, ``down_proj``, ``lm_head``,
``input_layernorm``, ``act_fn``, SDPA, cross-entropy -- is a standard
HuggingFace spelling already in it. Llama has no ``q_norm``/``k_norm``, and
those entries simply never match.

:data:`TASK_SOURCES` carries the three Level-2 boundaries, each keyed by this
model's own declared task. Llama's attention block has **no** ``q_norm``/
``k_norm`` supporting configurations, because there is nothing to observe; a
shared entry with those roles would find no match and fail the extraction
rather than producing a smaller snapshot, which is why the two workloads state
their task sources separately instead of sharing one table.
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

#: Bumped from /1 when the snapshot began to record ``run``: the verification
#: run its harvest observed (the integration's pinned weights and data, the first
#: TRAIN batch of order 101 and the initial parameter state).
SCHEMA_VERSION = "evograd-llama3-snapshot/2"

SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")

#: The shared decoder mapping, unchanged. See the module docstring.
LEVEL1_SOURCES: dict[str, dict[str, Any]] = DECODER_LEVEL1

#: Level-2 tasks the snapshot carries, keyed by *this model's* task
#: identities. A snapshot records what one captured step observed, so it is
#: keyed by the task those observations belong to and never by another model's.
TASK_SOURCES: dict[str, dict[str, Any]] = {
    "llama3_swiglu_mlp": {
        "task": "mlp",
        "roles": ["mlp"],
        "supporting_tasks": {
            "gate_up_projection": {"task": "linear", "roles": ["gate_proj", "up_proj"]},
            "down_projection": {"task": "linear", "roles": ["down_proj"]},
            "activation": {"task": "silu", "roles": ["act_fn"]},
        },
    },
    # The residual fusion site. Its primary configuration is the residual-width
    # RMSNorm, which the harvest deduplicates across `input_layernorm`,
    # `post_attention_layernorm` and the final `model.norm`. The residual *add*
    # is not an observed boundary (it is a bare `+`), so the number of fusion
    # sites is derived from the architecture and cross-checked against the
    # observed RMSNorm count.
    "llama3_residual_rmsnorm": {
        "task": "rms_norm",
        "roles": ["post_attention_layernorm"],
        "supporting_tasks": {
            "decoder_layer": {"task": "decoder_layer", "roles": ["decoder_layer"]},
        },
        "derived": "fusion_sites",
    },
    # The attention block's primary configuration is the `LlamaAttention` module
    # itself, so its frequency is the module's calls per step (one per layer).
    # The decoder layer passes `hidden_states=` by keyword, so the record's
    # input is its keyword tensor. The module's inner calls are supporting
    # configurations; there is no `q_norm`/`k_norm` to observe.
    #
    # `q_proj` and `o_proj` are both 2048 -> 2048, so the harvest deduplicates
    # them into one configuration and `q_projection` and `output_projection`
    # resolve to the same record. They really are the same shape.
    "llama3_attention_block": {
        "task": "attention",
        "roles": ["self_attn"],
        "supporting_tasks": {
            "q_projection": {"task": "linear", "roles": ["q_proj"]},
            "kv_projection": {"task": "linear", "roles": ["k_proj", "v_proj"]},
            "rope": {"task": "rope_apply", "roles": ["apply_rotary_pos_emb"]},
            "sdpa": {"task": "sdpa", "roles": ["scaled_dot_product_attention"]},
            "output_projection": {"task": "linear", "roles": ["o_proj"]},
        },
    },
}


#: What to say when the file is not there. The shared reader's message --
#: "it is tracked, not generated" -- is right for a workload whose snapshot has
#: been committed and has since gone missing. It is the wrong diagnosis here:
#: nobody has ever produced this one, and the fix is to run a harvest, not to
#: restore a file.
_UNHARVESTED = """{path} does not exist.

A snapshot is derived from a harvest, not authored, and no Llama-3 harvest has
been run. On a machine with a GPU:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python -m evograd.benchmark.topdown.llama3_2_1b.harvest.harvest \\
        --out results/llama3-level3/harvest.json
    python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot \\
        --harvest results/llama3-level3/harvest.json --write

Nothing here can synthesise one, and a hand-written snapshot would defeat its
purpose: the shapes are meant to be what a real training step ran."""


def load(path: Path | None = None) -> dict[str, Any]:
    """The frozen Llama-3 snapshot, with its hash and schema verified.

    Only *absence* is re-reported. A snapshot that exists and fails its hash or
    schema check still raises the shared reader's error, because that is a
    corrupt file rather than a missing run and the two need different answers.
    """
    path = Path(path or SNAPSHOT_PATH)
    if not path.is_file():
        raise SnapshotError(_UNHARVESTED.format(path=path))
    return _load(path, schema_version=SCHEMA_VERSION)


def task(name: str, path: Path | None = None) -> dict[str, Any]:
    """One Level-2 task entry from the Llama-3 snapshot."""
    path = Path(path or SNAPSHOT_PATH)
    if not path.is_file():
        raise SnapshotError(_UNHARVESTED.format(path=path))
    return _task(name, path, schema_version=SCHEMA_VERSION)


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
#: and last layers of a decoder see distributions the rest do not. Eight of 16
#: since this workload moved to Llama-3.2-1B; the old 16 was mid-stack of the
#: 8B's 32 and is one past the end here, which would have matched no layer at
#: all.
REPRESENTATIVE_LAYER = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot",
        description=(
            "Regenerate or validate the tracked Llama-3 workload snapshot from a "
            "full harvest manifest."
        ),
    )
    parser.add_argument("--harvest", type=Path,
                        default=Path("results/llama3-level3/harvest.json"),
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
    refused = args.write and _extract.tracked_write_refusal("llama_3_2_1b", manifest, args.out,
                                                            SNAPSHOT_PATH)
    if refused:
        print(refused, file=sys.stderr)
        return 1

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
