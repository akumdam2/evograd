#!/usr/bin/env python3
"""The 200-step training study for Llama-3.2-1B, one provider per process.

    PYTHONPATH=src python scripts/llama_training_study.py \
        --provider qkv_mlp_residual \
        --patch qkv_rope=results/evolve/best-llama3_qkv_rope.py \
        --patch swiglu_mlp=results/evolve/best-llama3_swiglu_mlp.py \
        --patch residual_rmsnorm=results/evolve/best-llama3_residual_rmsnorm.py \
        --out results/llama_training/qkv_mlp_residual.json

``tier3-bench --scope model`` times a training step; it does not train. The
2026-09-17 report's training half -- 200 steps, a token-weighted NLL curve, and
validation at fixed checkpoints -- runs through
:mod:`evograd.evaluation.tier3.gate.training`, which Qwen3 drives from its
``protocol4_cli`` and Llama had no driver for. This is that driver, and nothing
else: the loop, the batches and the evaluator all belong to the workload.

What the loop guarantees, and why it is not just a training script: validation
copies the trained weights into a **fresh unpatched model** and scores the same
held-out synthetic batches for every provider, so a kernel whose own forward is
wrong cannot flatter its own validation number. Training and validation streams
are disjoint by construction (``data_seed + 1000 + step`` against
``data_seed + 900000 + index``).

**Llama's tokens are synthetic and its weights are seeded.** The NLL curve says
whether a patched model executes and optimizes like the unpatched one. It is
not a language-model quality result and is not comparable with Qwen's real-text
losses. Every result this writes carries that label.

Run one provider per process, as the report did, so nothing a provider leaves
resident reaches the next one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SITES = ("qkv_rope", "attention", "swiglu_mlp", "residual_rmsnorm")


def build_kernels(patches: dict[str, str], registry):
    """``eager`` when nothing is patched; otherwise each site by its own route.

    ``compile`` and ``liger`` are the same specs ``--patch-set`` takes, so a
    provider here is spelled exactly as it is for ``tier3-bench``.
    """
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.providers import kernels_from_patches

    if not patches:
        return KernelSet(registry=registry)
    return kernels_from_patches(patches, registry, ops=TASKS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True,
                        help="name recorded in the result, e.g. eager or qkv_mlp_residual")
    parser.add_argument("--patch", action="append", default=[], metavar="SITE=SPEC",
                        help="SITE=compile|liger|PATH, repeatable; omit entirely for eager")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=50,
                        help="steps per reported training-NLL window; must divide --steps")
    parser.add_argument("--checkpoints", default="0,50,100,200",
                        help="steps at which validation runs")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-norms", action="store_true",
                        help="also record gradient and update norms at step 1 and checkpoints")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from evograd.evaluation.tier3.gate.training import TrainingPlan, run_training
    from evograd.evaluation.tier3.providers import parse_patch_specs
    from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import llama3_sites
    from evograd.evaluation.tier3.workloads.llama3_2_1b.training import (
        train_batches, validation_batches)
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    patches = parse_patch_specs(args.patch)
    unknown = sorted(set(patches) - set(SITES))
    if unknown:
        parser.error(f"unknown sites {unknown}; Llama's are {list(SITES)}")

    workload = Llama3Workload.from_config({
        "device": args.device, "seed": args.seed, "data_seed": args.data_seed,
    })
    plan = TrainingPlan(
        steps=args.steps, window=args.window,
        checkpoints=tuple(int(c) for c in args.checkpoints.split(",") if c != ""),
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
    )
    kernels = build_kernels(patches, llama3_sites())
    validation = validation_batches(workload)

    print(f"[training] {args.provider}: {plan.steps} steps, patches "
          f"{sorted(patches) or 'none (eager)'}", file=sys.stderr, flush=True)
    result: dict[str, Any] = run_training(
        workload, kernels, plan=plan,
        train_batches=train_batches(workload, plan.steps),
        validation_batches=validation,
        validation_steps=validation.batches_per_epoch,
        data_seed=args.data_seed,
        evaluator_factory=workload.eager_evaluator,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
        record_norms=args.record_norms,
    )
    result.update({
        "provider": args.provider,
        "patches": patches,
        "kernel_origin": [source.origin for source in kernels.sources],
        "model": "llama_3_2_1b",
        "data": ("deterministic synthetic tokens on seeded weights: an execution and "
                 "optimization diagnostic, not a language-model quality result, and "
                 "not comparable with a real-text loss"),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
