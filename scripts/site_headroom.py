#!/usr/bin/env python3
"""What is the most a single operator could possibly be worth?

    PYTHONPATH=src python scripts/site_headroom.py --scope both \
        --artifact results/llama3-level4/layer8.pt \
        --out results/headroom.json

Every speedup an evolved kernel reports is bounded by the share of the step its
site occupies, and that bound is usually much tighter than it looks: the QKV
boundary is 18.5% of a decoder layer and ~13% of a training step, so an
*infinitely fast* QKV kernel buys at most ~1.15x. Knowing that number before a
search starts is the difference between optimizing a boundary worth having and
one that cannot repay the effort.

This measures the bound directly, per site, by replacing the site with a **null
kernel**: a differentiable callable that returns cached output tensors and
zero-filled input gradients. It keeps the model's structure, the autograd graph
and every other site intact, and removes only this site's arithmetic. The
resulting step time is the floor; ``native / floor`` is the ceiling on any
kernel for that site.

**The null kernel computes wrong numbers by construction.** Nothing here is a
correctness run: the verification gate is disabled deliberately, gradients are
zeros, and a model trained under it would not learn. Timing is the only output
that means anything, and every record says so.

Two numbers are reported per site:

``measured_ceiling``
    native time / null-kernel time. Accounts for overlap: if the step is
    partly host-bound, removing device work may buy less than its duration.
``analytic_ceiling``
    step / (step - per_call_cost x calls), from a tier-2 report when one is
    given. A cross-check on the measured number, not a substitute -- it assumes
    the site's cost is fully on the critical path.

A null kernel still pays its launches, its allocations and the engine node, so
the floor is slightly pessimistic and the ceiling slightly conservative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SITES = ("qkv_rope", "attention", "swiglu_mlp", "residual_rmsnorm")
#: Site -> how many times the *model* runs it per step. The block runs each
#: once, except residual, whose cross-layer and final-norm fusions live outside
#: any single decoder layer.
MODEL_CALLS = {"qkv_rope": 16, "attention": 16, "swiglu_mlp": 16, "residual_rmsnorm": 32}


def null_kernel(op_name: str):
    """A differentiable stand-in that returns cached outputs and zero grads.

    The first call runs the declaration's own ``runtime_forward`` once, under
    ``no_grad``, to learn each output's shape, stride and dtype -- asking the
    declaration rather than hard-coding four signatures is what keeps this
    generic across sites. Every later call returns those tensors.
    """
    from evograd.benchmark import get_task
    from evograd.opdecl.oracle import resolve_runtime_forward

    reference = resolve_runtime_forward(get_task(op_name))
    cache: dict[str, Any] = {}

    class _Null(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            tensors = [a for a in args if torch.is_tensor(a)]
            ctx.save_for_backward(*tensors)
            ctx.arg_is_tensor = [torch.is_tensor(a) for a in args]
            if "outputs" not in cache:
                with torch.no_grad():
                    produced = reference(*args)
                cache["tuple"] = isinstance(produced, tuple)
                outs = produced if isinstance(produced, tuple) else (produced,)
                # Detached clones: the cache must not alias anything the model
                # later writes into, or a later step would change the "output".
                cache["outputs"] = tuple(o.detach().clone() for o in outs)
            outs = cache["outputs"]
            return outs if cache["tuple"] else outs[0]

        @staticmethod
        def backward(ctx, *grad_outputs):
            grads, index = [], 0
            for is_tensor in ctx.arg_is_tensor:
                if not is_tensor:
                    grads.append(None)
                    continue
                grads.append(torch.zeros_like(ctx.saved_tensors[index]))
                index += 1
            return tuple(grads)

    def kernel(*args):
        return _Null.apply(*args)

    return kernel


def kernels_for(site: str, registry, op_name: str):
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    return patch(
        KernelSet(registry=registry), site, null_kernel(op_name),
        source=KernelSource(site=site, op_name=op_name, module=None,
                            origin="headroom:null_kernel"),
    )


# ── block scope ──────────────────────────────────────────────────────────────


def block_time(adapter, kernels, *, device: str, warmup: int, samples: int, blocks: int):
    """Median forward+VJP of one decoder layer, timed exactly as tier 3 does."""
    from evograd.evaluation.tier3.block import _reset_fn, time_vjp

    built = adapter.build(device=device)
    installed = adapter.install(built, kernels) if kernels is not None else None
    invocation = adapter.prepare(built, device=device)
    result = time_vjp(built.module, invocation, reset=_reset_fn(adapter, built, installed),
                      device=device, warmup=warmup, samples=samples, blocks=blocks)
    return result["forward_backward_ms"]


def run_block(args) -> dict[str, Any]:
    from evograd.evaluation.tier3.workloads.llama3_2_1b.block import captured_case
    from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import SITE_OPS

    adapter = captured_case(args.artifact, layer_index=args.layer_index,
                            canonical=not args.no_canonical_check)
    native = block_time(adapter, None, device=args.device, warmup=args.warmup,
                        samples=args.samples, blocks=args.blocks)
    print(f"[block] native {native:.3f} ms", file=sys.stderr, flush=True)

    out: dict[str, Any] = {"native_ms": native, "sites": {}}
    for site in args.sites:
        kernels = kernels_for(site, adapter.registry, SITE_OPS[site])
        floor = block_time(adapter, kernels, device=args.device, warmup=args.warmup,
                           samples=args.samples, blocks=args.blocks)
        out["sites"][site] = {
            "floor_ms": floor,
            "measured_ceiling": native / floor,
            "site_share": 1.0 - floor / native,
        }
        print(f"[block] {site:18s} floor {floor:7.3f} ms  ceiling {native / floor:.3f}x"
              f"  share {(1 - floor / native) * 100:.1f}%", file=sys.stderr, flush=True)
    return out


# ── model scope ──────────────────────────────────────────────────────────────


def run_model(args) -> dict[str, Any]:
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.runner import measure_one
    from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import SITE_OPS, llama3_sites
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    workload = Llama3Workload.from_config({"device": args.device, "seed": args.seed,
                                           "data_seed": args.data_seed})
    registry = llama3_sites()
    # verify=False on purpose: the null kernel is numerically wrong and the
    # gate would reject it before timing. Nothing here is a correctness claim.
    options = dict(ops=TASKS, warmup=args.warmup, steps=args.steps, blocks=args.blocks,
                   loss_steps=0, learning_rate=1e-4, seed=args.seed,
                   verify=False, device=args.device)

    from evograd.evaluation.tier3.patch import KernelSet

    entry = measure_one(workload, "native", KernelSet(registry=registry), **options)
    native = entry["latency"]["median_ms"]
    print(f"[model] native {native:.2f} ms/step", file=sys.stderr, flush=True)

    out: dict[str, Any] = {"native_ms": native, "sites": {}}
    for site in args.sites:
        kernels = kernels_for(site, registry, SITE_OPS[site])
        measured = measure_one(workload, f"null_{site}", kernels, **options)
        if not measured.get("ok"):
            out["sites"][site] = {"error": str(measured.get("error"))[:300]}
            print(f"[model] {site:18s} FAILED {str(measured.get('error'))[:90]}",
                  file=sys.stderr)
            continue
        floor = measured["latency"]["median_ms"]
        out["sites"][site] = {
            "floor_ms": floor,
            "measured_ceiling": native / floor,
            "site_share": 1.0 - floor / native,
            "calls_per_step": MODEL_CALLS.get(site),
        }
        print(f"[model] {site:18s} floor {floor:8.2f} ms  ceiling {native / floor:.3f}x"
              f"  share {(1 - floor / native) * 100:.1f}%", file=sys.stderr, flush=True)
    return out


# ── analytic cross-check ─────────────────────────────────────────────────────


def analytic(t2_dir: Path, native_ms: float, sites, calls) -> dict[str, Any]:
    """step / (step - eager per-call cost x calls), from tier-2 reports.

    Assumes the site's isolated cost is entirely on the step's critical path,
    which the measured ceiling does not. Where the two disagree, the measured
    one is the answer and the gap is the overlap.
    """
    from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import SITE_OPS

    result = {}
    for site in sites:
        path = t2_dir / f"t2-{SITE_OPS[site]}.json"
        if not path.exists():
            continue
        providers = json.loads(path.read_text())["cases"][0]["providers"]
        per_call = (providers.get("eager", {}).get("full_step") or {}).get("median_ms")
        if not per_call:
            continue
        total = per_call * calls.get(site, 1)
        result[site] = {
            "eager_per_call_ms": per_call, "calls": calls.get(site, 1),
            "site_total_ms": total,
            "analytic_ceiling": native_ms / max(native_ms - total, 1e-9),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=("block", "model", "both"), default="both")
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    parser.add_argument("--artifact", type=Path, default=Path("results/llama3-level4/layer8.pt"))
    parser.add_argument("--layer-index", type=int, default=8)
    parser.add_argument("--no-canonical-check", action="store_true")
    parser.add_argument("--t2-dir", type=Path, default=None,
                        help="directory of t2-<op>.json reports, for the analytic cross-check")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10, help="block scope")
    parser.add_argument("--steps", type=int, default=10, help="model scope")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "schema": "evograd-site-headroom/1",
        "note": ("null kernels return cached outputs and zero gradients; these runs are "
                 "timing-only and numerically invalid by construction"),
        "environment": {"torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "sites": list(args.sites),
    }
    if args.scope in ("block", "both"):
        report["block"] = run_block(args)
        if args.t2_dir:
            report["block"]["analytic"] = analytic(
                args.t2_dir, report["block"]["native_ms"], args.sites,
                {s: 1 for s in args.sites})
    if args.scope in ("model", "both"):
        report["model"] = run_model(args)
        if args.t2_dir:
            report["model"]["analytic"] = analytic(
                args.t2_dir, report["model"]["native_ms"], args.sites, MODEL_CALLS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
