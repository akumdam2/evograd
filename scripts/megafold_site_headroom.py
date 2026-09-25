#!/usr/bin/env python3
"""What is the most a single MegaFold site could possibly be worth?

    PYTHONPATH=/path/to/MegaFold:src python scripts/megafold_site_headroom.py \
        --residues 384 --t2-dir results/megafold-level4/t2 \
        --calls results/megafold-level4/site_dtypes_identity.json \
        --out results/megafold-level4/headroom_model.json

The MegaFold port of ``scripts/site_headroom.py``, model scope only (there is
no captured MegaFold block yet). Same method: each site is replaced by a
**null kernel** -- a differentiable callable that returns cached outputs and
zero-filled input gradients -- so the step keeps its structure and loses only
this site's arithmetic. ``native / floor`` is the ceiling on any kernel there.

Two things differ from the Llama script. A MegaFold site is called at many
shapes and dtypes per step (the pair stack, the MSA stack, the diffusion
module and the heads all reach the same LayerNorm), so the null kernel caches
one output per (shape, dtype) signature instead of one output. And the tree
is MegaFold's ``eager`` provider: the runner builds every non-mode provider
name on the eager tree, which is exactly where the sites are patched.

**Timing only.** The null kernel computes wrong numbers by construction; the
verification gate is off on purpose and every record says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def null_kernel(site: str, registry, op_name: str):
    """A differentiable stand-in: cached outputs forward, zero gradients back.

    The output for a new input signature comes from the site's **identity
    control** -- the declared eager mathematics routed through the same
    ``bind`` adapter an evolved kernel uses -- so the probe sees the site's
    real reshaping and dtype handling (``layernorm_linear`` flattens its
    leading axes there, for one). Nothing is saved for backward: the zero
    gradients need only shapes, and saving a checkpointed input would make
    MegaFold's non-reentrant checkpoint unpack it a second time.
    """
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.patch import identity_control_kernels

    probe = identity_control_kernels(TASKS, (site,), registry=registry).kernel_for(site)
    cache: dict[tuple, tuple[bool, tuple]] = {}

    def signature(args):
        return tuple(
            (tuple(a.shape), str(a.dtype)) if torch.is_tensor(a) else ("scalar", repr(a))
            for a in args
        )

    class _Null(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.grad_specs = [
                (tuple(a.shape), a.dtype, a.device) if torch.is_tensor(a) else None
                for a in args
            ]
            key = signature(args)
            if key not in cache:
                with torch.no_grad():
                    produced = probe(*args)
                is_tuple = isinstance(produced, tuple)
                outs = produced if is_tuple else (produced,)
                cache[key] = (is_tuple, tuple(o.detach().clone() for o in outs))
            is_tuple, outs = cache[key]
            # A fresh view per call: returning the cached tensor object itself
            # would let each later call re-point its ``grad_fn``, detaching
            # every earlier call's node from the graph and skipping their
            # producers' backward -- a floor that is too low and a ceiling
            # that is too high. Views share the storage, so nothing is copied.
            outs = tuple(o.view(o.shape) for o in outs)
            return outs if is_tuple else outs[0]

        @staticmethod
        def backward(ctx, *grad_outputs):
            return tuple(
                None if spec is None else torch.zeros(spec[0], dtype=spec[1], device=spec[2])
                for spec in ctx.grad_specs
            )

    def kernel(*args):
        return _Null.apply(*args)

    kernel.cache = cache
    return kernel


def kernels_for(site: str, registry, op_name: str):
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    return patch(
        KernelSet(registry=registry), site, null_kernel(site, registry, op_name),
        source=KernelSource(site=site, op_name=op_name, module=None,
                            origin="headroom:null_kernel"),
    )


def calls_per_step(path: Path | None) -> dict[str, int]:
    """Site -> calls per training step, from the identity-patched dtype probe."""
    if path is None or not path.exists():
        return {}
    sites = json.loads(path.read_text())["sites"]
    probe_class = {
        "layer_norm": "_PatchedLayerNorm", "layernorm_linear": "_PatchedLayernormLinear",
        "transition": "_PatchedSwiGLUTransition", "pair_bias_attention": "_PatchedAttend",
        "triangle_attention": "_PatchedAttend",
    }
    out = {}
    for site, cls in probe_class.items():
        if cls in sites:
            out[site] = int(sites[cls]["count"])
        elif site == "layer_norm" and "LayerNorm" in sites:
            out[site] = int(sites["LayerNorm"]["count"])
    return out


def analytic(t2_dir: Path, native_ms: float, site_ops: dict[str, str], calls: dict[str, int]):
    """step / (step - eager per-call cost x calls) from the T2 reports, per dtype."""
    result: dict[str, Any] = {}
    for site, op in site_ops.items():
        per = {}
        for report in sorted(t2_dir.glob(f"{op}__evolved__*.json")):
            if "nocheck" in report.stem:
                continue
            providers = json.loads(report.read_text())["cases"][0]["providers"]
            eager = (providers.get("eager", {}).get("full_step") or {}).get("median_ms")
            if eager:
                per[report.stem.split("__")[-1]] = eager
        if not per or site not in calls:
            continue
        result[site] = {}
        for dtype, per_call in per.items():
            total = per_call * calls[site]
            result[site][dtype] = {
                "eager_per_call_ms": per_call, "calls": calls[site], "site_total_ms": total,
                "analytic_ceiling": native_ms / max(native_ms - total, 1e-9),
                "note": "T2 case is the pair-trunk shape only; the site also runs at other shapes",
            }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="megafold")
    parser.add_argument("--residues", type=int, default=384)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--t2-dir", type=Path, default=None)
    parser.add_argument("--calls", type=Path, default=None,
                        help="site_dtypes_identity.json from scripts/megafold_site_dtypes.py")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.cli import _parser
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.runner import measure_one
    from evograd.evaluation.tier3.workloads import tier3_adapter

    t3_args = _parser().parse_args(["--model", args.model, "--residues", str(args.residues),
                                    "--baseline", "none"])
    workload = tier3_adapter(args.model).build(t3_args)
    registry = workload.site_registry
    site_ops = dict(registry.site_ops)
    sites = args.sites or list(registry.names)
    options = dict(ops=TASKS, warmup=args.warmup, steps=args.steps, blocks=args.blocks,
                   loss_steps=0, learning_rate=1e-4, seed=args.seed, verify=False,
                   device=args.device)

    entry = measure_one(workload, "eager", KernelSet(registry=registry), **options)
    if not entry.get("ok"):
        print(f"[model] native FAILED {entry.get('error')}", file=sys.stderr)
        return 1
    native = entry["latency"]["median_ms"]
    print(f"[model] native {native:.2f} ms/step", file=sys.stderr, flush=True)
    calls = calls_per_step(args.calls)

    report: dict[str, Any] = {
        "schema": "evograd-site-headroom/1", "scope": "model", "model": args.model,
        "residues": args.residues,
        "note": ("null kernels return cached outputs (one per input signature) and zero "
                 "gradients; these runs are timing-only and numerically invalid by construction"),
        "environment": {"torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "native_ms": native, "native_peak_gib": entry["peak_memory_bytes"] / 2**30,
        "sites": {},
    }
    for site in sites:
        kernels = kernels_for(site, registry, site_ops[site])
        measured = measure_one(workload, f"null_{site}", kernels, **options)
        if not measured.get("ok"):
            report["sites"][site] = {"error": str(measured.get("error"))[:300]}
            print(f"[model] {site:20s} FAILED {str(measured.get('error'))[:90]}", file=sys.stderr)
            continue
        floor = measured["latency"]["median_ms"]
        report["sites"][site] = {
            "op": site_ops[site], "floor_ms": floor, "measured_ceiling": native / floor,
            "site_share": 1.0 - floor / native, "calls_per_step": calls.get(site),
            "signatures_seen": (len(cache) if (cache := getattr(kernels.kernel_for(site), "cache", None)) is not None else None),
            "peak_gib": measured["peak_memory_bytes"] / 2**30,
        }
        print(f"[model] {site:20s} floor {floor:9.2f} ms  ceiling {native / floor:.3f}x"
              f"  share {(1 - floor / native) * 100:.1f}%", file=sys.stderr, flush=True)
    if args.t2_dir:
        report["analytic"] = analytic(args.t2_dir, native, site_ops, calls)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
