"""Model-tier benchmark: evolved kernels inside a Llama-3 training step.

    python -m evograd.bench.tier3_cli --model llama_3_8b_4l \
        --candidate rmsnorm=evolved_rmsnorm.py --out ~/tmp/tier3.json

With no candidate it measures the reference line: eager against any declared
pair baseline that covers a patchable site. Validate the harness that way first
-- if Liger's end-to-end numbers reproduce here, the harness works, and that
check does not need an evolved kernel to exist.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

MODELS = ("llama_3_8b_4l", "llama_3_8b")


def load_program(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_tier3_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama_3_8b_4l", choices=MODELS)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="SITE=PATH",
        help="patch one site with an evolved program, e.g. rmsnorm=best.py. "
             "Repeatable; sites are rms_norm, swiglu, cross_entropy",
    )
    parser.add_argument(
        "--baseline",
        default="liger",
        help="declared pair baseline to patch in as a second provider, or 'none'",
    )
    parser.add_argument(
        "--identity-control",
        action="store_true",
        help=(
            "add a provider that patches every site with the eager kernel it "
            "already had. Same mathematics, all of the patching machinery: the "
            "gap to plain eager is the harness tax, and nothing smaller than it "
            "is a kernel result"
        ),
    )
    parser.add_argument(
        "--sites",
        default=None,
        help=(
            "restrict patching to these sites, comma separated "
            "(rms_norm,swiglu,cross_entropy). One at a time turns a blended "
            "number into an attribution"
        ),
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--loss-steps", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _baseline_kernels(baseline: str, ops):
    """A reviewed pair baseline patched into the sites it covers.

    This is what makes the harness checkable before an evolved kernel exists:
    Liger publishes end-to-end Llama numbers, so reproducing them here validates
    the measurement rather than the kernel.
    """
    from evograd.bench.tier3 import SITE_OPS, kernel_from_pair, patch, KernelSet
    from evograd.opdecl.baselines import baseline_candidate_module

    kernels, covered = KernelSet(), []
    for site, op_name in SITE_OPS.items():
        op = ops.get(op_name)
        if op is None or baseline not in op.performance_baselines:
            continue
        try:
            module = baseline_candidate_module(op, baseline)
        except ValueError:
            continue  # not a pair baseline; cannot stand in as a kernel
        kernels = patch(kernels, site, kernel_from_pair(op, module))
        covered.append(site)
    return (kernels, covered) if covered else (None, [])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

    import torch

    from evograd.bench.tier3 import (
        KernelSet, SITE_OPS, identity_control_kernels, loss_agreement,
        patched_kernels, restrict, run_tier3,
    )
    from evograd.bench.tier3_llama import LlamaWorkload
    from evograd.opdecl import models as model_registry
    from evograd.ops import OPS

    config = {
        "llama_3_8b_4l": model_registry.LLAMA_3_8B_4L,
        "llama_3_8b": model_registry.LLAMA_3_8B,
    }[args.model]

    sites = tuple(s.strip() for s in args.sites.split(",")) if args.sites else None
    if sites:
        unknown = set(sites) - set(SITE_OPS)
        if unknown:
            _parser().error(f"unknown sites {sorted(unknown)}; known: {sorted(SITE_OPS)}")

    def limited(kernels):
        return restrict(kernels, sites) if sites else kernels

    providers = {"eager": KernelSet()}

    if args.identity_control:
        providers["eager_through_bind"] = identity_control_kernels(OPS, sites)

    if args.baseline and args.baseline != "none":
        kernels, covered = _baseline_kernels(args.baseline, OPS)
        if kernels is not None:
            providers[args.baseline] = limited(kernels)
            print(f"[tier3] {args.baseline} patches {covered}", file=sys.stderr)
        else:
            print(
                f"[tier3] no site has a {args.baseline!r} pair baseline; skipping",
                file=sys.stderr,
            )

    if args.candidate:
        selected = {}
        for entry in args.candidate:
            site, _, path = entry.partition("=")
            if not path:
                _parser().error(f"--candidate wants SITE=PATH, got {entry!r}")
            if site not in SITE_OPS:
                _parser().error(f"unknown site {site!r}; known: {sorted(SITE_OPS)}")
            selected[site] = load_program(Path(path))
        providers["candidate"] = limited(patched_kernels(selected, OPS))
        print(f"[tier3] candidate patches {sorted(selected)}", file=sys.stderr)

    # The CLI picks the workload; the harness does not know what it is.
    workload = LlamaWorkload(
        config, batch=args.batch, tokens=args.tokens, device=args.device,
        dtype=getattr(torch, args.dtype), seed=args.seed,
    )
    report = run_tier3(
        workload, providers,
        warmup=args.warmup, steps=args.steps, blocks=args.blocks,
        loss_steps=args.loss_steps, seed=args.seed,
    )
    report["loss_agreement"] = loss_agreement(report)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)

    eager = report["providers"].get("eager", {})
    for name, entry in sorted(report["providers"].items()):
        if not entry.get("ok"):
            print(f"  {name:12} FAILED  {entry.get('error')}", file=sys.stderr)
            continue
        speedup = (
            eager["step_ms"] / entry["step_ms"]
            if eager.get("ok") and entry["step_ms"] else float("nan")
        )
        print(
            f"  {name:12} {entry['step_ms']:8.2f} ms/step  "
            f"{entry['units_per_second']:10.0f} {entry['unit_name']}/s  "
            f"{entry['peak_memory_bytes'] / 2**30:6.2f} GiB  "
            f"cpu_bound={entry['cpu_bound_fraction']:.2f}  {speedup:.3f}x",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
