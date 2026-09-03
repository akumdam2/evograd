"""Model-tier benchmark: evolved kernels inside a Llama-3 training step.

    python -m evograd.bench.tier3_cli --model llama_3_8b_4l \
        --candidate rms_norm=evolved_rmsnorm.py --out ~/tmp/tier3.json

With no candidate it measures the reference line: eager against any declared
pair baseline that covers a patchable site. Validate the harness that way first
-- if Liger's end-to-end numbers reproduce here, the harness works, and that
check does not need an evolved kernel to exist.

Each provider runs in its own killable child process, in a seeded random order
that the report records. An evolved kernel can hang, wedge the CUDA context, or
be OOM-killed by the operating system; none of those should cost the providers
that ran fine, and in one process all three do. ``--no-isolate`` keeps
everything here when you want a debugger.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = ("llama_3_8b_4l", "llama_3_8b", "qwen3_0_6b", "alphafold3_2l", "alphafold3")

#: Per-provider wall-clock budget. A tier-3 provider trains a model, so this is
#: minutes rather than the seconds a tier-1 case takes.
ISOLATION_TIMEOUT = int(os.environ.get("EVOGRAD_TIER3_TIMEOUT", "1800"))


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
        help="patch one site with an evolved program, e.g. rms_norm=best.py. "
             "Repeatable; the valid sites are the selected workload's "
             "(llama_3: rms_norm, swiglu, cross_entropy)",
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
            "already had. Same mathematics, all of the patching machinery. Its "
            "backward recomputes the forward, which no candidate does, so the "
            "gap to plain eager is an UPPER BOUND on what the patch plumbing "
            "costs -- a ceiling, not a measurement of the tax"
        ),
    )
    parser.add_argument(
        "--structural-identity",
        action="store_true",
        help=(
            "qwen3 only: add a provider whose adapters call the exact "
            "Transformers spellings through native autograd. It changes the "
            "module structure and no arithmetic, so it must be BITWISE "
            "identical to the unpatched model"
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
    # No defaults: the Qwen workload's canonical batch and sequence are part of
    # its identity, and a CLI default would silently replace them.
    parser.add_argument("--batch", type=int, default=None,
                        help="llama default 4; qwen3 defaults to its canonical 2")
    parser.add_argument("--tokens", type=int, default=None,
                        help="llama default 1024; qwen3 defaults to its canonical 2048")
    parser.add_argument("--layers", type=int, default=None,
                        help="qwen3 only: reduce the layer count for a smoke")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="qwen3 only: move the token stream without changing "
                             "the workload identity")
    parser.add_argument("--residues", type=int, default=None,
                        help="alphafold3 only: crop length; defaults to 128")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--loss-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # No hard default: the language models train bf16 and AlphaFold3 fp32 (as
    # MegaFold does), so the effective default follows the selected model and
    # only an explicit flag overrides it.
    parser.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"),
                        help="llama/qwen3 default bfloat16; alphafold3 defaults to float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "skip the tier-1 correctness preflight. Timings then describe "
            "whatever the kernel computes, correct or not"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=ISOLATION_TIMEOUT,
        help="seconds one provider may take before its child is killed",
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="run every provider in this process (a hang takes the run down)",
    )
    # Set by the parent when it re-invokes itself for one provider.
    parser.add_argument("--provider", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser


def _baseline_kernels(baseline: str, ops, registry):
    """A reviewed pair baseline patched into the sites it covers.

    This is what makes the harness checkable before an evolved kernel exists:
    Liger publishes end-to-end Llama numbers, so reproducing them here validates
    the measurement rather than the kernel.

    Discovery walks *this workload's* registry, so a baseline is only offered
    for sites the model being measured actually has.
    """
    from evograd.bench.tier3 import KernelSet, KernelSource, kernel_from_pair, patch
    from evograd.opdecl.baselines import baseline_candidate_module

    kernels, covered = KernelSet(registry=registry), []
    for site, op_name in registry.site_ops.items():
        op = ops.get(op_name) if op_name else None
        if op is None or baseline not in op.performance_baselines:
            continue
        try:
            module = baseline_candidate_module(op, baseline)
        except ValueError:
            continue  # not a pair baseline; cannot stand in as a kernel
        kernels = patch(
            kernels,
            site,
            kernel_from_pair(op, module),
            source=KernelSource(
                site=site, op_name=op_name, module=module,
                origin=f"baseline:{baseline}",
            ),
        )
        covered.append(site)
    return (kernels, covered) if covered else (None, [])


def _sites(args, registry):
    if not args.sites:
        return None
    sites = tuple(s.strip() for s in args.sites.split(","))
    unknown = [site for site in sites if site not in registry]
    if unknown:
        _parser().error(
            f"unknown sites {sorted(unknown)} for workload {registry.name!r}; "
            f"its sites are {sorted(registry.names)}"
        )
    return sites


def build_providers(args, *, quiet: bool = False) -> dict:
    """Every provider this invocation asks for, as ``{name: KernelSet}``.

    Built identically in the parent (which needs the names and the order) and in
    each child (which needs one of them), so an isolated run and an in-process
    run compare exactly the same things.
    """
    from evograd.bench.tier3 import (
        KernelSet, identity_control_kernels, patched_kernels, restrict,
        site_registry_for,
    )
    from evograd.ops import OPS

    # Every provider below is built against the registry the *workload* owns.
    # Nothing here reads a module-level site namespace, which is what let the
    # Llama identity control claim sites belonging to another model.
    registry = site_registry_for(build_workload(args))
    sites = _sites(args, registry)

    def limited(kernels):
        return restrict(kernels, sites) if sites else kernels

    providers = {"eager": KernelSet(registry=registry)}

    if getattr(args, "structural_identity", False):
        if args.model != "qwen3_0_6b":
            _parser().error("--structural-identity is a qwen3_0_6b provider")
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
            structural_identity_kernels,
        )

        providers["structural_identity"] = limited(
            structural_identity_kernels(registry)
        )

    if args.identity_control:
        providers["eager_through_bind"] = identity_control_kernels(
            OPS, sites, registry=registry
        )

    if args.baseline and args.baseline != "none":
        kernels, covered = _baseline_kernels(args.baseline, OPS, registry)
        if kernels is not None:
            providers[args.baseline] = limited(kernels)
            if not quiet:
                print(f"[tier3] {args.baseline} patches {covered}", file=sys.stderr)
        elif not quiet:
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
            if site not in registry:
                _parser().error(
                    f"unknown site {site!r} for workload {registry.name!r}; "
                    f"its sites are {sorted(registry.names)}"
                )
            selected[site] = load_program(Path(path))
        providers["candidate"] = limited(
            patched_kernels(selected, OPS, registry=registry)
        )
        if not quiet:
            print(f"[tier3] candidate patches {sorted(selected)}", file=sys.stderr)

    return providers


def build_workload(args):
    """The workload this invocation measures, rebuilt identically in each child.

    Both branches take only values that came off the command line, which is what
    lets a child process reconstruct the same workload from the same argv rather
    than inheriting an object it cannot pickle.
    """
    import torch

    dtype = args.dtype or (
        "float32" if args.model.startswith("alphafold3") else "bfloat16"
    )

    if args.model.startswith("alphafold3"):
        from evograd.opdecl import models as model_registry
        from evograd.opdecl.activity import Workload
        from evograd.ops.level4.alphafold3.workload import make_workload

        config = {
            "alphafold3_2l": model_registry.ALPHAFOLD3_2L,
            "alphafold3": model_registry.ALPHAFOLD3,
        }[args.model]
        case = Workload(
            dims=config.train_step_dims(
                batch=args.batch if args.batch is not None else 1,
                residues=args.residues if args.residues is not None else 128,
            ),
            dtype=dtype,
        )
        return make_workload(case, device=args.device, seed=args.seed, config=config)

    if args.model == "qwen3_0_6b":
        from evograd.bench.workloads.qwen3.evaluation.tier3.workload import Qwen3Workload

        config = {
            "dtype": dtype,
            "device": args.device,
            "seed": args.seed,
            "data_seed": args.data_seed,
        }
        # The canonical batch and sequence are the workload's, not the CLI's
        # defaults; only an explicit flag overrides them.
        if args.batch is not None:
            config["batch_size"] = args.batch
        if args.tokens is not None:
            config["seq_len"] = args.tokens
        if args.layers is not None:
            config["arch_overrides"] = {"num_hidden_layers": args.layers}
        return Qwen3Workload.from_config(config)

    from evograd.bench.tier3_llama import LlamaWorkload
    from evograd.opdecl import models as model_registry

    config = {
        "llama_3_8b_4l": model_registry.LLAMA_3_8B_4L,
        "llama_3_8b": model_registry.LLAMA_3_8B,
    }[args.model]
    return LlamaWorkload(
        config, batch=args.batch if args.batch is not None else 4,
        tokens=args.tokens if args.tokens is not None else 1024,
        device=args.device, dtype=getattr(torch, dtype), seed=args.seed,
    )


def _measure_options(args) -> dict:
    return {
        "warmup": args.warmup,
        "steps": args.steps,
        "blocks": args.blocks,
        "loss_steps": args.loss_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "verify": not args.no_verify,
        "device": args.device,
    }


def _run_one_provider(args) -> dict:
    """Worker mode: build one provider, measure it, hand back its entry."""
    from evograd.bench.tier3 import measure_one
    from evograd.ops import OPS

    providers = build_providers(args, quiet=True)
    if args.provider not in providers:
        return {
            "ok": False,
            "provider": args.provider,
            "error": f"provider {args.provider!r} was not built in this process",
            "failed_at": "setup",
        }
    return measure_one(
        build_workload(args), args.provider, providers[args.provider],
        ops=OPS, **_measure_options(args),
    )


def _run_isolated(argv: list[str], provider: str, timeout: int) -> dict:
    """Re-invoke this module for one provider and read back its JSON.

    Three failure modes are captured here rather than allowed to end the run: a
    provider that exceeds its budget is killed and recorded; one that dies --
    an uncaught exception, an OS OOM kill, a corrupted CUDA context -- is
    recorded with the tail of its stderr; one that exits cleanly without writing
    a result is recorded as such rather than silently missing from the report.
    """
    import tempfile

    handle, result_path = tempfile.mkstemp(prefix="evograd_tier3_", suffix=".json")
    os.close(handle)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "evograd.bench.tier3_cli", *argv,
             "--provider", provider, "--result-json", result_path],
            capture_output=True, text=True, timeout=timeout,
        )
        try:
            with open(result_path, encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {
                "ok": False,
                "provider": provider,
                "error": (
                    f"tier3 worker exited rc={process.returncode} without a result"
                ),
                "failed_at": "subprocess",
                "returncode": process.returncode,
                "stderr_tail": process.stderr[-4000:],
            }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "provider": provider,
            "error": f"provider exceeded {timeout}s and was killed",
            "failed_at": "timeout",
        }
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)

    # Worker mode: one provider, write JSON, exit.
    if args.provider is not None:
        entry = _run_one_provider(args)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(entry), encoding="utf-8")
        return 0

    from evograd.bench.tier3 import (
        assemble_report, loss_agreement, measure_one, provider_order,
    )
    from evograd.ops import OPS

    providers = build_providers(args)
    order = provider_order(providers, seed=args.seed)

    workload = build_workload(args)
    parent_argv = [a for a in argv]
    results: dict = {}
    for index, name in enumerate(order):
        print(
            f"[tier3] {index + 1}/{len(order)} {name}", file=sys.stderr, flush=True
        )
        if args.no_isolate:
            results[name] = measure_one(
                workload, name, providers[name], ops=OPS, **_measure_options(args)
            )
        else:
            results[name] = _run_isolated(parent_argv, name, args.timeout)

    report = assemble_report(
        workload, results, order,
        warmup=args.warmup, steps=args.steps, blocks=args.blocks,
        loss_steps=args.loss_steps, learning_rate=args.learning_rate,
        seed=args.seed, verify=not args.no_verify,
        isolation=(
            "single process" if args.no_isolate
            else f"one child process per provider, {args.timeout}s budget"
        ),
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
            print(
                f"  {name:18} FAILED [{entry.get('failed_at', '?')}] "
                f"{entry.get('error')}",
                file=sys.stderr,
            )
            continue
        speedup = (
            eager["step_ms"] / entry["step_ms"]
            if eager.get("ok") and entry["step_ms"] else float("nan")
        )
        print(
            f"  {name:18} {entry['step_ms']:8.2f} ms/step  "
            f"{entry['units_per_second']:10.0f} {entry['unit_name']}/s  "
            f"{entry['peak_memory_bytes'] / 2**30:6.2f} GiB  "
            f"cpu_bound={entry['cpu_bound_fraction']:.2f}  {speedup:.3f}x",
            file=sys.stderr,
        )
    if args.identity_control and report["providers"].get("eager_through_bind", {}).get("ok"):
        print(
            "  note: eager_through_bind recomputes each forward in its backward, "
            "so its gap to eager is an UPPER BOUND on patch plumbing cost",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
