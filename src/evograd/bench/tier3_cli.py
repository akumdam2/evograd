"""Model-tier benchmark: evolved kernels inside a real training step.

    python -m evograd.bench.tier3_cli --model <workload> \
        --candidate <site>=evolved.py --out ~/tmp/tier3.json

``--model`` lists the workloads that have tier-3 sites; ``--sites`` with an
unknown name lists the selected one's.

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

from evograd.bench.workloads import tier3_model_names

#: The workloads tier 3 can measure. Read from the registry rather than written
#: here, so adding one is an entry in ``bench.workloads`` plus a package, not an
#: edit to this file. Names only: the registry maps them to dotted paths and
#: imports nothing until one is chosen, which matters because an adapter reaches
#: torch and possibly Transformers.
MODELS = tier3_model_names()

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
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="SITE=PATH",
        help="patch one site with an evolved program, e.g. rms_norm=best.py. "
             "Repeatable; the valid sites are the selected workload's "
             "own; --sites with an unknown name lists them",
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
            "add a provider whose adapters call the exact production spellings "
            "through native autograd. It changes the module structure and no "
            "arithmetic, so it must be BITWISE identical to the unpatched "
            "model. Only workloads that declare it accept this"
        ),
    )
    parser.add_argument(
        "--sites",
        default=None,
        help=(
            "restrict patching to these sites, comma separated. One at a time "
            "turns a blended number into an attribution; an unknown name lists "
            "the workload's own sites"
        ),
    )
    # No defaults: a workload's canonical batch and sequence can be part of its
    # identity, and a CLI default would silently replace them. Each adapter
    # supplies its own.
    parser.add_argument("--batch", type=int, default=None,
                        help="override the workload's own batch size")
    parser.add_argument("--tokens", type=int, default=None,
                        help="override the workload's own sequence length")
    parser.add_argument("--layers", type=int, default=None,
                        help="reduce the layer count for a smoke, where the "
                             "workload accepts it")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="move the token stream without changing the "
                             "workload identity, where the workload accepts it")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="the whole-model gate's calibration artifact, for "
                             "workloads that have one. A calibration is bound "
                             "to the workload it was measured for, so a shrunk "
                             "run needs its own rather than the canonical one")
    parser.add_argument("--residues", type=int, default=None,
                        help="crop length, where the workload accepts it")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--loss-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # No hard default. Workloads disagree about their training dtype -- a
    # language model trains bf16, AlphaFold3 fp32 -- and the canonical value is
    # part of a workload's identity, so it belongs in that workload's spec
    # rather than in this parser. Unset means "whatever the workload says".
    parser.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"),
                        help="override the workload's own training dtype")
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

    This is what makes the harness checkable before an evolved kernel exists: a
    published baseline's end-to-end numbers, reproduced here, validate the
    measurement rather than the kernel.

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
    # Nothing here reads a module-level site namespace, which is what let one
    # model's identity control claim sites belonging to another.
    registry = site_registry_for(build_workload(args))
    sites = _sites(args, registry)

    def limited(kernels):
        return restrict(kernels, sites) if sites else kernels

    providers = {"eager": KernelSet(registry=registry)}

    # Whatever the workload itself offers beyond the providers every workload
    # has. Asked rather than branched on, so this file names no architecture.
    adapter = tier3_adapter(args.model)
    if adapter.providers is not None:
        for name, kernels in adapter.providers(args, registry).items():
            providers[name] = limited(kernels)

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


def tier3_adapter(name):
    """The selected workload's adapter, with an argparse-shaped error."""
    from evograd.bench.workloads import UnknownWorkload
    from evograd.bench.workloads import tier3_adapter as lookup

    try:
        return lookup(name)
    except UnknownWorkload as exc:  # pragma: no cover - argparse checks choices
        _parser().error(str(exc))


def check_options(args) -> None:
    """Refuse a flag the selected workload does not understand, by name.

    ``--structural-identity`` is meaningless without adapters that can call a
    production spelling; ``--layers`` is meaningless where the layer count comes
    from a frozen config. Accepting either silently would report a number for a
    run that ignored what was asked, so each workload declares what it takes and
    anything else is an error here.
    """
    adapter = tier3_adapter(args.model)
    #: Flag -> the value that means "not requested".
    optional = {"structural_identity": False, "layers": None, "data_seed": 0,
                "calibration": None, "residues": None}
    for dest, unset in optional.items():
        if dest in adapter.options:
            continue
        if getattr(args, dest, unset) != unset:
            flag = "--" + dest.replace("_", "-")
            accepted = sorted(adapter.options)
            _parser().error(
                f"{flag} is not a {args.model} option; that workload accepts "
                + (f"{accepted}" if accepted else "no workload-specific flags")
            )


def build_workload(args):
    """The workload this invocation measures, rebuilt identically in each child.

    Every adapter takes only values that came off the command line, which is
    what lets a child process reconstruct the same workload from the same argv
    rather than inheriting an object it cannot pickle.
    """
    return tier3_adapter(args.model).build(args)


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
    # Before anything is built: a flag the selected workload does not
    # understand is an error, not a silently ignored request.
    check_options(args)

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
