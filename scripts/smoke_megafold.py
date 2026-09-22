"""One MegaFold training step per provider, on a GPU node.

    PYTHONPATH=/path/to/MegaFold:src python scripts/smoke_megafold.py

    # the full 48-block model at MegaFold's own crop
    PYTHONPATH=/path/to/MegaFold:src python scripts/smoke_megafold.py \
        --model megafold --residues 384

    # with real evolved kernels instead of the identity stand-in
    PYTHONPATH=/path/to/MegaFold:src python scripts/smoke_megafold.py \
        --candidate layer_norm=best_layernorm.py \
        --candidate transition=best_swiglu.py

This is a smoke test, not a benchmark. It builds each provider's model, runs a
forward and a backward, and prints what came back: the loss, the peak
allocation, and the step time. It deliberately does *not* run the tier-3
protocol -- no correctness preflight, no loss trajectory, no repeated blocks,
no bootstrap interval -- so nothing it prints is a measurement anyone should
quote. Its job is to answer whether the thing runs at all, and that is the
question standing between here and a real tier-3 run.

    PYTHONPATH=/path/to/MegaFold:src python -m evograd.evaluation.tier3.cli \
        --model megafold_2l --residues 128 --baseline none --out tier3.json

Three providers, which are three different module trees on identical weights
and an identical batch:

``eager``
    Every fused module replaced by the plain PyTorch mathematics it stands for.
    The reference.
``megafold``
    The tree as MegaFold builds it, Triton EvoAttention on.
``compile``
    ``torch.compile`` of the eager model. No module surgery.
``compile_clean``
    ``compile`` with MegaFold's scaffolding rebound out of the compiler's way.
    Same mathematics; see ``workloads/megafold/scaffolding.py`` for the list.
``evolved``
    The eager tree with kernels patched into the sites. With no ``--candidate``
    this is the *identity control*: the eager mathematics routed through
    ``bind`` and the declaration adapters, which is the exact plumbing an
    evolved kernel travels. It therefore smokes the patch path without needing
    an evolved program to exist, and its step time is not a kernel result.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: Providers this script knows how to build, in the order it runs them.
#: ``compile_clean`` is last and must stay last: it rebinds MegaFold module
#: globals and dynamo config for the life of the process, and the workload
#: refuses to build any other provider after it.
DEFAULT_PROVIDERS = ("eager", "megafold", "compile", "evolved", "compile_clean")


def die(*lines: str) -> "NoReturn":  # noqa: F821 - quoted for older typing
    """Fail with something a person can act on, not a traceback."""
    print("", file=sys.stderr)
    print("smoke_megafold: FAILED", file=sys.stderr)
    for line in lines:
        print(f"  {line}", file=sys.stderr)
    print("", file=sys.stderr)
    raise SystemExit(1)


def require_megafold() -> None:
    """Refuse to start unless MegaFold is importable, and say why if not.

    The workload classifies its own import failures into three: not on
    ``PYTHONPATH``, a missing dependency, or a machine with no GPU. All three
    are actionable and none of them is this script's fault, so they are
    reported as the workload phrased them rather than as a stack trace from
    somewhere inside Triton.
    """
    try:
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            _require_model_module,
        )
    except ImportError as exc:
        die(
            "evograd itself is not importable.",
            f"{type(exc).__name__}: {exc}",
            "Run with PYTHONPATH=/path/to/MegaFold:src from the repository root.",
        )
    try:
        _require_model_module()
    except ImportError as exc:
        die(
            "MegaFold is not usable in this environment.",
            str(exc),
            "",
            "This script builds the model, so it cannot run without it.",
        )


def load_program(path: Path):
    """Load an evolved program from a file path, as the tier-3 CLI does."""
    spec = importlib.util.spec_from_file_location(f"smoke_megafold_{path.stem}", path)
    if spec is None or spec.loader is None:
        die(f"could not load candidate program {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_candidates(entries: list[str], registry) -> dict[str, Any]:
    """``SITE=PATH`` pairs, refusing a site this workload does not have."""
    selected: dict[str, Any] = {}
    for entry in entries:
        site, _, path = entry.partition("=")
        if not path:
            die(f"--candidate wants SITE=PATH, got {entry!r}")
        if site not in registry:
            die(
                f"unknown site {site!r} for the megafold workload",
                f"its sites are {sorted(registry.names)}",
            )
        selected[site] = load_program(Path(path))
    return selected


def kernels_for(provider: str, registry, candidates: dict[str, Any]):
    """The kernel set each provider runs with.

    Three of the four carry nothing: they are different *trees*, selected by
    the provider's name, and the tree is the whole difference. Only ``evolved``
    patches sites, and with no candidate programs it patches them with the
    eager mathematics routed through the real adapter plumbing.
    """
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.patch import (
        KernelSet,
        identity_control_kernels,
        patched_kernels,
    )

    if provider != "evolved":
        how = "no kernels patched; the tree is the provider"
        if provider == "compile_clean":
            how += "; scaffolding patches installed, see the list below"
        return KernelSet(registry=registry), how
    if candidates:
        return (
            patched_kernels(candidates, TASKS, registry=registry),
            f"candidate programs at {sorted(candidates)}",
        )
    return (
        identity_control_kernels(TASKS, registry=registry),
        "identity control (eager mathematics through the real patch path)",
    )


def peak_memory_bytes(device: str) -> int | None:
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def one_step(workload, model, batch, device: str) -> tuple[float, Any]:
    """One forward and backward. Returns the seconds it took and the loss."""
    import torch

    sync(device)
    started = time.perf_counter()
    loss = workload.loss(model, batch)
    loss.backward()
    sync(device)
    elapsed = time.perf_counter() - started
    # Gradients are dropped rather than stepped: this smokes the backward, and
    # an optimizer step would add state this script never reports.
    model.zero_grad(set_to_none=True)
    return elapsed, loss.detach()


def run_provider(
    provider: str, args, registry, candidates: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from evograd.evaluation.tier3.model import build_with_provenance
    from evograd.evaluation.tier3.workloads.megafold.adapter import _config
    from evograd.evaluation.tier3.workloads.megafold.workload import make_workload
    from evograd.opdecl.activity import Workload

    config = _config(args.model)
    case = Workload(
        dims=config.train_step_dims(batch=args.batch, residues=args.residues),
        dtype=args.dtype,
    )
    settings = {}
    if args.recycling is not None:
        settings["num_recycling_steps"] = args.recycling
    workload = make_workload(
        case, device=args.device, seed=args.seed, config=config, **settings
    )
    kernels, how = kernels_for(provider, registry, candidates)

    built_at = time.perf_counter()
    model, provenance = build_with_provenance(workload, kernels, provider=provider)
    build_seconds = time.perf_counter() - built_at

    batch = workload.batch_for(seed=args.data_seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # The first step is reported separately and never averaged in. For the
    # compile provider it is dominated by compilation, and hiding that inside a
    # mean would make the compiler look slow and the reader none the wiser.
    first_seconds, _first_loss = one_step(workload, model, batch, args.device)
    timings = []
    for _ in range(args.steps):
        seconds, loss = one_step(workload, model, batch, args.device)
        timings.append(seconds)

    return {
        "provider": provider,
        "kernels": how,
        "patched_sites": list(provenance.actual_sites),
        "modules_replaced": sum(len(p) for p in provenance.paths.values()),
        "build_method": provenance.method,
        "build_seconds": build_seconds,
        "first_step_seconds": first_seconds,
        "step_seconds": min(timings),
        "loss": float(loss),
        "loss_shape": tuple(loss.shape),
        "loss_finite": bool(torch.isfinite(loss).all()),
        "peak_memory_bytes": peak_memory_bytes(args.device),
    }


def describe_batch(batch: dict) -> list[str]:
    import torch

    return [
        f"{name}: {tuple(value.shape)} {str(value.dtype).removeprefix('torch.')}"
        for name, value in sorted(batch.items())
        if torch.is_tensor(value)
    ]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="megafold_2l",
                   choices=("megafold", "megafold_2l"),
                   help="megafold_2l is two pairformer blocks; iterate on it")
    p.add_argument("--residues", type=int, default=128, help="crop length")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "bfloat16", "float16"),
                   help="parameter dtype; MegaFold holds float32 and gets "
                        "mixed precision from its own autocast")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="weight seed")
    p.add_argument("--data-seed", type=int, default=1, help="batch seed")
    p.add_argument("--steps", type=int, default=2,
                   help="timed steps after the first; the minimum is reported")
    p.add_argument("--recycling", type=int, default=None,
                   help="recycling steps. MegaFold reads this as the upper end "
                        "of a randint(1, N) draw, so 1 removes the draw "
                        "entirely and makes the step fully deterministic")
    p.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS),
                   help="comma separated subset of " + ",".join(DEFAULT_PROVIDERS))
    p.add_argument("--candidate", action="append", default=[], metavar="SITE=PATH",
                   help="patch one site with an evolved program. Repeatable. "
                        "Without any, the evolved provider runs the identity "
                        "control instead")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require_megafold()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        die(
            "--device cuda was asked for and torch.cuda.is_available() is False.",
            "MegaFold needs a visible GPU to import at all, so this should not "
            "be reachable; something is wrong with the environment.",
        )

    from evograd.evaluation.tier3.workloads.megafold.workload import MEGAFOLD_SITES

    requested = [name.strip() for name in args.providers.split(",") if name.strip()]
    unknown = [name for name in requested if name not in DEFAULT_PROVIDERS]
    if unknown:
        die(f"unknown providers {unknown}", f"known: {list(DEFAULT_PROVIDERS)}")

    candidates = parse_candidates(args.candidate, MEGAFOLD_SITES)

    print(f"model        {args.model} at {args.residues} residues, batch {args.batch}")
    print(f"device       {args.device}, parameters {args.dtype}")
    print(f"seeds        weights {args.seed}, batch {args.data_seed}")
    print(f"providers    {', '.join(requested)}")
    print("")

    results, failures = [], []
    for index, provider in enumerate(requested):
        print(f"[{index + 1}/{len(requested)}] {provider} ...", flush=True)
        try:
            entry = run_provider(provider, args, MEGAFOLD_SITES, candidates)
        except Exception as exc:  # one provider must not take the smoke down
            failures.append((provider, f"{type(exc).__name__}: {exc}"))
            print(f"    FAILED {type(exc).__name__}: {exc}", flush=True)
            # The traceback, not just the message. A provider failing inside
            # MegaFold is the common case for this script, and "CUDA error:
            # device-side assert triggered" without a frame is unactionable.
            traceback.print_exc()
            sys.stdout.flush()
            continue
        results.append(entry)
        print(f"    {entry['kernels']}")
        if provider == "compile_clean":
            from evograd.evaluation.tier3.workloads.megafold import scaffolding

            for patch in scaffolding.APPLIED:
                print(f"      patched {patch['target']}")
                print(f"        at    {patch['upstream']}")
        if entry["patched_sites"]:
            print(
                f"    replaced {entry['modules_replaced']} modules at "
                f"{entry['patched_sites']}"
            )
        print(f"    loss {entry['loss']:.6f}  shape {entry['loss_shape']}")

    if results:
        print("")
        print("shapes of the batch every provider saw")
        from evograd.evaluation.tier3.workloads.megafold.adapter import _config
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload
        from evograd.opdecl.activity import Workload

        config = _config(args.model)
        probe = make_workload(
            Workload(
                dims=config.train_step_dims(
                    batch=args.batch, residues=args.residues
                ),
                dtype=args.dtype,
            ),
            device=args.device,
            seed=args.seed,
            config=config,
        )
        for line in describe_batch(probe.batch_for(seed=args.data_seed)):
            print(f"    {line}")

        print("")
        header = (
            f"{'provider':<10} {'loss':>12} {'step ms':>10} "
            f"{'first ms':>10} {'build s':>9} {'peak GiB':>9}"
        )
        print(header)
        print("-" * len(header))
        for entry in results:
            peak = entry["peak_memory_bytes"]
            peak_text = f"{peak / 2 ** 30:9.2f}" if peak is not None else "        -"
            print(
                f"{entry['provider']:<10} {entry['loss']:>12.6f} "
                f"{entry['step_seconds'] * 1000:>10.1f} "
                f"{entry['first_step_seconds'] * 1000:>10.1f} "
                f"{entry['build_seconds']:>9.1f} {peak_text}"
            )
        print("")
        print("The first step is reported separately and never averaged in: for")
        print("the compile provider it is mostly compilation. Nothing here is a")
        print("benchmark result; run the tier-3 CLI for that.")

        reference = next((e for e in results if e["provider"] == "eager"), None)
        if reference is not None and len(results) > 1:
            print("")
            print("loss agreement against eager")
            for entry in results:
                if entry["provider"] == "eager":
                    continue
                delta = abs(entry["loss"] - reference["loss"])
                print(f"    {entry['provider']:<10} {delta:.3e}")

        nonfinite = [e["provider"] for e in results if not e["loss_finite"]]
        if nonfinite:
            die(f"non-finite loss from {nonfinite}")

    if failures:
        die(*[f"{provider}: {error}" for provider, error in failures])
    print("")
    print("smoke_megafold: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
