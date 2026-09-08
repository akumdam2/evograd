"""The command line every Level-4 workload shares: overrides, banner, report.

Two entry points, one argument surface. ``smoke_main`` runs the canonical step
and writes its report; ``harvest_main`` runs the same step with the observer
attached and writes a manifest. They take the same overrides deliberately -- the
harvest must describe *the* canonical execution, and two parsers that could
accept different workloads would defeat that.

Every option here shrinks or shifts the workload for debugging. Any of them
makes the run non-canonical: the report says so, the id changes, and a banner is
printed to stderr. Nothing here can turn the cache or gradient checkpointing on
-- those are refused by the spec, not by this parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .level4 import Level4Workload
from .spec import WorkloadSpecError


def add_override_arguments(
    parser: argparse.ArgumentParser, workload: Level4Workload
) -> argparse.ArgumentParser:
    """The debug overrides, shared by every entry point so the commands cannot
    drift into accepting different workloads.

    Help text quotes this workload's own canonical values rather than a fixed
    string, so ``--help`` describes the model in front of you.
    """
    canonical = workload.canonical
    arch = canonical.arch
    parser.add_argument("--device", default=None,
                        help=f"debug override (canonical: {canonical.device})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"debug override (canonical: {canonical.batch_size})")
    parser.add_argument("--seq-len", type=int, default=None,
                        help=f"debug override (canonical: {canonical.seq_len})")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help=f"debug override (canonical: {canonical.dtype})",
    )
    parser.add_argument(
        "--attn",
        dest="attn_implementation",
        choices=("sdpa", "eager"),
        default=None,
        help=f"debug override (canonical: {canonical.attn_implementation})",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help=f"debug override (canonical: {canonical.seed})")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="debug override: shrink num_hidden_layers "
             f"(canonical: {arch.get('num_hidden_layers')})",
    )
    return parser


def resolve_spec(args: argparse.Namespace, workload: Level4Workload):
    """Apply the command line's overrides to this workload's canonical spec."""
    overrides = {
        key: getattr(args, key)
        for key in ("device", "batch_size", "seq_len", "dtype",
                    "attn_implementation", "seed")
        if getattr(args, key, None) is not None
    }
    if getattr(args, "layers", None) is not None:
        overrides["arch"] = {"num_hidden_layers": args.layers}
    canonical = workload.canonical
    return canonical.replace(**overrides) if overrides else canonical


def warn_if_non_canonical(spec, workload: Level4Workload) -> None:
    """A shrunk run is useful; a shrunk run mistaken for the reference is not."""
    if spec.is_canonical:
        return
    print(
        "WARNING: non-canonical workload -- this run is a debug variant and its "
        f"numbers must not be reported as the canonical result.\n"
        f"         canonical: {workload.canonical.workload_id}\n"
        f"         this run:  {spec.workload_id}",
        file=sys.stderr,
    )


# ── entry points ─────────────────────────────────────────────────────────────


def smoke_main(workload: Level4Workload, argv: list[str] | None = None,
               description: str | None = None) -> int:
    """Run the canonical training smoke and write its JSON report."""
    from .smoke import run_smoke

    parser = argparse.ArgumentParser(
        prog=f"python -m {workload.package}",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON report here")
    add_override_arguments(parser, workload)
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="print the resolved workload id and hashes, then exit without running",
    )
    args = parser.parse_args(argv)

    try:
        spec = resolve_spec(args, workload)
    except WorkloadSpecError as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 2

    warn_if_non_canonical(spec, workload)

    if args.print_spec:
        print(f"workload_id   {spec.workload_id}")
        print(f"workload_hash {spec.workload_hash}")
        print(f"config_hash   {spec.config_hash}")
        print(f"canonical     {spec.is_canonical}")
        return 0

    report = run_smoke(workload, spec)
    print(report.summary())
    if args.out is not None:
        print(f"wrote {report.write(args.out)}")
    else:
        print(report.to_json())
    return 0 if report.ok else 1


def harvest_main(workload: Level4Workload, argv: list[str] | None = None,
                 description: str | None = None) -> int:
    """Run the canonical step under observation and write its manifest.

    Unlike the smoke run, a failure here produces no file. A smoke report that
    says "it failed" is useful; a manifest missing a boundary is worse than no
    manifest, because everything derived from it would inherit the gap silently.
    """
    import time

    from .harvest import run_harvest
    from .manifest import summarize, write_manifest

    parser = argparse.ArgumentParser(
        prog=f"python -m {workload.package}.harvest.harvest",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="write the manifest here")
    add_override_arguments(parser, workload)
    args = parser.parse_args(argv)

    try:
        spec = resolve_spec(args, workload)
    except WorkloadSpecError as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 2

    warn_if_non_canonical(spec, workload)

    started = time.perf_counter()
    try:
        manifest = run_harvest(workload, spec)
    except Exception as exc:  # no partial manifest, on purpose
        print(f"harvest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize(manifest, label=f"{workload.label} harvest"))
    print(f"\nharvested in {time.perf_counter() - started:.1f}s")
    if args.out is not None:
        print(f"wrote {write_manifest(manifest, args.out)}")
    return 0
