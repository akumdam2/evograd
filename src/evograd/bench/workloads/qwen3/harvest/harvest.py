"""Harvest the canonical Qwen3 training step into a workload manifest.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.harvest \
        --out results/qwen3-level4/harvest.json

This is the Level-4 smoke run with an observer attached. It reuses
``build_model``, ``make_inputs``, ``training_step``, the effective-setting
checks and the gradient validation unchanged -- the point of the milestone is
that the harvest describes *the* canonical execution, and a second, subtly
different implementation of it would defeat that.

Unlike the smoke run, a failure here does not produce a file. A smoke report
that says "it failed" is useful; a manifest missing a boundary is worse than no
manifest, because everything derived from it would inherit the gap silently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .manifest import build_manifest, summarize, write_manifest
from ..levels.level4.model import (
    build_model,
    check_effective_settings,
    effective_settings,
    make_inputs,
    require_transformers,
    training_step,
)
from .observe import check_mandatory_boundaries, observe
from ..levels.level4.smoke import environment_info, gradient_coverage, workload_info
from ..levels.level4.spec import CANONICAL, WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """Execute the canonical step under observation and return the manifest.

    Raises on any failure -- a bad effective setting, a missing mandatory
    boundary, a non-finite loss, an incomplete gradient. The caller decides what
    to do with that; nothing is written here.
    """
    spec = (spec or CANONICAL).validate()
    require_transformers()
    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "the canonical harvest runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a (non-canonical) debug run"
        )
    if spec.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model = build_model(spec)
    input_ids, labels = make_inputs(spec)

    effective = effective_settings(model, spec)
    effective["input_ids_dtype"] = str(input_ids.dtype)
    effective["input_ids_device"] = input_ids.device.type
    effective["input_ids_shape"] = list(input_ids.shape)
    effective["input_ids_checksum"] = int(input_ids.sum().item())
    effective["labels_match_input_ids"] = bool(torch.equal(input_ids, labels))
    problems = check_effective_settings(effective, spec)
    if problems:
        raise RuntimeError(
            "the built model does not match the requested workload: " + "; ".join(problems)
        )

    with observe(model, workload_id=spec.workload_id, config_hash=spec.config_hash) as observation:
        outputs = training_step(model, input_ids, labels)

    # Outside the context on purpose: whatever these read, they read from an
    # unobserved process, which is also what proves the observer let go.
    check_mandatory_boundaries(observation)

    loss = outputs.loss
    if spec.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    effective["loss_dtype"] = str(loss.dtype)
    effective["logits_dtype"] = str(outputs.logits.dtype)
    effective["returned_past_key_values"] = outputs.past_key_values is not None

    loss_value = float(loss.detach().float().item())
    coverage = gradient_coverage(model)
    validation = {
        "loss": loss_value,
        "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
        **coverage,
        "effective": effective,
    }
    if not validation["loss_is_finite"]:
        raise RuntimeError(f"loss is not finite: {loss_value}")
    if coverage["missing_grad_params"]:
        raise RuntimeError(
            f"{len(coverage['missing_grad_params'])} trainable parameters received no "
            f"gradient, first: {coverage['missing_grad_params'][:5]}"
        )
    if coverage["non_finite_grad_params"]:
        raise RuntimeError(
            f"{len(coverage['non_finite_grad_params'])} parameter gradients are not finite, "
            f"first: {coverage['non_finite_grad_params'][:5]}"
        )

    return build_manifest(
        observation,
        workload=workload_info(spec),
        environment=environment_info(spec),
        validation=validation,
        diagnostics={
            "note": "diagnostic only -- one observed step, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if spec.device.startswith("cuda") else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if spec.device.startswith("cuda") else None
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    from ..cli import add_override_arguments

    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.harvest.harvest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON manifest here")
    parser.add_argument(
        "--summary-out", type=Path, default=None, help="also write the text summary here"
    )
    add_override_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    from ..cli import resolve_spec
    from ..levels.level4.spec import WorkloadSpecError

    args = build_parser().parse_args(argv)
    try:
        spec = resolve_spec(args)
    except WorkloadSpecError as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 2

    if not spec.is_canonical:
        print(
            "WARNING: non-canonical workload -- this harvest is a debug variant and its "
            "manifest must not be reported as the canonical one.\n"
            f"         canonical: {CANONICAL.workload_id}\n"
            f"         this run:  {spec.workload_id}",
            file=sys.stderr,
        )

    try:
        manifest = run_harvest(spec)
    except Exception as exc:
        print(f"harvest failed, no manifest written: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = summarize(manifest)
    print(summary)
    if args.out is not None:
        print(f"\nwrote {write_manifest(manifest, args.out)}")
    if args.summary_out is not None:
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
