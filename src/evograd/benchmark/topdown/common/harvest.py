"""Harvest a canonical training step into a workload manifest.

    PYTHONPATH=src python -m <workload package>.harvest.harvest \
        --out results/harvest.json

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

from .level4 import Level4Workload
from .manifest import build_manifest, summarize, write_manifest
from .model import check_effective_settings, effective_settings, training_step
from .observe import check_mandatory_boundaries, observe
from .smoke import environment_info, gradient_coverage, workload_info
from .spec import WorkloadSpec


def run_harvest(workload: Level4Workload,
                spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """Execute the canonical step under observation and return the manifest.

    Raises on any failure -- a bad effective setting, a missing mandatory
    boundary, a non-finite loss, an incomplete gradient. The caller decides what
    to do with that; nothing is written here.
    """
    spec = workload.resolve(spec)
    workload.require_transformers()
    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "the canonical harvest runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a (non-canonical) debug run"
        )
    if spec.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model = workload.build_model(spec)
    input_ids, labels = workload.make_inputs(spec)

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

    with observe(model, workload_id=spec.workload_id, config_hash=spec.config_hash,
                 plan=workload.plan) as observation:
        outputs = training_step(model, input_ids, labels)

    # Outside the context on purpose: whatever these read, they read from an
    # unobserved process, which is also what proves the observer let go.
    check_mandatory_boundaries(observation, workload.plan)

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
        schema_version=workload.manifest_schema,
        function_wrappers=workload.function_wrappers,
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
