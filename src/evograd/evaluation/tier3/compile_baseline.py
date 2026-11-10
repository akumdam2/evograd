"""Whole-model compile baseline construction."""
from __future__ import annotations

import torch

from evograd.evaluation.tier3.patch import KernelSet, KernelSource, PatchProvenance

COMPILE_SETTINGS = {"backend": "inductor", "mode": None, "dynamic": False, "fullgraph": False}
ORIGIN = "whole_model_torch_compile"


def whole_model_compile_kernels(registry):
    return KernelSet(registry=registry, sources=(KernelSource(
        site="__whole_model__", origin=ORIGIN),))


def whole_model_compile_requested(kernels):
    return any(source.origin == ORIGIN for source in kernels.sources)


def compile_model(model, kernels):
    if kernels.patched:
        raise ValueError("whole-model compile requires an unpatched model")
    return torch.compile(model, **COMPILE_SETTINGS), PatchProvenance(
        method=ORIGIN, requested_sites=(), actual_sites=(), paths={"__whole_model__": ("model",)})


def require_compiled_graphs(row):
    """A successful timing with no captured graphs is not a compile baseline."""
    row["compile_settings"] = dict(COMPILE_SETTINGS)
    if row.get("ok") and not (row.get("dynamo") or {}).get("unique_graphs", 0):
        row.update(ok=False, failed_at="compile_coverage",
                   error="whole-model compile captured no graphs")
