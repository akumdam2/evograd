"""Whole-model compile baseline construction and independent correctness evidence."""
from __future__ import annotations

import copy
import gc

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


def _canonical(values):
    # OptimizedModule adds this component to parameter names without changing
    # the underlying parameter. Preserve every other name component.
    return {name.replace("_orig_mod.", ""): value for name, value in values.items()}


def _cpu(execution):
    execution.results = {name: value.detach().cpu() if torch.is_tensor(value) else value
                         for name, value in execution.results.items()}
    return execution


def compare(reference, compiled):
    """Compare every declared output, loss and parameter/input gradient.

    This is independent baseline evidence, not candidate acceptance or a
    replacement for calibrated site policies. Use PyTorch's dtype-specific
    assert_close defaults and record the policy so failures remain visible.
    """
    failures = []
    for label, execution in (("eager", reference), ("torch_compile", compiled)):
        if execution.error:
            failures.append({"subject": label, "error": execution.error})
        if execution.mutated:
            failures.append({"subject": label, "error": f"mutated inputs: {execution.mutated}"})
    if failures:
        return failures
    if reference.input_digest != compiled.input_digest:
        failures.append({"subject": "inputs", "error": "input/cotangent content differs"})
    if _canonical(reference.initial_state) != _canonical(compiled.initial_state):
        failures.append({"subject": "weights", "error": "initial model state differs"})
    a, b = _canonical(reference.results), _canonical(compiled.results)
    if a.keys() != b.keys():
        failures.append({"subject": "results", "error": "output/gradient names differ"})
    for name in sorted(a.keys() & b.keys()):
        try:
            torch.testing.assert_close(b[name], a[name], equal_nan=False)
        except (AssertionError, TypeError, ValueError) as exc:
            failures.append({"subject": name, "error": str(exc)[:1200]})
    return failures


def check_correctness(model, workload):
    """Fresh executions on the formal model seeds, outside the timing region.

    Compile the same underlying workload model as the performance provider;
    retain the existing objective adapter and its output/backward contracts.
    Move reference results to CPU before building the compiled model.
    """
    from evograd.evaluation.protocol import whole_model_compile
    from evograd.verification.common.execution import Deployment
    from evograd.verification.integrations.base import configure
    from evograd.verification.integrations.gates import MODEL_ORDERS, model_scope
    from evograd.verification.gate2.runner import execute

    rows = []
    try:
        configure()
        native = model_scope(model, workload)
        compiled = copy.copy(native)
        compiled.workload = copy.copy(workload)
        build = compiled.workload.build_patched
        kernels = whole_model_compile(workload)
        compiled.workload.build_patched = lambda _unused: build(kernels)
        for sample in native.samples(MODEL_ORDERS):
            reference = _cpu(execute(native, Deployment.native(), sample))
            if hasattr(workload, "_last"):
                workload._last = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            actual = _cpu(execute(compiled, Deployment.native(), sample))
            if hasattr(compiled.workload, "_last"):
                compiled.workload._last = None
            failures = compare(reference, actual)
            rows.append({"seed": sample.seed, "status": "failed" if failures else "passed",
                         "failures": failures, "results_checked": len(reference.results)})
            del reference, actual
        return {"status": "passed" if all(r["status"] == "passed" for r in rows) else "failed",
                "policy": "torch.testing.assert_close dtype defaults; equal_nan=False",
                "scope": "whole-model outputs, loss, input and parameter gradients",
                "samples": rows}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "samples": rows}
    finally:
        for owner in (workload, getattr(locals().get("compiled"), "workload", None)):
            if hasattr(owner, "_last"):
                owner._last = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
