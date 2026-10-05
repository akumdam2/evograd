"""Untimed seed acceptance through deployment and required benchmark shapes.

Small correctness cases use the existing numerical oracle and tolerances.
Benchmark cases check execution, output/gradient contracts and finiteness, not
a second numerical policy or performance score. Run in the pipeline's killable
verification subprocess so a broken CUDA candidate cannot stop repair.
"""

from __future__ import annotations

import traceback

import torch

from evograd.evaluation.tier2.runner import candidate_module, check_module
from evograd.opdecl.activity import bind_shape
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import (
    arg_dtype_name,
    as_output_tuple,
    make_case_inputs,
    resolve_dtype,
    upstream_grad_values,
)


def _require_tensor(name, actual, shape, dtype):
    if not torch.is_tensor(actual):
        raise ValueError(f"{name}: no tensor produced")
    if tuple(actual.shape) != tuple(shape) or actual.dtype != dtype:
        raise ValueError(
            f"{name}: expected {tuple(shape)}/{dtype}, got "
            f"{tuple(actual.shape)}/{actual.dtype}"
        )
    if not bool(torch.isfinite(actual).all()):
        raise ValueError(f"{name}: non-finite values")


def _check_outputs(op, output, workload):
    """Every declared output, against the declaration rather than an allocation.

    The expected dtype comes from ``arg_dtype_name``, which is what an output
    with a fixed dtype (an FP32 auxiliary result under a BF16 workload) declares
    and what the case builder itself honours. Reading it off the upstream
    gradient tensor instead would make the check depend on how the inputs
    happened to be allocated, and would silently pass a task whose
    ``make_inputs`` hook is the thing that is wrong.
    """
    for declared, actual in zip(op.outputs, as_output_tuple(op, output)):
        _require_tensor(
            declared.name, actual, bind_shape(declared.shape, workload.dims),
            resolve_dtype(arg_dtype_name(declared, workload)),
        )


def _check_gradients(op, grads, values):
    for arg in op.active_args():
        expected = values[arg.name]
        _require_tensor(arg.grad_name, grads.get(arg.grad_name), expected.shape, expected.dtype)


def _pair_smoke(op, candidate, workload, device):
    values = make_case_inputs(op, workload, device=device)
    forward, backward = lookup_pair(op, candidate)
    args = [values.get(a.name, getattr(a, "default", None)) for a in op.args]
    with torch.no_grad():
        output, saved = forward(*args)
        _check_outputs(op, output, workload)
        grads = backward(
            upstream_grad_values(op, values), saved,
            **backward_inactive_kwargs(op, backward, values),
        )
        grads = (grads,) if torch.is_tensor(grads) else tuple(grads)
        if len(grads) != len(op.grad_names()):
            raise ValueError(f"expected {len(op.grad_names())} gradients, got {len(grads)}")
        _check_gradients(op, dict(zip(op.grad_names(), grads)), values)


def _deployment_smoke(op, module, values, workload):
    activations = [a.detach().clone().requires_grad_(True) for a in module.activations(values)]
    module.zero_grad(set_to_none=True)
    output = module(*activations)
    _check_outputs(op, output, workload)
    upstream = upstream_grad_values(op, values)
    torch.autograd.backward(
        as_output_tuple(op, output), upstream if isinstance(upstream, tuple) else (upstream,)
    )
    by_name = dict(zip(module._activation_names, activations))
    by_name.update({name: getattr(module, name) for name in module._parameter_names})
    _check_gradients(op, {a.grad_name: by_name[a.name].grad for a in op.active_args()}, values)


def check_seed_acceptance(op, candidate, *, benchmark_suite=None, device="cuda",
                          correctness=None):
    """Return fail-fast, shape-labelled feedback for the synthesis repair loop.

    Direct-pair correctness is checked by the caller first. Deployment checks
    use the same correctness set. A benchmark failure must reject the seed
    before the synthesis loop can publish it as ``best``.
    """
    benchmarks = op.benchmark_workloads(suite=benchmark_suite)
    correctness = op.correctness if correctness is None else correctness
    reports = []
    stages = (
        ("deployment_correctness", correctness),
        ("benchmark_pair", benchmarks),
        ("benchmark_deployment", benchmarks),
    )
    for stage, cases in stages:
        for workload in cases:
            row = {"stage": stage, "dims": dict(workload.dims), "dtype": workload.dtype}
            try:
                if stage == "benchmark_pair":
                    _pair_smoke(op, candidate, workload, device)
                else:
                    # A separate scope releases graph/module state between cases.
                    row.update(_deployment_case(op, candidate, workload, device, stage))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                row.setdefault("ok", True)
            except Exception as exc:
                row.update(ok=False, error_type=type(exc).__name__, error_message=str(exc),
                           traceback=traceback.format_exc(limit=8))
            reports.append(row)
            if not row["ok"]:
                return {"ok": False, "cases": reports, "failure": row}
    return {"ok": True, "cases": reports, "benchmark_suite": benchmark_suite,
            "benchmark_case_count": len(benchmarks),
            "benchmark_check": "execution, shape, dtype and finiteness; no numerical oracle"}


def _deployment_case(op, candidate, workload, device, stage):
    values = make_case_inputs(op, workload, device=device)
    module = candidate_module(op, candidate, values=values)
    if stage == "deployment_correctness":
        report = check_module(op, module, values, workload)
        row = {**report, "adapter_kind": module.adapter_kind}
        if not report.get("ok", False):
            row["error_message"] = _summarize_failed_checks(report.get("checks", {}))
        return row
    _deployment_smoke(op, module, values, workload)
    return {"ok": True, "adapter_kind": module.adapter_kind}


def _summarize_failed_checks(checks) -> str:
    """Name the tensors that failed, ahead of the full per-tensor detail.

    The whole report is handed to the repair prompt, so the detail is never
    lost -- but the numerical gate reports a nested dict, and a repair prompt
    that opens with "which results are wrong" is easier to act on than one that
    opens with a dict the model has to search.
    """
    failed = []
    for name, check in sorted(checks.items()):
        if isinstance(check, dict) and not check.get("ok", True):
            detail = ", ".join(
                f"{key}={check[key]}"
                for key in ("reason", "shape", "expected_shape", "dtype",
                            "expected_dtype", "max_abs", "max_rel")
                if key in check
            )
            failed.append(f"{name}({detail})" if detail else name)
    return ("numerical/contract check failed for: " + "; ".join(failed)
            if failed else "numerical/contract check failed")


def main(argv=None):
    """Isolated seed acceptance entry point; no timing or dtype filtering."""
    import argparse
    import json
    from dataclasses import replace
    from pathlib import Path

    from evograd.benchmark import get_task, load_task
    from evograd.evolve.evaluator import _capture_native_output, _native_output_tail
    from evograd.opdecl.verify import verify
    from evograd.opdecl.verify_cli import load_candidate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration")
    parser.add_argument("--forward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-suite")
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)

    payload = {"metrics": {"correct": 0.0}}
    native_path = None
    try:
        op = load_task(args.declaration) if args.declaration else get_task(args.op)
        if op.name != args.op:
            raise ValueError(f"declaration {op.name!r} does not match --op {args.op!r}")
        if args.forward:
            op = replace(op, forward=args.forward)
            op.validate()
        if not op.correctness:
            raise ValueError("seed acceptance requires declared correctness cases")
        op.benchmark_workloads(suite=args.benchmark_suite)
        with _capture_native_output() as captured:
            native_path = captured
            candidate = load_candidate(args.candidate)
            report = verify(op, candidate, device=args.device)
            payload["verify"] = report.to_dict()
            if report.ok:
                acceptance = check_seed_acceptance(
                    op, candidate, device=args.device, benchmark_suite=args.benchmark_suite,
                )
                payload["seed_acceptance"] = acceptance
                payload["metrics"]["correct"] = float(acceptance["ok"])
                if acceptance["ok"] and hasattr(candidate, "UNIFIED_DEPLOYMENT_CONFIG"):
                    from evograd.pipelines.unified.validation import validate_numerical_variants

                    variants = validate_numerical_variants(op, candidate, device=args.device)
                    payload["numerical_variants"] = variants
                    payload["metrics"]["correct"] = float(variants["ok"])
    except Exception:
        payload["metrics"]["correct"] = 0.0
        payload["error"] = traceback.format_exc(limit=12)
    finally:
        native = _native_output_tail(native_path)
        if native and native.get("bytes", 0) > 0:
            payload["native_output"] = native
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["metrics"]["correct"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
