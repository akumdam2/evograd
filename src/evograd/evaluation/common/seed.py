"""Seed acceptance: Gate 1 on a model case's search payloads, then its deployment.

One rule for every seed pipeline. A seed is accepted when

1. Gate 1 passes on the calibration payloads the case's task pack names for
   search (:func:`evograd.verification.payloads.gate1.search_correctness`), the
   rule search feedback is judged by; and
2. its deployment, the ``torch.autograd.Function`` a model installs, runs
   forward and backward on the first of those payloads with every declared
   output and gradient present, of the declared shape and dtype, and finite.

The deployment's values are judged by operator Gate 2 at evaluation, not here.
No input is generated. Runs in the pipeline's
killable verification subprocess so a broken CUDA candidate cannot stop repair.
"""

from __future__ import annotations

import traceback

import torch

from evograd.evaluation.tier2.runner import candidate_module
from evograd.opdecl.activity import bind_shape
from evograd.opdecl.inputs import arg_dtype_name, as_output_tuple, resolve_dtype, upstream_grad_values


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
    """Every declared output, against the declaration rather than the payload.

    The expected dtype comes from ``arg_dtype_name``, which is what an output
    with a fixed dtype (an FP32 auxiliary result under a BF16 workload)
    declares. Reading it off the recorded cotangent instead would let the
    payload, not the declaration, decide what the candidate must return.
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


def accept(task, candidate, *, device: str = "cuda") -> tuple[bool, dict]:
    """``(accepted, report)`` for ``candidate`` on task pack ``task`` (an ``AgentTask``)."""
    from evograd.verification.payloads import gate1 as payload_gate1
    from evograd.verification.payloads.corpus import Corpus

    spec = task.spec.get("search_correctness") or {}
    if spec.get("rule") != payload_gate1.SEARCH_RULE:
        raise ValueError(f"{task.root}: seed acceptance needs a model case's task pack "
                         f"(search rule {payload_gate1.SEARCH_RULE!r}), not {spec.get('rule')!r}")
    op = task.op()
    report: dict = {"pack": str(task.root), "case_id": spec["case_id"]}
    summary = payload_gate1.search_correctness(op, candidate, spec, device=device)
    report["search_correctness"] = summary
    if summary["passed"] != summary["total"]:
        return False, report
    payload_id = spec["payload_ids"][0]
    row = {"stage": "deployment", "payload_id": payload_id}
    try:
        (workload,) = op.benchmark
        values = Corpus.open(spec["corpus"]["manifest"]).load(payload_id, device=device)
        module = candidate_module(op, candidate, values=values)
        _deployment_smoke(op, module, values, workload)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        row.update(ok=True, adapter_kind=module.adapter_kind)
    except Exception as exc:
        row.update(ok=False, error_type=type(exc).__name__, error_message=str(exc),
                   traceback=traceback.format_exc(limit=8))
    report["deployment"] = row
    return row["ok"], report


def main(argv=None):
    """Isolated seed acceptance entry point; prints the JSON report."""
    import argparse
    import json
    from pathlib import Path

    from evograd.adbench_tools.task import load as load_pack
    from evograd.evolve.evaluator import _capture_native_output, _native_output_tail
    from evograd.opdecl.importing import load_candidate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="task pack of one model case")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)

    payload = {"metrics": {"correct": 0.0}}
    native_path = None
    try:
        task = load_pack(args.pack)
        with _capture_native_output() as captured:
            native_path = captured
            accepted, report = accept(task, load_candidate(args.candidate), device=args.device)
        payload.update(report)
        payload["metrics"]["correct"] = float(accepted)
    except Exception:
        payload["error"] = traceback.format_exc(limit=12)
    finally:
        native = _native_output_tail(native_path)
        if native and native.get("bytes", 0) > 0:
            payload["native_output"] = native
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["metrics"]["correct"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
