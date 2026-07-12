"""Correctness gate: a candidate seed vs the autograd oracle.

One implementation for every operator — the declaration supplies workloads,
tolerances, gradient names, and input construction. Replaces the correctness
half of the per-bench evaluators and ``test_correctness.py`` files.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import torch

from evograd.opdecl.activity import OpDecl, Workload
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle


@dataclass(frozen=True)
class TensorCheck:
    name: str
    ok: bool
    max_abs_err: float
    atol: float
    rtol: float
    actual_shape: tuple[int, ...] = ()
    expected_shape: tuple[int, ...] = ()
    actual_dtype: str = ""
    expected_dtype: str = ""


@dataclass(frozen=True)
class CaseResult:
    dims: dict[str, int]
    dtype: str
    checks: tuple[TensorCheck, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(c.ok for c in self.checks)


@dataclass(frozen=True)
class VerifyReport:
    op_name: str
    cases: tuple[CaseResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(c.ok for c in self.cases)

    def to_dict(self) -> dict:
        return {
            "op": self.op_name,
            "ok": self.ok,
            "cases": [
                {
                    "dims": c.dims,
                    "dtype": c.dtype,
                    "ok": c.ok,
                    "error": c.error,
                    "checks": [
                        {
                            "name": t.name,
                            "ok": t.ok,
                            "max_abs_err": t.max_abs_err,
                            "atol": t.atol,
                            "rtol": t.rtol,
                            "actual_shape": list(t.actual_shape),
                            "expected_shape": list(t.expected_shape),
                            "actual_dtype": t.actual_dtype,
                            "expected_dtype": t.expected_dtype,
                        }
                        for t in c.checks
                    ],
                }
                for c in self.cases
            ],
        }


def _check(name: str, actual, expected, atol: float, rtol: float) -> TensorCheck:
    if not torch.is_tensor(actual):
        return TensorCheck(
            name=name,
            ok=False,
            max_abs_err=float("inf"),
            atol=atol,
            rtol=rtol,
            expected_shape=tuple(expected.shape),
            actual_dtype=type(actual).__name__,
            expected_dtype=str(expected.dtype),
        )
    same_shape = tuple(actual.shape) == tuple(expected.shape)
    same_dtype = actual.dtype == expected.dtype
    if same_shape:
        max_abs_err = (actual.float() - expected.float()).abs().max().item()
        values_ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    else:
        max_abs_err = float("inf")
        values_ok = False
    return TensorCheck(
        name=name,
        ok=bool(same_shape and same_dtype and values_ok),
        max_abs_err=max_abs_err,
        atol=atol,
        rtol=rtol,
        actual_shape=tuple(actual.shape),
        expected_shape=tuple(expected.shape),
        actual_dtype=str(actual.dtype),
        expected_dtype=str(expected.dtype),
    )


def _run_case(op: OpDecl, module, workload: Workload, device: str) -> CaseResult:
    dims, dtype = dict(workload.dims), workload.dtype
    try:
        inputs = make_case_inputs(op, workload, device=device)
        y_ref, ref_grads = oracle(op, inputs)

        fwd, bwd = lookup_pair(op, module)
        positional = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
        y, saved = fwd(*positional)
        kwargs = backward_inactive_kwargs(op, bwd, inputs)
        grads = bwd(inputs[op.upstream_grad_name], saved, **kwargs)
        grads = (grads,) if torch.is_tensor(grads) else tuple(grads)

        atol, rtol = op.tolerance_for(workload)
        checks = [_check(op.output.name, y, y_ref, atol, rtol)]
        if len(grads) != len(op.grad_names()):
            raise ValueError(
                f"backward returned {len(grads)} gradients, "
                f"contract requires {len(op.grad_names())}: {op.grad_names()}"
            )
        for name, grad in zip(op.grad_names(), grads):
            atol, rtol = op.tolerance_for(workload, name)
            checks.append(_check(name, grad, ref_grads[name], atol, rtol))
        return CaseResult(dims=dims, dtype=dtype, checks=tuple(checks))
    except Exception:
        return CaseResult(dims=dims, dtype=dtype, error=traceback.format_exc())


def verify(
    op: OpDecl,
    module,
    device: str = "cuda",
    workloads: tuple[Workload, ...] | None = None,
) -> VerifyReport:
    """Compare a seed module's backward against the oracle on every workload.

    ``workloads`` defaults to the declaration's correctness set.
    """
    selected = workloads if workloads is not None else op.correctness
    return VerifyReport(
        op_name=op.name,
        cases=tuple(_run_case(op, module, w, device) for w in selected),
    )
