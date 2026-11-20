"""Operator Gate 2: the accuracy of one operator call, against the float32 truth.

On every recorded payload ``x`` of a case, four computations on that payload's
exact values: the **truth** ``t`` (Gate 1's: the model reference with its floating
inputs below float32 widened to float32, TF32 off; for a call already at float32
it is native itself), and three executions through the operator scope's autograd
module -- **native** (the model reference at the real dtypes), the
**torch.compile** control (native under inductor) and the **candidate** (through
its deployment entry). Every output, input gradient and parameter gradient ``z``
is measured against the truth, in float64::

    E2(a, t) = ||a - t||_2 / ||t||_2        the result's overall error
    M(a, t)  = max|a - t| / max|t|          its worst element, in units of its largest value

The candidate passes when, on every payload, both are within the result's
threshold::

    T[z, m] = max( FACTOR * max over payloads x and trusted r in (native, compile) of m(r_x[z], t_x[z]),
                   FLOORS[dtype of z] )

``err(candidate) <= max(floor, factor * err(trusted))``: Gate 1's form, with both
trusted implementations and a floor of one rounding instead of Gate 1's lenient
budget. One threshold per result per case, pooled over every payload as Gate 1
pools native's error. ``FACTOR`` 2 follows FlashAttention's tests (a kernel's error
at most twice that of the PyTorch implementation against an fp32 reference). The
floor matters only where both trusted implementations are (nearly) exact: a
bfloat16 or float16 result's unit roundoff (2^-8, 2^-11; one correct rounding
gives E2 <= u; flash-linear-attention and DeepGEMM hold bfloat16 kernels to an E2
of about 4e-3 to 5e-3), and 1e-5 for float32, which torch.testing and
TransformerEngine also use (different float32 algorithms differ by up to 1.4e-6 on
the benchmark's calls). A result whose truth is exactly zero has error 0 when the
candidate is zero too and is infinite otherwise; a trusted implementation's
infinite error there widens nothing.

What it catches: precision lost by a kernel that computes the right thing --
accumulation in bfloat16 (Liger's fused linear cross-entropy weight gradient,
1.3%), dropped float32 upcasts, a small systematic scale (``E2``), or a few
elements computed less accurately than the trusted implementations (``M``). What
it leaves to others: gross errors and input mutation (Gate 1), propagation
through the model and through training (model Gate 2, Gate 3).

If the compile control cannot be built or fails on a payload, native alone is
trusted there (recorded in the report). Nothing is calibrated or cached: the
trusted errors are measured on the same payloads in the same run.
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Mapping, Sequence

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import NATIVE, Deployment, Sample
from evograd.verification.common.reports import (
    CaseReport,
    ResultCheck,
    VerificationReport,
    failure,
    missing,
)
from evograd.verification.common.tensors import relative_error
from evograd.verification.gate2.controls import compile_control
from evograd.verification.gate2.runner import GATE, execute, integration_contract, structure_checks

PROTOCOL = "evograd-verification-gate2-operator/1"
RULE = "evograd-gate2-operator-truth/1"
#: Twice the trusted implementations' error, as in FlashAttention's tests.
FACTOR = 2.0
#: The least threshold, by the result's dtype.
FLOORS = {"bfloat16": 3.9e-3, "float16": 4.9e-4, "float32": 1e-5, "float64": 1e-12}
METRICS = ("E2", "M")
TRUSTED = ("native", "compile")
COMPARISON = ("E2 = ||c-t||_2 / ||t||_2 and M = max|c-t| / max|t| per result and payload, "
              "float64, t = float32 truth; T[z,m] = max(FACTOR * max over payloads and trusted "
              "(native, torch.compile) of m, FLOORS[dtype of z])")


def rule_identity() -> dict[str, Any]:
    """The rule, as recorded in reports and stage keys."""
    return {"rule": RULE, "factor": FACTOR, "floors": dict(FLOORS), "metrics": list(METRICS),
            "trusted": list(TRUSTED), "pooling": "per result, the maximum over every payload",
            "truth": "model reference with floating inputs below float32 widened to float32, "
                     "TF32 off", "comparison": COMPARISON}


def errors(candidate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """``E2`` and ``M`` of ``candidate`` against ``truth``, in float64 (both finite)."""
    err = relative_error(candidate, truth)
    if err.bitwise:
        return {"E2": 0.0, "M": 0.0}
    t = truth.detach()
    peak = max(abs(float(t.max())), abs(float(t.min()))) if t.numel() else 0.0
    return {"E2": err.diff_l2 / err.ref_l2 if err.ref_l2 > 0 else math.inf,
            "M": err.diff_max / peak if peak > 0 else math.inf}


def truth_names(op) -> dict[str, str]:
    """The oracle's result names (Gate 1's) to the operator scope's."""
    params = set(op.parameter_args or ())
    names = {C.PREFIX[C.OUTPUT] + n: C.PREFIX[C.OUTPUT] + n for n in op.output_names}
    for arg in op.active_args():
        kind = C.PARAM_GRAD if arg.name in params else C.INPUT_GRAD
        names[C.PREFIX[C.INPUT_GRAD] + arg.grad_name] = C.PREFIX[kind] + arg.name
    return names


def truth(scope, sample: Sample) -> dict[str, Any] | None:
    """The float32 truth on ``sample`` by the scope's result names; ``None`` when the
    call is already at float32 or above (native is then the truth)."""
    from evograd.verification.gate1.runner import truth_results

    out = truth_results(scope.op, scope.values(sample), scope.device, None)
    if out is None:
        return None
    names = truth_names(scope.op)
    return {names[k]: v for k, v in out.items() if k in names}


def verify_operator(scope, deployment: Deployment, *, samples: Sequence[Sample],
                    compile_backend: str = "inductor", compile_fullgraph: bool = True
                    ) -> VerificationReport:
    """Judge ``deployment`` on every one of ``samples`` (the case's payloads)."""
    from evograd.opdecl.oracle import reference_identity

    report = VerificationReport(
        gate=GATE, protocol=PROTOCOL,
        subject={"scope": scope.scope, "adapter": dict(scope.identity()),
                 "numerics": dict(scope.numerics()), "deployment": deployment.describe(),
                 "reference": {"kind": "float32_truth", "implementation": reference_identity(scope.op)},
                 "rule": rule_identity(), "samples": [s.to_dict() for s in samples]})
    if deployment.role == NATIVE:
        report.findings.append(missing("not_a_candidate", "the native deployment is trusted, not judged"))
    try:
        control, graphs = compile_control(scope, scope.sites(), backend=compile_backend,
                                          fullgraph=compile_fullgraph)
        report.evidence["compile"] = {"deployment": control.describe(), "graphs": graphs}
    except Exception as exc:  # native alone is trusted
        control = None
        report.evidence["compile"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    trusted: dict[str, dict[str, float]] = {}       # result -> metric -> largest trusted error
    source: dict[str, dict[str, str]] = {}          # result -> metric -> which trusted set it
    dtypes: dict[str, str] = {}
    judged: list[ResultCheck] = []
    for sample in samples:
        case = CaseReport(case_id=sample.sample_id, evidence={"sample": sample.to_dict()})
        report.cases.append(case)
        measured = _measure(scope, sample, deployment, control, case)
        if measured is None:
            continue
        for label, by_result in measured["trusted"].items():
            for name, errs in by_result.items():
                for metric, value in errs.items():
                    # A trusted residue on an exactly-zero truth (infinite) widens nothing:
                    # there, as in Gate 1, only an exactly-zero result passes.
                    if math.isfinite(value) and value > trusted.setdefault(name, {}).get(metric, -1.0):
                        trusted[name][metric] = value
                        source.setdefault(name, {})[metric] = label
        dtypes.update(measured["dtypes"])
        judged.extend(measured["checks"])
    thresholds: dict[str, dict[str, Any]] = {}
    for name, dtype in sorted(dtypes.items()):
        floor = FLOORS.get(dtype)
        entry: dict[str, Any] = {"dtype": dtype, "floor": floor}
        for metric in METRICS:
            own = trusted.get(name, {}).get(metric)       # None: every trusted error was infinite
            if floor is None:
                entry[metric] = None
                continue
            binds = own is not None and FACTOR * own > floor
            entry[metric] = {"threshold": FACTOR * own if binds else floor, "trusted": own,
                             "from": source[name][metric] if binds else "floor"}
        thresholds[name] = entry
    report.subject["thresholds"] = thresholds
    for check in judged:
        entry = thresholds.get(check.name) or {}
        limits = {m: (entry.get(m) or {}).get("threshold") for m in METRICS}
        check.limits = {**limits, "dtype": entry.get("dtype")}
        if any(v is None for v in limits.values()):
            check.findings.append(missing("no_threshold", f"no floor for a {entry.get('dtype')} "
                                          "result", subject=check.name))
            continue
        for metric in METRICS:
            value, bound = check.metrics[metric], limits[metric]
            if not value <= bound:
                check.findings.append(failure(
                    "threshold_exceeded",
                    f"{metric} {value:.3g} > {bound:.3g} ({value / bound:.3g}x; "
                    f"{_basis(entry[metric])})", subject=check.name, metric=metric,
                    value=value, threshold=bound))
    return report


def _basis(entry: Mapping[str, Any]) -> str:
    if entry["from"] == "floor":
        return "the dtype's floor"
    return f"{FACTOR:g} x {entry['from']}'s error {entry['trusted']:.3g}"


def _measure(scope, sample: Sample, deployment: Deployment, control: Deployment | None,
             case: CaseReport) -> dict[str, Any] | None:
    """One payload: native, the compile control and the candidate, then the truth.

    The truth runs last and alone (the others keep only their results): at twice
    native's size it fits in device memory on the largest calls only when it never
    coexists with another execution's working state. ``None`` when the reference
    side failed, which leaves ``case`` incomplete.
    """
    native = execute(scope, Deployment.native(), sample)
    if native.error:
        case.findings.append(missing("reference_runtime_error", f"native failed at {native.error_stage}",
                                     trace=native.error))
        return None
    contract = integration_contract(native)
    runs = {"native": native.results}
    if control is not None:
        ctl = execute(scope, control, sample)
        if ctl.error:
            case.evidence["compile"] = {"failed_at": ctl.error_stage, "trace": ctl.error}
        else:
            runs["compile"] = ctl.results
        del ctl
    cand = execute(scope, deployment, sample)
    case.evidence["candidate_seconds"] = cand.seconds
    checks: dict[str, ResultCheck] = {}
    numeric: list[str] = []
    if cand.error:
        code = "output_structure" if cand.error_stage == "outputs" else "candidate_runtime_error"
        case.findings.append(failure(code, f"deployment failed at {cand.error_stage}",
                                     trace=cand.error))
    else:
        checks, numeric, loose = structure_checks(contract, cand.results, native.results)
        case.findings.extend(loose)
        runs["candidate"] = {n: cand.results[n] for n in numeric}
    del cand
    try:
        wide = truth(scope, sample)
    except Exception:
        case.findings.append(missing("truth_failed", "the float32 truth failed",
                                     trace=traceback.format_exc(limit=8)))
        return None
    target = native.results if wide is None else wide
    measured: dict[str, Any] = {"trusted": {}, "dtypes": {}, "checks": []}
    for spec in contract.compared():
        ref = native.results.get(spec.name)
        if spec.exact or not C.is_float_tensor(ref):
            continue
        t = target.get(spec.name)
        if not torch.is_tensor(t) or tuple(t.shape) != tuple(ref.shape) \
                or not bool(torch.isfinite(t).all()) or not bool(torch.isfinite(ref).all()):
            case.findings.append(missing("invalid_truth", "the truth or native is missing, "
                                         "misshapen or non-finite", subject=spec.name))
            continue
        measured["dtypes"][spec.name] = C.dtype_name(ref.dtype)
        for label in TRUSTED:
            value = runs.get(label, {}).get(spec.name)
            if torch.is_tensor(value) and tuple(value.shape) == tuple(t.shape) \
                    and bool(torch.isfinite(value).all()):
                measured["trusted"].setdefault(label, {})[spec.name] = errors(value, t)
        if spec.name in runs.get("candidate", {}):
            check = checks[spec.name]
            check.compared = True
            check.metrics = errors(runs["candidate"][spec.name], t)
            measured["checks"].append(check)
    case.evidence["trusted"] = measured["trusted"]
    case.results.extend(checks.values())
    return measured
