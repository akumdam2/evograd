"""Gate 1: the local pair against the declared PyTorch oracle. One core.

**What is compared.** Every declared forward output, and every gradient the
declaration requires (``op.grad_names()``), from the candidate's own forward
and backward: the backward receives the candidate forward's saved state.

**Against what.** ``OpDecl.forward`` evaluated eagerly under autograd
(:func:`evograd.opdecl.oracle.oracle`), at ``reference_dtype`` when the
declaration sets one, with the results cast back to the required dtypes. For a
model-reference declaration (``op.reference_contract``, e.g. every Qwen3-0.6B
task) ``forward`` is the pinned model's own computation, differentiated at the
operands' real dtypes; ``reference_dtype`` is refused there. The report's
``subject.reference`` records which contract applied
(:func:`evograd.opdecl.oracle.reference_identity`).

**How.** Per element, per result:
``|candidate_i - reference_i| <= max(atol_z + rtol_z * |reference_i|, envelope_z)``,
with model-reference tasks using fixed ``torch.testing.assert_close`` dtype
defaults from PyTorch 2.11 for each reference result and ``envelope_z`` gamma
times the largest elementwise deviation the trusted ``torch.compile`` control
showed on that result over the calibration payloads (passed in by the caller
from the operator Gate 2 policy; absent, 0). Declaration overrides,
multipliers and shape hooks do not apply to model tasks. Generic tasks retain
``op.tolerance_for(workload, result)``. Computed in float64.

**What this core adds to the earlier verifiers** (all listed in the package
README): explicit finiteness (a non-finite candidate fails, a non-finite
reference makes the case incomplete instead of being compared by
``allclose``'s inf==inf), result matching by name with arity reported rather
than raised, input mutation as an explicit contract, and an empty case set that
never passes.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from evograd.opdecl.activity import Active, OpDecl, Workload, bind_shape
from evograd.opdecl.inputs import (
    arg_dtype_name,
    make_case_inputs,
    resolve_dtype,
    upstream_grad_values,
)
from evograd.opdecl.oracle import oracle, reference_identity
from evograd.verification.gate1.tolerances import policy_identity, result_tolerances
from evograd.verification.common import contracts as C
from evograd.verification.common.reports import (
    CaseReport,
    ResultCheck,
    VerificationReport,
    failure,
    missing,
)
from evograd.verification.common.tensors import (
    elementwise_error,
    isolate,
    isolated_rng,
    mutations,
    snapshot,
    synchronize,
)

GATE = "gate1"
PROTOCOL = "evograd-verification-gate1/2"
SMOKE_PROTOCOL = "evograd-verification-gate1-smoke/1"

#: Input mutation policies. ``reject`` fails a candidate that changes any input
#: except those the declaration's ``backward_may_overwrite`` names (whose
#: layout must still survive); ``record`` only reports it.
REJECT = "reject"
RECORD = "record"

ORACLE = "oracle"
SMOKE = "smoke"

REQUIRED_CASE = "required"
PROBE_CASE = "probe"


@dataclass(frozen=True)
class PairUnderTest:
    """The direct pair ABI, normalized.

    ``forward(values) -> (output, saved)``; ``backward(output_grads, saved,
    values) -> gradients in op.grad_names() order``. ``prepare(values)``, when
    present, runs before the forward on a separate copy of the case inputs
    (compilation, specialization).
    """

    name: str
    forward: Callable[[dict[str, Any]], tuple[Any, Any]]
    backward: Callable[[Any, Any, dict[str, Any]], Any]
    prepare: Callable[[dict[str, Any]], None] | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Gate1Case:
    """One concrete case. ``values=None`` means "the declaration's inputs"."""

    case_id: str
    workload: Workload
    values: Mapping[str, Any] | None = None
    #: ``required`` cases decide the verdict. A ``probe`` (e.g. a stress
    #: pattern) whose reference is invalid is recorded as skipped, never passed.
    role: str = REQUIRED_CASE
    provenance: Mapping[str, Any] = field(default_factory=dict)


def case_id_for(op: OpDecl, workload: Workload, variant: str = "declared") -> str:
    dims = ",".join(f"{k}={v}" for k, v in sorted(workload.dims.items()))
    return f"{op.name}/{dims}/{workload.dtype}/{variant}"


def declared_cases(op: OpDecl, workloads: Sequence[Workload] | None = None) -> tuple[Gate1Case, ...]:
    selected = op.correctness if workloads is None else tuple(workloads)
    return tuple(Gate1Case(case_id_for(op, w), w) for w in selected)


def result_contract(op: OpDecl, workload: Workload | None = None) -> C.ResultContract:
    """Every declared output and required gradient, named ``out:`` / ``grad_in:``.

    With a workload, declared shapes and dtypes are bound into the contract
    (and checked against the reference).
    """
    def bound(arg) -> tuple[tuple[int, ...] | None, str | None]:
        if workload is None:
            return None, None
        shape = getattr(arg, "shape", None)
        dims = None
        if shape not in (None, "unspecified"):
            dims = tuple(bind_shape(shape, workload.dims))
        return dims, arg_dtype_name(arg, workload)

    specs = []
    for out in op.outputs:
        shape, dtype = bound(out)
        specs.append(C.ResultSpec(C.PREFIX[C.OUTPUT] + out.name, C.OUTPUT, shape=shape, dtype=dtype))
    by_grad = {a.grad_name: a for a in op.active_args()}
    for grad in op.grad_names():
        shape, dtype = bound(by_grad[grad])
        specs.append(C.ResultSpec(C.PREFIX[C.INPUT_GRAD] + grad, C.INPUT_GRAD, shape=shape,
                                  dtype=dtype))
    return C.ResultContract(tuple(specs))


def declared_name(result_name: str) -> str:
    """``out:y`` -> ``y``; ``grad_in:dx`` -> ``dx``: the name tolerances use."""
    return result_name.split(":", 1)[1]


# ── the core ─────────────────────────────────────────────────────────────────


def verify_pair(
    op: OpDecl,
    candidate: Any,
    *,
    cases: Sequence[Gate1Case] | None = None,
    workloads: Sequence[Workload] | None = None,
    device: str = "cuda",
    mode: str = ORACLE,
    input_mutation: str = REJECT,
    observers: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
    envelope: Mapping[str, float] | None = None,
) -> VerificationReport:
    """Run ``candidate`` on every case and judge it against the oracle.

    ``envelope`` maps a result name (``out:q``) to the least allowance of any
    of its elements: gamma times the trusted compile control's largest
    elementwise deviation on that result (see :mod:`.tolerances`).

    ``candidate`` is a :class:`PairUnderTest`, a seed module exposing
    ``forward_with_saved``/``backward_from_saved`` (generic or op-prefixed), or
    a ``PairProvider``-like object; see :func:`gate1.adapters.as_pair`.
    ``cases`` overrides ``workloads`` (default: ``op.correctness``).

    ``mode="smoke"`` runs the pair and checks only the declared shapes, dtypes
    and finiteness. Its report is never ``passed``: it says ``incomplete``
    with ``smoke_only``, because no oracle was consulted.

    ``observers`` are named callables given each executed case's context
    (values, saved state, outputs, gradients); what they return is recorded as
    evidence and never changes the verdict.
    """
    from evograd.verification.gate1.adapters import as_pair

    if mode not in (ORACLE, SMOKE):
        raise ValueError(f"mode must be {ORACLE!r} or {SMOKE!r}")
    if input_mutation not in (REJECT, RECORD):
        raise ValueError(f"input_mutation must be {REJECT!r} or {RECORD!r}")
    pair = as_pair(op, candidate)
    selected = tuple(cases) if cases is not None else declared_cases(op, workloads)
    report = VerificationReport(
        gate=GATE, protocol=PROTOCOL if mode == ORACLE else SMOKE_PROTOCOL,
        subject={"op": op.name, "candidate": pair.name, "candidate_identity": dict(pair.identity),
                 "device": str(device), "mode": mode, "input_mutation": input_mutation,
                 "reference": reference_identity(op),
                 "tolerance_policy": policy_identity(op),
                 "comparison": "|c-r| <= max(atol + rtol*|r|, envelope) per element, float64",
                 "envelope": dict(envelope or {}),
                 "cases": [c.case_id for c in selected]},
    )
    ids = [c.case_id for c in selected]
    if len(ids) != len(set(ids)):
        report.findings.append(missing("duplicate_case_ids", "case ids must be unique",
                                       repeated=sorted({i for i in ids if ids.count(i) > 1})))
    if mode == SMOKE:
        report.findings.append(missing("smoke_only", "smoke mode ran the pair without an oracle; "
                                                     "this is not a numerical verification"))
    for case in selected:
        report.cases.append(_run_case(op, pair, case, device=device, mode=mode,
                                      input_mutation=input_mutation, observers=observers or {},
                                      envelope=envelope))
    return report


def _run_case(op: OpDecl, pair: PairUnderTest, case: Gate1Case, *, device: str, mode: str,
              input_mutation: str, observers: Mapping[str, Callable],
              envelope: Mapping[str, float] | None = None) -> CaseReport:
    report = CaseReport(case_id=case.case_id, role=case.role,
                        evidence={"dims": dict(case.workload.dims), "dtype": case.workload.dtype,
                                  "provenance": dict(case.provenance)})
    contract = result_contract(op, case.workload)
    report.evidence["contract"] = contract.to_dict()

    # 1. the inputs: generated from the declaration, or supplied by the caller.
    try:
        if case.values is None:
            with isolated_rng(None, device):
                values = make_case_inputs(op, case.workload, device=device)
            report.evidence["inputs"] = "declaration"
        elif callable(getattr(case.values, "materialize", None)):
            # A recorded payload, loaded only now (exact layouts and aliases).
            values = dict(case.values.materialize())
            report.evidence["inputs"] = str(case.provenance.get("inputs", "materialized"))
        else:
            values = dict(case.values)
            report.evidence["inputs"] = "supplied"
    except Exception as exc:
        report.findings.append(missing("input_generation_failed", f"{type(exc).__name__}: {exc}",
                                       traceback=traceback.format_exc(limit=6)))
        return report
    problems = case_input_problems(op, case.workload, values, device)
    if problems:
        report.findings.extend(missing("invalid_case_inputs", p) for p in problems)
        return report

    # 2. the reference, on its own copy.
    reference: dict[str, Any] = {}
    if mode == ORACLE:
        try:
            with isolated_rng(None, device):
                y_ref, g_ref = oracle(op, isolate(values))
            synchronize(device)
            outs, arity = named_outputs(op, y_ref)
            if arity:
                raise ValueError(arity)
            reference.update(outs)
            reference.update({C.PREFIX[C.INPUT_GRAD] + n: g for n, g in g_ref.items()})
        except Exception as exc:
            finding = missing("reference_failed", f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc(limit=6))
            if case.role == PROBE_CASE:
                report.skipped = "invalid_reference"
            report.findings.append(finding)
            return report
        invalid = [name for name, t in reference.items()
                   if C.is_float_tensor(t) and not bool(torch.isfinite(t).all())]
        if invalid and case.role == PROBE_CASE:
            report.skipped = "invalid_reference"
            report.findings.append(missing("invalid_reference",
                                           "the oracle is non-finite on this probe input",
                                           results=invalid))
            return report

    # 3. the candidate, on another copy; prepare on a third.
    values_c = isolate(values)
    allowed = tuple(op.backward_may_overwrite)
    before = snapshot(values_c, metadata_only=allowed)
    candidate: dict[str, Any] = {}
    context: dict[str, Any] = {"case": case, "values": values_c}
    try:
        if pair.prepare is not None:
            pair.prepare(isolate(values))
        output, saved = pair.forward(values_c)
        outs, arity = named_outputs(op, output)
        if arity:
            report.findings.append(failure("output_structure", arity))
        # Outputs are judged as the forward produced them: a backward may reuse
        # a dead output buffer, as it may in training once consumers have read it.
        candidate.update(isolate(outs))
        grads = pair.backward(upstream_grad_values(op, values_c), saved, values_c)
        named, arity = named_gradients(op, grads)
        if arity:
            report.findings.append(failure("gradient_arity", arity))
        candidate.update(named)
        synchronize(device)
        context.update(saved=saved, output=output, gradients=grads)
    except Exception as exc:
        report.findings.append(failure("candidate_runtime_error", f"{type(exc).__name__}: {exc}",
                                       traceback=traceback.format_exc(limit=8)))
        return report

    changed = mutations(values_c, before)
    report.evidence["input_mutation"] = {"changed": changed, "allowed_overwrite": list(allowed),
                                         "policy": input_mutation}
    if changed and input_mutation == REJECT:
        report.findings.append(failure("input_mutation",
                                       f"the pair changed inputs {changed} (only the contents of "
                                       f"{list(allowed)} may be overwritten)", inputs=changed))

    # 4. compare, by name.
    if mode == ORACLE:
        report.results.extend(compare_results(op, case.workload, contract, candidate, reference,
                                              envelope=envelope))
    else:
        report.results.extend(_smoke_results(contract, candidate))

    for name, observer in observers.items():
        try:
            report.evidence[name] = observer(context)
        except Exception as exc:  # an observer never changes the verdict
            report.evidence[name] = {"observer_error": f"{type(exc).__name__}: {exc}"}
    return report


def compare_results(op: OpDecl, workload: Workload, contract: C.ResultContract,
                    candidate: Mapping[str, Any], reference: Mapping[str, Any],
                    envelope: Mapping[str, float] | None = None) -> list[ResultCheck]:
    """One :class:`ResultCheck` per contract entry, in contract order."""
    pairs, presence = C.match_results(contract, candidate, reference)
    by_name = {spec.name: ResultCheck(spec.name, spec.kind) for spec in contract.specs}
    extra: list[ResultCheck] = []
    for finding in presence:
        target = by_name.get(finding.subject or "")
        if target is None:
            check = ResultCheck(finding.subject or "?", "unexpected")
            check.findings.append(finding)
            extra.append(check)
        else:
            target.findings.append(finding)
    for pair in pairs:
        check = by_name[pair.spec.name]
        structure = C.check_structure(pair)
        check.contract = structure.meta
        check.findings.extend(structure.findings)
        if structure.mode == "exact":
            check.compared = True
        if structure.mode != "numeric":
            continue
        try:
            atol, rtol = result_tolerances(op, workload, declared_name(pair.spec.name), pair.reference)
            floor = float((envelope or {}).get(pair.spec.name, 0.0))
            err = elementwise_error(pair.candidate, pair.reference, atol=atol, rtol=rtol,
                                    abs_floor=floor)
        except Exception as exc:  # no tolerance for this dtype, or an invalid one
            check.findings.append(missing("tolerance_unavailable", f"{type(exc).__name__}: {exc}",
                                          subject=pair.spec.name))
            continue
        check.compared = True
        check.limits = {"atol": atol, "rtol": rtol, "envelope": floor}
        check.metrics = err.to_dict()
        unbounded = [k for k, v in check.metrics.items()
                     if isinstance(v, float) and not math.isfinite(v)]
        for key in unbounded:
            check.metrics[key] = None
        if unbounded:
            check.metrics["unbounded"] = unbounded
        if not err.ok:
            check.findings.append(failure(
                "tolerance_exceeded",
                f"{err.violations}/{err.elements} elements outside max(atol={atol:g} + rtol={rtol:g}*|ref|, "
                f"{floor:.3g}) (max abs err {err.max_abs_err:.3g}, headroom {err.headroom:.3g})",
                violations=err.violations,
                subject=pair.spec.name))
    return [by_name[s.name] for s in contract.specs] + extra


def _smoke_results(contract: C.ResultContract, candidate: Mapping[str, Any]) -> list[ResultCheck]:
    out = []
    for spec in contract.specs:
        check = ResultCheck(spec.name, spec.kind)
        value = candidate.get(spec.name)
        if not torch.is_tensor(value):
            check.findings.append(failure("missing_result", "no tensor produced", subject=spec.name))
        else:
            check.contract = {"candidate": C.tensor_meta(value)}
            if spec.shape is not None and tuple(value.shape) != spec.shape:
                check.findings.append(failure("shape_mismatch",
                                              f"shape {tuple(value.shape)} != declared {spec.shape}",
                                              subject=spec.name))
            if spec.dtype is not None and C.dtype_name(value.dtype) != spec.dtype:
                check.findings.append(failure("dtype_mismatch",
                                              f"dtype {value.dtype} != declared {spec.dtype}",
                                              subject=spec.name))
            if C.is_float_tensor(value) and not bool(torch.isfinite(value).all()):
                check.findings.append(failure("non_finite_candidate", "non-finite values",
                                              subject=spec.name))
        out.append(check)
    return out


# ── naming and input contract ────────────────────────────────────────────────


def named_outputs(op: OpDecl, output: Any) -> tuple[dict[str, Any], str | None]:
    """Declared outputs by name, or a reason the structure does not match."""
    names = [C.PREFIX[C.OUTPUT] + n for n in op.output_names]
    if op.is_multi_output:
        if torch.is_tensor(output) or not isinstance(output, (tuple, list)):
            return {}, (f"declares {len(names)} outputs {list(op.output_names)}; forward returned "
                        f"a single {type(output).__name__}")
        if len(output) != len(names):
            return {}, (f"forward returned {len(output)} outputs, the contract requires "
                        f"{len(names)}: {list(op.output_names)}")
        return dict(zip(names, output)), None
    if not torch.is_tensor(output):
        return {}, f"declares one output {op.output_names[0]!r}; forward returned {type(output).__name__}"
    return {names[0]: output}, None


def named_gradients(op: OpDecl, grads: Any) -> tuple[dict[str, Any], str | None]:
    names = [C.PREFIX[C.INPUT_GRAD] + n for n in op.grad_names()]
    grads = (grads,) if torch.is_tensor(grads) or grads is None else tuple(grads)
    if len(grads) != len(names):
        return {}, (f"backward returned {len(grads)} gradients, the contract requires "
                    f"{len(names)}: {list(op.grad_names())}")
    return dict(zip(names, grads)), None


def case_input_problems(op: OpDecl, workload: Workload, values: Mapping[str, Any],
                        device: str) -> list[str]:
    """Supplied inputs must match the declaration; a mismatch is a bad case, not a bad candidate."""
    problems = []
    want = torch.device(device)
    entries = [(arg.name, arg) for arg in op.args] + [(o.grad_name, o) for o in op.outputs]
    for name, arg in entries:
        shape = getattr(arg, "shape", None)
        is_tensor = isinstance(arg, Active) or getattr(arg, "is_tensor", False) or name in op.upstream_grad_names
        if not is_tensor:
            continue
        value = values.get(name)
        if not torch.is_tensor(value):
            problems.append(f"input {name!r}: expected a tensor, got {type(value).__name__}")
            continue
        if shape not in (None, "unspecified"):
            expected = tuple(bind_shape(shape, workload.dims))
            if tuple(value.shape) != expected:
                problems.append(f"input {name!r}: shape {tuple(value.shape)} != declared {expected}")
        dtype = resolve_dtype(arg_dtype_name(arg, workload))
        if value.dtype != dtype:
            problems.append(f"input {name!r}: dtype {value.dtype} != declared {dtype}")
        if value.device.type != want.type or (want.index is not None and value.device.index != want.index):
            problems.append(f"input {name!r}: device {value.device} != {device}")
    for name in op.upstream_grad_names:
        value = values.get(name)
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            problems.append(f"upstream gradient {name!r} is non-finite")
    return problems
