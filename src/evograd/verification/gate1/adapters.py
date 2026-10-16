"""Gate 1 adapters: what goes into the core, and what its report turns into.

Inputs to :func:`evograd.verification.gate1.runner.verify_pair`:

* :func:`pair_from_module` -- a seed/candidate module (``forward_with_saved``
  and ``backward_from_saved``, generic or op-prefixed names);
* :func:`pair_from_provider` -- anything with the ``PairProvider`` surface
  (``forward(values)``, ``backward(dout, saved, values)``, optional
  ``prepare``). Duck-typed, so this package does not import the provider
  module;
* case builders -- the declaration's inputs at another generator seed,
  upstream-gradient patterns, and caller-named stress probes. Nothing here
  knows an operator family: which activations a stress pattern may touch is
  the caller's statement.

Outputs: :func:`evolution_correctness` (the per-result rates the evolution
evaluator reports) and :func:`saved_state_observer` (saved-state bytes and
retained inputs, as evidence).
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

import torch

from evograd.opdecl.activity import OpDecl, Workload
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import apply_recorded_strides, case_seed, make_case_inputs
from evograd.verification.common import contracts as C
from evograd.verification.common.reports import VerificationReport
from evograd.verification.common.tensors import isolate, isolated_rng, tensors_in
from evograd.verification.gate1.runner import (
    PROBE_CASE,
    REQUIRED_CASE,
    Gate1Case,
    PairUnderTest,
    case_id_for,
    declared_name,
)


# ── candidates ───────────────────────────────────────────────────────────────


def _source_hash(*functions: Callable) -> str:
    digest = hashlib.sha256()
    for fn in functions:
        try:
            digest.update(inspect.getsource(fn).encode())
        except (OSError, TypeError):
            digest.update(repr(fn).encode())
    return digest.hexdigest()


def pair_from_module(op: OpDecl, module: Any, *, name: str = "candidate") -> PairUnderTest:
    """A seed module, called exactly as the evolution evaluator calls it."""
    fwd, bwd = lookup_pair(op, module)

    def forward(values):
        return fwd(*[values.get(a.name, getattr(a, "default", None)) for a in op.args])

    def backward(output_grads, saved, values):
        return bwd(output_grads, saved, **backward_inactive_kwargs(op, bwd, values))

    return PairUnderTest(name=name, forward=forward, backward=backward,
                         identity={"adapter_kind": "direct_pair_module",
                                   "module": getattr(module, "__name__", repr(module)),
                                   "source_hash": _source_hash(fwd, bwd)})


def pair_from_provider(provider: Any) -> PairUnderTest:
    """A ``PairProvider`` (or look-alike), with its ``prepare`` and its own call path.

    Nothing is re-routed: an engine-mediated provider stays engine-mediated and
    a direct provider stays direct.
    """
    return PairUnderTest(
        name=str(getattr(provider, "name", "provider")),
        forward=provider.forward, backward=provider.backward,
        prepare=getattr(provider, "prepare", None),
        identity={"adapter_kind": getattr(provider, "adapter_kind", None),
                  "source_hash": getattr(provider, "source_hash", None),
                  "uses_autograd_engine": bool(getattr(provider, "uses_autograd_engine", False))})


def pair_from_baseline(op: OpDecl, name: str, hook: Any | None = None) -> PairUnderTest:
    """A declared performance baseline (``op.performance_baselines[name]``).

    A hook with ``pair_factory`` (plus ``forward_args``/``backward_extras``) is
    called as a direct pair. A hook with only ``reference_run(values) ->
    (output, gradients)`` runs whole; its gradients are taken for the case's
    upstream gradient, which is what such a hook computes. Verification
    caching stays with the caller.
    """
    hook = hook if hook is not None else op.performance_baselines[name]
    factory = getattr(hook, "pair_factory", None)
    if factory is not None:
        fwd, bwd = factory()
        forward_args = tuple(getattr(hook, "forward_args", ()))
        extras = tuple(getattr(hook, "backward_extras", ()))
        return PairUnderTest(
            name=name, forward=lambda values: fwd(*(values[n] for n in forward_args)),
            backward=lambda dout, saved, values: bwd(dout, saved, *(values[n] for n in extras)),
            identity={"adapter_kind": f"declared_{name}_pair", "source_hash": _source_hash(fwd, bwd)})
    run = getattr(hook, "reference_run", None)
    if run is None:
        raise ValueError(f"{op.name}: baseline {name!r} exposes neither pair_factory nor reference_run")
    return PairUnderTest(
        name=name, forward=lambda values: run(values),
        backward=lambda dout, saved, values: saved,
        identity={"adapter_kind": f"declared_{name}_reference_run",
                  "source_hash": _source_hash(run),
                  "note": "forward and backward run together; gradients use the case upstream"})


def as_pair(op: OpDecl, candidate: Any) -> PairUnderTest:
    if isinstance(candidate, PairUnderTest):
        return candidate
    if all(callable(getattr(candidate, a, None)) for a in ("forward", "backward")) and \
            hasattr(candidate, "adapter_kind"):
        return pair_from_provider(candidate)
    return pair_from_module(op, candidate)


# ── cases ────────────────────────────────────────────────────────────────────


class _OffsetSeedTorch:
    """``torch`` with every ``manual_seed`` shifted by a fixed offset.

    Input recipes receive ``torch`` as a parameter, so a recipe can be re-run
    at another seed without patching anything global.
    """

    def __init__(self, offset: int):
        self.offset = int(offset)

    def __getattr__(self, name: str):
        return getattr(torch, name)

    def manual_seed(self, seed: int):
        return torch.manual_seed((int(seed) + self.offset) % (2**63 - 1))


def declared_values(op: OpDecl, workload: Workload, *, device: str,
                    seed_offset: int = 0) -> dict[str, Any]:
    """The declaration's inputs for ``workload``, optionally at a shifted seed.

    The recipe is the declaration's own: a ``make_inputs`` hook is re-run with
    its ``manual_seed`` shifted; the default generator redraws its Active
    inputs and upstream gradients (``0.5 * randn``) after the shifted seed.
    Recorded strides are re-applied either way. The caller's RNG is untouched.
    """
    with isolated_rng(None, device):
        if not seed_offset:
            return make_case_inputs(op, workload, device=device)
        if op.make_inputs is not None:
            proxy = _OffsetSeedTorch(seed_offset)
            proxy.manual_seed(case_seed(workload))
            return apply_recorded_strides(op, workload,
                                          dict(op.make_inputs(proxy, op, workload, device)))
        values = make_case_inputs(op, workload, device=device)
        torch.manual_seed((case_seed(workload) + int(seed_offset)) % (2**63 - 1))
        for name in [a.name for a in op.active_args()] + list(op.upstream_grad_names):
            value = values[name]
            values[name] = torch.randn(tuple(value.shape), device=value.device,
                                       dtype=value.dtype) * 0.5
        return apply_recorded_strides(op, workload, values)


def generated_case(op: OpDecl, workload: Workload, *, device: str, seed_offset: int = 0,
                   role: str = REQUIRED_CASE) -> Gate1Case:
    """A declared case at ``seed_offset``; the provenance lists what the shift changed."""
    values = declared_values(op, workload, device=device, seed_offset=seed_offset)
    provenance: dict[str, Any] = {"generator": "declaration", "seed_offset": int(seed_offset),
                                  "case_seed": case_seed(workload)}
    if seed_offset:
        base = declared_values(op, workload, device=device)
        provenance["changed_tensors"] = [
            n for n, v in values.items()
            if torch.is_tensor(v) and torch.is_tensor(base.get(n)) and not torch.equal(v, base[n])]
        if not provenance["changed_tensors"]:
            # The recipe ignores the seed it was given: this is the default case again.
            provenance["note"] = "the input recipe did not respond to the seed shift"
    variant = "declared" if not seed_offset else f"seed+{seed_offset}"
    return Gate1Case(case_id_for(op, workload, variant), workload, values, role, provenance)


def upstream_variants(op: OpDecl, case: Gate1Case, *, device: str,
                      patterns: Iterable[str] = ("ones", "alternating")) -> list[Gate1Case]:
    """The same primals with structured upstream gradients."""
    base = dict(case.values) if case.values is not None else declared_values(
        op, case.workload, device=device)
    out = []
    for pattern in patterns:
        values = isolate(base)
        for name in op.upstream_grad_names:
            values[name] = _pattern(values[name], pattern)
        out.append(Gate1Case(f"{case.case_id}+upstream_{pattern}", case.workload, values,
                             case.role, {**case.provenance, "upstream_pattern": pattern}))
    return out


def stress_probes(op: OpDecl, case: Gate1Case, names: Iterable[str], *, device: str,
                  patterns: Iterable[str] = ("large_constant", "cancellation"),
                  magnitude: float = 1000.0) -> list[Gate1Case]:
    """Probe cases that set the named floating inputs to stress patterns.

    ``names`` is the caller's statement of which activations the operator's
    domain allows to take these values; nothing is inferred from a family. A
    probe whose oracle is invalid is recorded as skipped, never as passed.
    """
    names = tuple(names)
    base = dict(case.values) if case.values is not None else declared_values(
        op, case.workload, device=device)
    unknown = [n for n in names if not (torch.is_tensor(base.get(n)) and base[n].is_floating_point())]
    if unknown:
        raise ValueError(f"{op.name}: stress inputs {unknown} are not floating tensor inputs")
    out = []
    for pattern in patterns:
        values = isolate(base)
        for name in names:
            values[name] = _pattern(values[name], pattern, magnitude)
        out.append(Gate1Case(f"{case.case_id}+stress_{pattern}", case.workload, values,
                             PROBE_CASE, {**case.provenance, "stress_pattern": pattern,
                                          "stress_inputs": list(names), "magnitude": magnitude}))
    return out


def _pattern(value: torch.Tensor, pattern: str, magnitude: float = 1.0) -> torch.Tensor:
    if pattern == "ones":
        return torch.ones_like(value)
    if pattern == "large_constant":
        return torch.full_like(value, magnitude)
    signs = (torch.arange(value.numel(), device=value.device) % 2).reshape(value.shape)
    if pattern == "alternating":
        return (2 * signs - 1).to(value.dtype)
    if pattern == "cancellation":
        return ((1 - 2 * signs) * magnitude).to(value.dtype)
    raise ValueError(f"unknown pattern {pattern!r}")


# ── report adapters ──────────────────────────────────────────────────────────


def evolution_correctness(report: VerificationReport, op: OpDecl) -> dict[str, Any]:
    """The evolution evaluator's correctness summary, derived from one report.

    Rates are over the report's decisive cases: ``partial_correctness`` (cases
    that passed), ``forward_correctness`` (cases whose every output passed)
    and ``<grad>_correctness`` per required gradient. ``status``/``complete``
    are carried so an incomplete report cannot read as a 0% candidate.
    """
    grad_names = op.grad_names()
    cases = report.decisive_cases()
    total = len(cases)
    passed = forward = 0
    per_grad = {g: 0 for g in grad_names}
    rows = []
    for case in cases:
        row: dict[str, Any] = {"case_id": case.case_id, **{k: case.evidence.get(k)
                                                           for k in ("dims", "dtype")},
                               "status": case.status, "correct": case.status == "passed"}
        outputs_ok = bool(case.results) and not case.findings
        forward_abs = 0.0
        for result in case.results:
            name = declared_name(result.name) if ":" in result.name else result.name
            ok = result.status == "passed" and not case.findings
            if result.kind == C.OUTPUT:
                outputs_ok = outputs_ok and ok
                forward_abs = max(forward_abs, float(result.metrics.get("max_abs_err") or 0.0))
                if op.is_multi_output:
                    row[f"{name}_correct"] = ok
            else:
                row[f"{name}_correct"] = ok
                row[f"{name}_max_abs_error"] = result.metrics.get("max_abs_err")
                row[f"{name}_max_rel_error"] = result.metrics.get("max_rel_err")
                per_grad[name] = per_grad.get(name, 0) + int(ok)
        row["forward_correct"] = outputs_ok
        row["forward_max_abs_error"] = forward_abs
        messages = [f.message for f in case.all_findings()]
        if messages:
            row["error_message"] = "; ".join(messages[:4])
        passed += int(row["correct"])
        forward += int(outputs_ok)
        rows.append(row)
    denominator = max(1, total)
    return {"status": report.status, "complete": report.complete, "passed": passed,
            "total": total, "partial_correctness": passed / denominator,
            "forward_correctness": forward / denominator,
            **{f"{g}_correctness": per_grad[g] / denominator for g in grad_names},
            "reports": rows}


def saved_state_observer(context: Mapping[str, Any]) -> dict[str, Any]:
    """Saved-state size and which saved tensors share storage with an input."""
    saved = context.get("saved")
    items = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
    inputs = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
              for _, t in tensors_in(context.get("values", {}))}
    logical = 0
    unique: dict[int, int] = {}
    retained: dict[int, int] = {}
    described = []
    for item in items:
        if torch.is_tensor(item):
            logical += item.numel() * item.element_size()
            storage = item.untyped_storage()
            unique[storage.data_ptr()] = storage.nbytes()
            if storage.data_ptr() in inputs:
                retained[storage.data_ptr()] = storage.nbytes()
            described.append({"kind": "tensor", **C.tensor_meta(item)})
        else:
            described.append({"kind": "value", "type": type(item).__name__,
                              "repr": repr(item)[:120]})
    return {"logical_saved_bytes": logical, "unique_saved_storage_bytes": sum(unique.values()),
            "retained_input_bytes": sum(retained.values()), "saved": described}


def with_role(case: Gate1Case, role: str) -> Gate1Case:
    return replace(case, role=role)
