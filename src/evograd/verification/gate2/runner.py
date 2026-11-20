"""Gate 2: one autograd forward/backward of a deployed scope.

**Execution, every scope.** :func:`execute` builds the scope's module with a
deployment installed, makes the declared differentiable inputs fresh leaves,
runs ``module(*args, **kwargs)``, takes the declared outputs, and calls
``torch.autograd.backward`` from the declared cotangents (for a model: the
loss, seeded with one). No optimizer exists and none steps. The sample's seed
governs all randomness inside, so every execution of a sample sees the same
inputs, weights, cotangents and random draws.

**Judgement** is the scopes' own: :mod:`.operator` holds each call to the float32
truth, :mod:`.model` holds the whole model to native eager. Both share the result
contract (:func:`integration_contract`) and the structural check
(:func:`structure_checks`).
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import (
    Deployment,
    Invocation,
    Sample,
    SiteHooks,
    replace_at,
    resolve,
)
from evograd.verification.common.reports import Finding, ResultCheck
from evograd.verification.common.tensors import isolate, isolated_rng, synchronize

GATE = "gate2"
#: Gate 2's scopes: :func:`evograd.verification.gate2.operator.verify_operator` and
#: :func:`evograd.verification.gate2.model.verify_model`.
SCOPES = ("operator", "model")


class OutputStructureError(ValueError):
    """The scope's result does not have the declared output structure."""


class ScopeAdapter(Protocol):
    """What a scope supplies. It builds and runs; it never decides a verdict."""

    scope: str
    device: str
    #: Outputs that only seed the backward (a cotangent names them) and are not
    #: results: never recorded, never judged.
    backward_only: tuple[str, ...]

    def identity(self) -> Mapping[str, Any]: ...
    def numerics(self) -> Mapping[str, Any]: ...
    def sites(self) -> tuple[str, ...]: ...
    def native_site(self, site: str) -> Callable: ...
    def build(self, deployment: Deployment, sample: Sample, hooks: SiteHooks) -> torch.nn.Module: ...
    def expected_invocations(self, deployment: Deployment) -> Mapping[str, int]: ...
    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation: ...
    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> Mapping[str, Any]: ...
    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]: ...
    def cotangents(self, outputs: Mapping[str, Any], invocation: Invocation,
                   sample: Sample) -> Mapping[str, torch.Tensor]: ...
    def numerics_context(self): ...
    def structure(self, result: Any) -> Any: ...
    def reference_identity(self) -> Mapping[str, Any]: ...


# ── one execution ────────────────────────────────────────────────────────────


@dataclass
class Execution:
    deployment: Deployment
    sample: Sample
    results: dict[str, Any] = field(default_factory=dict)
    specs: tuple[C.ResultSpec, ...] = ()
    structure: Any = None
    parameter_groups: list[tuple[str, ...]] = field(default_factory=list)
    trainable: dict[str, bool] = field(default_factory=dict)
    #: Calls per replaced site, and the calls the scope declares.
    counts: dict[str, int] = field(default_factory=dict)
    expected: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    error_stage: str | None = None
    seconds: float = 0.0


def execute(adapter: ScopeAdapter, deployment: Deployment, sample: Sample) -> Execution:
    """Build, run forward + backward once, collect every named result.

    The sample's seed governs all randomness inside, and the caller's RNG is
    restored afterwards. Nothing is judged here.
    """
    ex = Execution(deployment, sample)
    hooks = SiteHooks()
    started = time.perf_counter()
    stage = "build"
    try:
        with isolated_rng(sample.seed, adapter.device):
            module = adapter.build(deployment, sample, hooks)
            # Building drops the previous run's model (a workload keeps its last
            # build), and a tier-3 workload's patched layers hold themselves
            # (``forward`` bound to the instance): only the cyclic collector frees
            # them, and device memory does not trigger it (Qwen3.5-4B: ~7.5 GiB per
            # run, 2026-11-02). Free them before this run allocates its own. A scope
            # whose builds hold no cycles skips it (``cyclic_builds``): with a model
            # loaded the collection takes ~0.1 s, more than half of operator
            # Gate 2's time per call (2026-11-20).
            if getattr(adapter, "cyclic_builds", True):
                gc.collect()
            ex.expected = dict(adapter.expected_invocations(deployment))
            ex.parameter_groups = C.parameter_groups(module)
            params = dict(module.named_parameters())
            ex.trainable = {g[0]: params[g[0]].requires_grad for g in ex.parameter_groups}
            stage = "prepare"
            invocation = adapter.invocation(sample, module)
            roots = isolate({"args": tuple(invocation.args), "kwargs": dict(invocation.kwargs)})
            for path in invocation.differentiable_inputs.values():
                leaf = resolve(roots, path)
                if not (torch.is_tensor(leaf) and leaf.is_floating_point()):
                    raise ValueError(f"differentiable input {path} is not a floating tensor")
                roots = replace_at(roots, path, leaf.requires_grad_(True))
            leaves = {name: resolve(roots, path)
                      for name, path in invocation.differentiable_inputs.items()}
            module.zero_grad(set_to_none=True)
            hooks.reset()
            stage = "forward"
            with adapter.numerics_context():
                result = module(*roots["args"], **roots["kwargs"])
                stage = "outputs"
                outputs = dict(adapter.outputs(result, invocation, sample))
            ex.specs = tuple(adapter.output_specs(invocation))
            ex.structure = adapter.structure(result)
            # Judged as the forward produced them: the backward may legitimately
            # reuse an output buffer nothing reads any more.
            produced_outputs = isolate(_detach_tree(
                {n: v for n, v in outputs.items() if n not in adapter.backward_only}))
            stage = "backward"
            seeds = isolate(dict(adapter.cotangents(outputs, invocation, sample)))
            unknown = sorted(set(seeds) - set(outputs))
            if unknown:
                raise ValueError(f"cotangents name outputs that do not exist: {unknown}")
            if seeds:
                torch.autograd.backward([outputs[n] for n in seeds], [seeds[n] for n in seeds])
            synchronize(adapter.device)
            stage = "collect"
            ex.counts = dict(hooks.counts)
            by_spec = {s.name: s for s in ex.specs}
            for name, value in produced_outputs.items():
                key = (C.PREFIX[C.METADATA] if (C.PREFIX[C.METADATA] + name) in by_spec
                       else C.PREFIX[C.OUTPUT]) + name
                ex.results[key] = value
            for name, leaf in leaves.items():
                ex.results[C.PREFIX[C.INPUT_GRAD] + name] = (None if leaf.grad is None
                                                             else leaf.grad.detach())
            for group in ex.parameter_groups:
                p = params[group[0]]
                ex.results[C.PREFIX[C.PARAM_GRAD] + group[0]] = (None if p.grad is None
                                                                else p.grad.detach())
    except Exception:
        ex.error = traceback.format_exc(limit=10)
        ex.error_stage = stage
    ex.seconds = time.perf_counter() - started
    return ex


def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _detach_tree(v) for k, v in value.items()}
    return value


# ── what is compared ─────────────────────────────────────────────────────────


def integration_contract(reference: Execution, invocation_names: Iterable[str] | None = None
                         ) -> C.ResultContract:
    """The results Gate 2 compares, from the declaration and the native run.

    Declared outputs (the adapter's specs) and declared differentiable inputs
    are required. Parameters come from the native module: a trainable
    parameter whose native gradient exists is required, one whose native
    gradient is ``None`` (it takes no part) is ``absent``. Tied parameters are
    one result with aliases. Shapes and dtypes are the native ones.
    """
    specs = []
    for spec in reference.specs:
        value = reference.results.get(spec.name)
        specs.append(_bound(spec, value))
    names = (sorted(k[len(C.PREFIX[C.INPUT_GRAD]):] for k in reference.results
                    if k.startswith(C.PREFIX[C.INPUT_GRAD]))
             if invocation_names is None else list(invocation_names))
    for name in names:
        key = C.PREFIX[C.INPUT_GRAD] + name
        specs.append(_bound(C.ResultSpec(key, C.INPUT_GRAD), reference.results.get(key)))
    for group in reference.parameter_groups:
        if not reference.trainable.get(group[0], False):
            continue
        key = C.PREFIX[C.PARAM_GRAD] + group[0]
        grad = reference.results.get(key)
        presence = C.REQUIRED if grad is not None else C.ABSENT
        specs.append(_bound(C.ResultSpec(key, C.PARAM_GRAD, presence=presence,
                                         aliases=tuple(group[1:])), grad))
    return C.ResultContract(tuple(specs))


def _bound(spec: C.ResultSpec, value: Any) -> C.ResultSpec:
    if not torch.is_tensor(value):
        return spec
    exact = spec.exact or not C.is_float_tensor(value)
    return C.ResultSpec(spec.name, spec.kind, spec.presence, exact, spec.aliases,
                        tuple(value.shape), C.dtype_name(value.dtype))


def structure_checks(contract: C.ResultContract, candidate: Mapping[str, Any],
                     reference: Mapping[str, Any]
                     ) -> tuple[dict[str, ResultCheck], list[str], list[Finding]]:
    """Every contract result's presence, type, shape, dtype, device and finiteness
    against the reference's (exact equality for non-floating results).

    Returns the result checks by name, the results ready for a numerical
    comparison (floating, present, structurally sound) and the findings that
    belong to no single result.
    """
    checks = {s.name: ResultCheck(s.name, s.kind) for s in contract.specs
              if s.presence != C.ABSENT}
    loose: list[Finding] = []
    pairs, presence = C.match_results(contract, candidate, reference)
    for finding in presence:
        target = checks.get(finding.subject or "")
        (target.findings if target is not None else loose).append(finding)
    numeric = []
    for pair in pairs:
        check, structure = checks[pair.spec.name], C.check_structure(pair)
        check.contract = structure.meta
        check.findings.extend(structure.findings)
        if structure.mode == "exact":
            check.compared = True
        elif structure.mode == "numeric" and not structure.findings:
            numeric.append(pair.spec.name)
    return checks, numeric, loose
