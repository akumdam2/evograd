"""Gate 2: one autograd forward/backward of a deployed scope, against native eager.

**Execution, every scope.** :func:`execute` builds the scope's module with a
deployment installed, makes the declared differentiable inputs fresh leaves,
runs ``module(*args, **kwargs)``, takes the declared outputs, and calls
``torch.autograd.backward`` from the declared cotangents (for a model: the
loss, seeded with one). No optimizer exists and none steps; the parameters
must be bit-identical afterwards.

**Reference.** The same scope with nothing replaced -- the native eager
operator, or the unpatched model -- on the same inputs, weights, RNG seed,
numerics settings and cotangents.

**Scopes and sample protocols.** Two scopes, each with one sample protocol,
recorded in the policy and checked on use (never inferred from which samples
happen to overlap):

* ``operator`` -- ``disjoint_holdout``: thresholds pooled over calibration
  samples, judged on different (generated) samples;
* ``model`` -- ``matched_inputs``: thresholds per sample and per tensor, judged
  on fresh executions of the *same* samples, whose content (inputs, labels and
  other context, cotangents, initial weights, module mode, seed) must match the
  calibrated sample's fingerprint.


**Comparison, every result.** :func:`judge` settles structure, presence and
finiteness, then computes ``E2`` and ``Emax`` in float64 and holds them to the
frozen per-tensor thresholds ``T[z, E2]`` and ``T[z, Emax]``. Every required
result of every validation sample must pass. There is no KL, no global
gradient norm, no elementwise ``atol/rtol``, no envelope.

**Evidence a numeric match cannot fake.** The return structure, the parameter
list and its tying, the initial weights, the per-site call counts, input
mutation, and -- when the policy asks for it -- every live call boundary of a
replaced site, judged by the same comparator against matching thresholds.
"""

from __future__ import annotations

import contextlib
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import (
    CANDIDATE,
    NATIVE,
    BoundaryCall,
    Deployment,
    capture_grad,
    content_identity,
    Invocation,
    Sample,
    SiteHooks,
    describe_tree,
    replace_at,
    resolve,
)
from evograd.verification.common.reports import (
    CaseReport,
    Finding,
    ResultCheck,
    VerificationReport,
    failure,
    missing,
)
from evograd.verification.common.tensors import (
    isolate,
    isolated_rng,
    module_state_digest,
    mutations,
    relative_error,
    snapshot,
    synchronize,
    tensor_digest,
    tensors_in,
)

GATE = "gate2"
PROTOCOL = "evograd-verification-gate2/1"
METRICS = ("E2", "Emax")
SCOPES = ("operator", "model")
#: The parts of a sample's content identity, in a fixed order.
IDENTITY_COMPONENTS = ("seed", "inputs", "context", "cotangents", "initial_state", "module_mode",
                       "differentiable_inputs", "mutable_inputs")
BOUNDARY = "boundary:"


class OutputStructureError(ValueError):
    """The scope's result does not have the declared output structure."""


class ScopeAdapter(Protocol):
    """What a scope supplies. It builds and runs; it never decides a verdict."""

    scope: str
    device: str

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
    initial_state: dict[str, str] = field(default_factory=dict)
    state_changed: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    expected: dict[str, int] = field(default_factory=dict)
    mutated: list[str] = field(default_factory=list)
    boundaries: list[BoundaryCall] = field(default_factory=list)
    #: Content hash of the inputs and cotangents this run was given: what
    #: makes two samples the same data, whatever their ids and seeds.
    input_digest: str | None = None
    #: The full content identity of the sample as executed (see
    #: :data:`IDENTITY_COMPONENTS`): every input taken *before* execution,
    #: including inputs a contract lets the computation overwrite, plus the
    #: invocation context (labels, masks), cotangents, initial state, module
    #: mode and seed. ``fingerprint`` hashes the components.
    sample_identity: dict[str, Any] = field(default_factory=dict)
    #: Content hash of every collected result (an execution's answer).
    results_digest: str | None = None
    #: Unique per execution: a fresh run is never another run's copy.
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error: str | None = None
    error_stage: str | None = None
    seconds: float = 0.0

    def evidence(self) -> dict[str, Any]:
        return {"execution_id": self.execution_id,
                "deployment": self.deployment.describe(), "sample": self.sample.to_dict(),
                "site_calls": dict(self.counts), "expected_site_calls": dict(self.expected),
                "mutated_inputs": list(self.mutated), "state_changed": list(self.state_changed),
                "parameter_groups": [list(g) for g in self.parameter_groups if len(g) > 1],
                "boundary_calls": len(self.boundaries), "input_digest": self.input_digest,
                "sample_identity": dict(self.sample_identity),
                "results_digest": self.results_digest,
                "initial_state_digest": (C.stable_hash(self.initial_state)
                                         if self.initial_state else None),
                "error_stage": self.error_stage,
                "seconds": self.seconds}


def execute(adapter: ScopeAdapter, deployment: Deployment, sample: Sample, *,
            record_boundaries: bool = False) -> Execution:
    """Build, run forward + backward once, collect every named result.

    The sample's seed governs all randomness inside, and the caller's RNG is
    restored afterwards. Nothing is judged here.
    """
    ex = Execution(deployment, sample)
    hooks = SiteHooks(record=record_boundaries)
    started = time.perf_counter()
    stage = "build"
    try:
        with isolated_rng(sample.seed, adapter.device):
            module = adapter.build(deployment, sample, hooks)
            ex.expected = dict(adapter.expected_invocations(deployment))
            ex.parameter_groups = C.parameter_groups(module)
            params = dict(module.named_parameters())
            ex.trainable = {g[0]: params[g[0]].requires_grad for g in ex.parameter_groups}
            ex.initial_state = module_state_digest(module)
            stage = "prepare"
            invocation = adapter.invocation(sample, module)
            roots = isolate({"args": tuple(invocation.args), "kwargs": dict(invocation.kwargs)})
            # Content identity before anything runs: inputs the contract lets a
            # kernel overwrite are hashed here too, not only their layout.
            inputs_id, opaque = content_identity(roots)
            context_id, opaque_ctx = content_identity(dict(invocation.context), "context")
            for path in invocation.differentiable_inputs.values():
                leaf = resolve(roots, path)
                if not (torch.is_tensor(leaf) and leaf.is_floating_point()):
                    raise ValueError(f"differentiable input {path} is not a floating tensor")
                roots = replace_at(roots, path, leaf.requires_grad_(True))
            leaves = {name: resolve(roots, path)
                      for name, path in invocation.differentiable_inputs.items()}
            mutable = tuple(invocation.mutable_inputs)
            before = snapshot(roots, metadata_only=tuple(p for p in mutable
                                                         if not p.startswith("cotangent:")))
            module.zero_grad(set_to_none=True)
            hooks.reset()
            stage = "forward"
            with adapter.numerics_context():
                result = module(*roots["args"], **roots["kwargs"])
                stage = "outputs"
                outputs = dict(adapter.outputs(result, invocation, sample))
            ex.specs = tuple(adapter.output_specs(invocation))
            ex.structure = describe_tree(result)
            # Judged as the forward produced them: the backward may legitimately
            # reuse an output buffer nothing reads any more.
            produced_outputs = isolate(_detach_tree(outputs))
            stage = "backward"
            seeds = isolate(dict(adapter.cotangents(outputs, invocation, sample)))
            seed_before = snapshot(seeds, metadata_only=tuple(
                p.split(":", 1)[1] for p in mutable if p.startswith("cotangent:")))
            ex.input_digest = C.stable_hash(
                [(path, tensor_digest(t)) for path, t in tensors_in(
                    {"inputs": before_values(before), "cotangents": seeds})])
            cotangents_id, opaque_cot = content_identity(seeds, "cotangent")
            ex.sample_identity = sample_identity(
                sample, inputs=inputs_id, context=context_id, cotangents=cotangents_id,
                initial_state=ex.initial_state, training=module.training,
                differentiable=invocation.differentiable_inputs, mutable=mutable,
                opaque=opaque + opaque_ctx + opaque_cot)
            unknown = sorted(set(seeds) - set(outputs))
            if unknown:
                raise ValueError(f"cotangents name outputs that do not exist: {unknown}")
            if seeds:
                torch.autograd.backward([outputs[n] for n in seeds], [seeds[n] for n in seeds])
            synchronize(adapter.device)
            stage = "collect"
            ex.counts = dict(hooks.counts)
            ex.boundaries = list(hooks.calls)
            ex.mutated = mutations(roots, before) + [f"cotangent:{p}" for p in mutations(seeds, seed_before)]
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
            after = module_state_digest(module)
            ex.state_changed = sorted(k for k in set(after) | set(ex.initial_state)
                                      if after.get(k) != ex.initial_state.get(k))
            ex.results_digest = results_digest(ex.results)
    except Exception:
        ex.error = traceback.format_exc(limit=10)
        ex.error_stage = stage
    ex.seconds = time.perf_counter() - started
    return ex


def sample_identity(sample: Sample, *, inputs: Any, context: Any, cotangents: Any,
                    initial_state: Mapping[str, str], training: bool,
                    differentiable: Mapping[str, str], mutable: Sequence[str],
                    opaque: Sequence[str]) -> dict[str, Any]:
    """The components of a sample's content identity, their details, and the
    fingerprint over the components. ``opaque`` lists inputs that could not be
    identified by content (a matched policy refuses to bind them)."""
    components = {
        "seed": int(sample.seed),
        "inputs": C.stable_hash(inputs),
        "context": C.stable_hash(context),
        "cotangents": C.stable_hash(cotangents),
        "initial_state": C.stable_hash(dict(initial_state)),
        "module_mode": "train" if training else "eval",
        "differentiable_inputs": dict(sorted(dict(differentiable).items())),
        "mutable_inputs": sorted(mutable),
    }
    return {"fingerprint": C.stable_hash([components[k] for k in IDENTITY_COMPONENTS]),
            "components": components, "opaque": list(opaque),
            "detail": {"inputs": inputs, "context": context, "cotangents": cotangents}}


def results_digest(results: Mapping[str, Any]) -> str:
    """Content hash of every collected result, by name."""
    return C.stable_hash({name: (tensor_digest(v) if torch.is_tensor(v) else describe_tree(v))
                          for name, v in sorted(results.items())})


def identity_differences(stored: Mapping[str, Any], live: Mapping[str, Any],
                         components: Sequence[str] = IDENTITY_COMPONENTS) -> list[str]:
    """Which identity components differ (all of them when either side is absent)."""
    a = (stored or {}).get("components") or {}
    b = (live or {}).get("components") or {}
    if not a or not b:
        return list(components)
    return [k for k in components if a.get(k) != b.get(k)]


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


def before_values(snap: Mapping[str, Any]) -> dict[str, Any]:
    """The input tensors a snapshot kept (metadata-only entries have none)."""
    return {path: s.value for path, s in snap.items() if s.value is not None}


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


# ── one comparison ───────────────────────────────────────────────────────────


def judge(spec: C.ResultSpec, candidate: Any, reference: Any,
          limit: Mapping[str, float] | None) -> ResultCheck:
    """The single Gate 2 numerical judgement, for any scope and any boundary.

    Structure first (type, shape, dtype, device, finiteness; exact equality
    for non-floating results), then ``E2``/``Emax`` against ``limit``. A
    missing limit leaves the result incomplete: no threshold, no pass.
    """
    check = ResultCheck(spec.name, spec.kind)
    structure = C.check_structure(C.Pairing(spec, candidate, reference))
    check.contract = structure.meta
    check.findings.extend(structure.findings)
    if structure.mode == "exact":
        check.compared = True
        check.limits = {"exact": True}
        return check
    if structure.mode != "numeric":
        return check
    try:
        err = relative_error(candidate, reference)
    except Exception as exc:  # a comparison that cannot be computed is not a pass
        check.findings.append(missing("comparison_failed", f"{type(exc).__name__}: {exc}",
                                      subject=spec.name))
        return check
    check.compared = True
    check.metrics = err.to_dict()
    if limit is None:
        check.findings.append(missing("no_threshold", "no frozen threshold for this result",
                                      subject=spec.name))
        return check
    check.limits = {m: limit.get(m) for m in METRICS}
    for metric in METRICS:
        bound = limit.get(metric)
        if not isinstance(bound, (int, float)) or isinstance(bound, bool) or not bound >= 0:
            check.findings.append(missing("invalid_threshold", f"threshold {metric}={bound!r}",
                                          subject=spec.name))
            continue
        value = getattr(err, metric)
        if not value <= bound:
            check.findings.append(failure(
                "threshold_exceeded",
                f"{metric} {value:.3g} > {bound:.3g} ({value / bound if bound else float('inf'):.3g}x)",
                subject=spec.name, metric=metric, value=value, threshold=bound))
    return check


def compare_executions(contract: C.ResultContract, reference: Execution, candidate: Execution,
                       thresholds: Mapping[str, Mapping[str, float]] | None
                       ) -> tuple[list[ResultCheck], list[Finding]]:
    """Every contract result through :func:`judge`, plus the structural evidence.

    Problems on the reference side are ``incomplete``; on the candidate side
    ``failed``. Both are always collected.
    """
    findings: list[Finding] = []
    side_findings(reference, findings, reference=True)
    side_findings(candidate, findings, reference=False)
    if candidate.structure != reference.structure:
        findings.append(failure("output_structure_mismatch",
                                "the scope's return structure differs from the native one",
                                candidate=candidate.structure, reference=reference.structure))
    if candidate.parameter_groups != reference.parameter_groups:
        findings.append(failure("parameter_structure_mismatch",
                                "parameter names, order or tying differ from the native module",
                                candidate=[list(g) for g in candidate.parameter_groups],
                                reference=[list(g) for g in reference.parameter_groups]))
    if candidate.initial_state != reference.initial_state:
        differ = sorted(k for k in set(candidate.initial_state) | set(reference.initial_state)
                        if candidate.initial_state.get(k) != reference.initial_state.get(k))
        findings.append(failure("initial_state_mismatch",
                                "the deployment starts from different weights or buffers",
                                differing=differ[:16]))
    pairs, presence = C.match_results(contract, candidate.results, reference.results)
    checks = {spec.name: ResultCheck(spec.name, spec.kind) for spec in contract.specs
              if spec.presence != C.ABSENT}
    for finding in presence:
        target = checks.get(finding.subject or "")
        if target is not None:
            target.findings.append(finding)
        else:
            findings.append(finding)
    limits = thresholds or {}
    for pair in pairs:
        check = judge(pair.spec, pair.candidate, pair.reference,
                      None if thresholds is None else limits.get(pair.spec.name))
        checks[pair.spec.name] = check
    return list(checks.values()), findings


def side_findings(ex: Execution, out: list[Finding], *, reference: bool) -> None:
    def note(code: str, message: str, **details: Any) -> None:
        if reference:
            out.append(missing("reference_" + code, message, subject=ex.deployment.name, **details))
        else:
            out.append(failure(code, message, subject=ex.deployment.name, **details))

    if ex.state_changed:
        note("parameters_modified", "parameters or buffers changed during one forward/backward "
                                    "(an optimizer step or an in-place write)",
             changed=ex.state_changed[:16])
    if ex.mutated:
        note("input_mutation", f"inputs changed: {ex.mutated}", inputs=ex.mutated)
    if ex.counts != ex.expected:
        note("coverage_mismatch",
             f"site calls {ex.counts} differ from the declared {ex.expected}",
             observed=ex.counts, expected=ex.expected)


# ── boundaries ───────────────────────────────────────────────────────────────


def run_callable(fn: Callable, inputs: tuple[tuple[Any, ...], dict[str, Any]],
                 grad_paths: Sequence[str], cotangents: Mapping[str, torch.Tensor],
                 forward_context: Callable[[], Any] | None = None) -> dict[str, Any]:
    """``fn`` once on isolated copies of a recorded call: outputs and VJP by path.

    ``forward_context`` wraps the forward only (the scope's autocast), exactly
    as :func:`execute` does; the backward runs outside it, as in the scope.
    """
    args, kwargs = isolate(inputs)
    roots: dict[str, Any] = {"args": tuple(args), "kwargs": dict(kwargs)}
    vjp: dict[str, torch.Tensor] = {}
    for path in grad_paths:
        # Per-argument identity views, as at the live call: an input passed at
        # two positions yields one gradient per position, not their sum.
        leaf = resolve(roots, path).requires_grad_(True)
        local = leaf.view_as(leaf)
        local.register_hook(capture_grad(vjp, path))
        roots = replace_at(roots, path, local)
    with torch.enable_grad(), (forward_context() if forward_context else contextlib.nullcontext()):
        out = fn(*roots["args"], **roots["kwargs"])
    outputs = dict(tensors_in(out, "out"))
    # A legal backward can reuse a dead output buffer. Compare the forward's
    # actual values, not whatever the buffer contains after backward.
    results: dict[str, Any] = isolate({f"{C.PREFIX[C.OUTPUT]}{p}": t.detach()
                                     for p, t in outputs.items()})
    tensors = [(p, t) for p, t in outputs.items() if t.requires_grad]
    if tensors:
        grads = [cotangents.get(p, torch.zeros_like(t)) for p, t in tensors]
        torch.autograd.backward([t for _, t in tensors], grads)
    for path in grad_paths:
        results[f"{C.PREFIX[C.INPUT_GRAD]}{path}"] = vjp.get(path)
    return results


def boundary_name(call: BoundaryCall, result: str) -> str:
    return f"{BOUNDARY}{call.key}/{result}"


def boundary_contract(call: BoundaryCall, native: Mapping[str, Any]) -> C.ResultContract:
    specs = []
    for key, value in native.items():
        kind = C.OUTPUT if key.startswith(C.PREFIX[C.OUTPUT]) else C.INPUT_GRAD
        specs.append(_bound(C.ResultSpec(key, kind, presence=C.REQUIRED if value is not None
                                         else C.ABSENT), value))
    return C.ResultContract(tuple(specs))


def boundary_results(adapter: ScopeAdapter, deployment: Deployment, call: BoundaryCall,
                     *, native_fn: Callable | None = None) -> tuple[dict, dict, dict]:
    """Native and deployed site implementations on one recorded call's inputs.

    Returns ``(native, deployed, deployed_again)``: the second deployed run is
    the repeatability (purity) evidence -- the same inputs must give the same
    answer.
    """
    native_fn = native_fn or adapter.native_site(call.site)
    deployed_fn = deployment.sites[call.site]
    # Same numerics settings as the scope run (autocast included); one seed
    # per call so a site that draws randomness draws the same on every side.
    runs = []
    for fn in (native_fn, deployed_fn, deployed_fn):
        with isolated_rng(call.index, adapter.device):
            runs.append(run_callable(fn, call.inputs, call.grad_paths, call.upstream,
                                     forward_context=adapter.numerics_context))
    return tuple(runs)


def boundary_checks(adapter: ScopeAdapter, deployment: Deployment, execution: Execution,
                    thresholds: Mapping[str, Mapping[str, float]] | None
                    ) -> tuple[list[ResultCheck], list[Finding], dict[str, Any]]:
    """Every recorded call of every replaced site, judged like any other result.

    Each call is a sub-context with a stable name ``boundary:<site>#<k>/...``.
    Gradients for inputs the site received as arguments are attributable here;
    parameters a site closes over are not, and are covered by the scope's own
    parameter-gradient results -- the evidence says so.
    """
    checks: list[ResultCheck] = []
    findings: list[Finding] = []
    evidence: dict[str, Any] = {"calls": len(execution.boundaries),
                                "attribution": "gradients of floating tensor arguments; "
                                               "closed-over parameters are covered by the "
                                               "scope's parameter-gradient results",
                                "upstream": "external cotangents at returned identity views; "
                                            "internal site paths excluded; output aliases seeded once",
                                "live_gradients": {}}
    for call in execution.boundaries:
        if call.unattributable_outputs:
            findings.append(missing("boundary_cotangent_unattributable",
                                    f"cannot isolate output cotangents: {call.unattributable_outputs}",
                                    subject=call.key))
            continue
        if not any(True for _ in tensors_in(call.outputs)):
            findings.append(missing("boundary_uncaptured",
                                    f"the site returned {type(call.outputs).__name__}, which holds "
                                    "no tensor this observer can see; nothing was checked",
                                    subject=call.key))
            continue
        try:
            native, deployed, again = boundary_results(adapter, deployment, call)
        except Exception as exc:
            findings.append(failure("boundary_runtime_error", f"{type(exc).__name__}: {exc}",
                                    subject=call.key))
            continue
        contract = boundary_contract(call, native)
        pairs, presence = C.match_results(contract, deployed, native)
        findings.extend(presence)
        for pair in pairs:
            name = boundary_name(call, pair.spec.name)
            spec = C.ResultSpec(name, pair.spec.kind, exact=pair.spec.exact)
            limit = None if thresholds is None else thresholds.get(name)
            checks.append(judge(spec, pair.candidate, pair.reference, limit))
            # Repeatability: the deployed site again on identical inputs, held
            # to the same bound. A site that remembers earlier calls fails here.
            repeat = judge(C.ResultSpec(name + "~repeat", pair.spec.kind, exact=pair.spec.exact),
                           again.get(pair.spec.name), pair.candidate, limit)
            checks.append(repeat)
        live = dict(tensors_in(call.outputs, "out"))
        for path, value in live.items():
            key = f"{C.PREFIX[C.OUTPUT]}{path}"
            if key in deployed and torch.is_tensor(deployed[key]):
                name = boundary_name(call, key)
                limit = None if thresholds is None else thresholds.get(name)
                checks.append(judge(C.ResultSpec(name + "~live", C.OUTPUT),
                                    deployed[key], value, limit))
        # The gradients this call's real backward emitted, against the native
        # VJP on the same live inputs and live upstream gradient.
        more, status = live_gradient_checks(call, native, thresholds)
        checks.extend(more[0])
        findings.extend(more[1])
        evidence["live_gradients"][status] = evidence["live_gradients"].get(status, 0) + 1
    return checks, findings, evidence


LIVE = "~live"


def live_gradient_pairs(call: BoundaryCall, native: Mapping[str, Any]
                        ) -> tuple[list[C.Pairing], list[Finding], str]:
    """Pair each argument's actually emitted gradient with the native VJP.

    Returns the pairs (named ``boundary:<site>#<k>/grad_in:<path>~live``),
    presence findings, and a status: ``compared``; ``no_backward`` when no
    gradient reached the call's differentiable outputs (nothing was emitted,
    nothing to compare); ``no_gradient_inputs``. An argument whose gradient
    could not be isolated is reported ``incomplete``, never skipped.
    """
    findings: list[Finding] = []
    for path in call.unattributable:
        findings.append(missing("boundary_gradient_unattributable",
                                f"the gradient emitted for {path} could not be isolated to this call",
                                subject=boundary_name(call, f"{C.PREFIX[C.INPUT_GRAD]}{path}{LIVE}")))
    paths = [p for p in call.grad_paths if p not in call.unattributable]
    if not paths:
        return [], findings, "no_gradient_inputs"
    if call.differentiable_outputs and not call.upstream:
        return [], findings, "no_backward"
    def named(key: str) -> str:
        return boundary_name(call, key) + LIVE
    reference = {named(f"{C.PREFIX[C.INPUT_GRAD]}{p}"): native.get(f"{C.PREFIX[C.INPUT_GRAD]}{p}")
                 for p in paths}
    actual = {named(f"{C.PREFIX[C.INPUT_GRAD]}{p}"): call.emitted.get(p) for p in paths}
    contract = C.ResultContract(tuple(
        _bound(C.ResultSpec(k, C.INPUT_GRAD, presence=C.REQUIRED if v is not None else C.ABSENT), v)
        for k, v in reference.items()))
    pairs, presence = C.match_results(contract, actual, reference)
    return pairs, findings + presence, "compared"


def live_gradient_checks(call: BoundaryCall, native: Mapping[str, Any],
                         thresholds: Mapping[str, Mapping[str, float]] | None
                         ) -> tuple[tuple[list[ResultCheck], list[Finding]], str]:
    """:func:`live_gradient_pairs` through :func:`judge`, with the boundary's own
    frozen threshold (the name without ``~live``)."""
    pairs, findings, status = live_gradient_pairs(call, native)
    checks = []
    for pair in pairs:
        base = pair.spec.name[:-len(LIVE)]
        limit = None if thresholds is None else thresholds.get(base)
        checks.append(judge(C.ResultSpec(pair.spec.name, C.INPUT_GRAD), pair.candidate,
                            pair.reference, limit))
    return (checks, findings), status


# ── the gate ─────────────────────────────────────────────────────────────────


def verify_integration(adapter: ScopeAdapter, deployment: Deployment, *, policy: Any,
                       samples: Sequence[Sample]) -> VerificationReport:
    """Judge ``deployment`` against native eager on every validation sample.

    ``policy`` is a frozen :class:`~evograd.verification.gate2.calibration.Gate2Policy`
    calibrated for this adapter, patch set and environment. Its sample
    protocol decides which samples it may judge:

    * ``disjoint_holdout`` (operator scope): samples disjoint from the
      calibration samples, by id, seed and content; pooled thresholds;
    * ``matched_inputs`` (model scope): exactly the calibrated samples, each
      re-executed fresh and matched by content fingerprint; each sample is
      judged against its own per-tensor thresholds.

    Without a valid, matching policy everything still runs and is measured,
    and the affected results are ``incomplete``: a binding mismatch withholds
    the thresholds (no pass, no fail) rather than judging with the wrong ones.
    """
    from evograd.verification.gate2.calibration import (
        BINDING_CODES,
        MATCHED,
        CalibrationError,
        Gate2Policy,
        policy_findings,
    )

    report = VerificationReport(
        gate=GATE, protocol=PROTOCOL,
        subject={"scope": adapter.scope, "adapter": dict(adapter.identity()),
                 "numerics": dict(adapter.numerics()), "deployment": deployment.describe(),
                 "reference": {"kind": "native_eager", "scope": adapter.scope},
                 "metrics": "E2=||c-r||2/max(||r||2,1e-30); Emax=||c-r||inf/max(RMS(r),1e-30)",
                 "samples": [s.to_dict() for s in samples]},
    )
    if adapter.scope not in SCOPES:
        report.findings.append(missing("unknown_scope", f"scope {adapter.scope!r} is not a Gate 2 "
                                                        f"scope {SCOPES}"))
    if deployment.role == NATIVE:
        report.findings.append(missing("not_a_candidate", "the native deployment is the reference"))
    unknown = sorted(set(deployment.patch_set) - set(adapter.sites()))
    if unknown:
        report.findings.append(failure("unknown_sites", f"sites {unknown} are not replaceable here"))
    report.findings.extend(policy_findings(policy, adapter, deployment, samples))
    if isinstance(policy, Mapping):
        try:
            policy = Gate2Policy.from_dict(policy)
        except CalibrationError:
            policy = None
    if not isinstance(policy, Gate2Policy):
        policy = None
    # Thresholds bound to another scope, case, reference, environment, patch
    # set or sample protocol are not evidence about this run: measure, but
    # leave every numeric result undecided (no pass, no fail).
    withheld = sorted({f.code for f in report.findings} & BINDING_CODES)
    if withheld:
        report.evidence["thresholds_withheld"] = withheld
    protocol = policy.sample_protocol if policy is not None else None
    report.evidence["sample_protocol"] = protocol
    record = bool(policy is not None and policy.boundaries)
    report.subject["policy"] = None if policy is None else {
        "policy_hash": policy.policy_hash, "schema": policy.schema, "boundaries": policy.boundaries,
        "sample_protocol": protocol,
        "reference_implementation": {k: v.get("digest") for k, v in
                                     policy.reference["implementation"].items()}}
    used_executions = set(policy.calibration_executions) if policy is not None else set()
    for sample in samples:
        case = CaseReport(case_id=sample.sample_id, evidence={"sample": sample.to_dict()})
        report.cases.append(case)
        # The previous sample's executions hold its results (for a whole model,
        # every parameter gradient); release them before running the next one.
        reference = cand = None
        reference = execute(adapter, Deployment.native(), sample)
        case.evidence["reference"] = reference.evidence()
        if reference.error:
            case.findings.append(missing("reference_runtime_error",
                                         f"native run failed at {reference.error_stage}",
                                         trace=reference.error))
            continue
        cand = execute(adapter, deployment, sample, record_boundaries=record)
        case.evidence["candidate"] = cand.evidence()
        if cand.error:
            code = "output_structure" if cand.error_stage == "outputs" else "candidate_runtime_error"
            case.findings.append(failure(code, f"deployment failed at {cand.error_stage}",
                                         trace=cand.error))
            continue
        contract = integration_contract(reference)
        case.evidence["contract_digest"] = contract.digest()
        unbound: list[str] = []
        if policy is not None:
            if contract.digest() != policy.contract_digest:
                unbound.append("contract_mismatch")
                case.findings.append(missing("contract_mismatch",
                                             "this sample's result contract differs from the policy's"))
            if used_executions & {reference.execution_id, cand.execution_id}:
                unbound.append("calibration_execution_reused")
                case.findings.append(missing("calibration_execution_reused",
                                             "a calibration execution is not fresh evidence"))
            if protocol == MATCHED:
                unbound.extend(_matched_checks(policy, sample, reference, cand, case))
            else:
                used = {s.get("input_digest") for s in policy.calibration_samples} - {None}
                if reference.input_digest in used:
                    unbound.append("sample_overlap")
                    case.findings.append(missing("sample_overlap", "this sample's inputs and "
                                                 "cotangents are identical to a calibration sample's"))
                fixed = (policy.weights or {}).get("fixed")
                if fixed is not None and C.stable_hash(reference.initial_state) != fixed:
                    unbound.append("policy_weights_mismatch")
                    case.findings.append(missing("policy_weights_mismatch",
                                                 "the native weights differ from those the policy "
                                                 "was calibrated on"))
        thresholds = None
        if policy is not None and not withheld and not unbound:
            thresholds = policy.thresholds_for(sample.sample_id)
        if unbound:
            case.evidence["thresholds_withheld"] = sorted(set(unbound))
        results, findings = compare_executions(contract, reference, cand, thresholds)
        case.results.extend(results)
        case.findings.extend(findings)
        if record:
            checks, more, evidence = boundary_checks(adapter, deployment, cand, thresholds)
            case.results.extend(checks)
            case.findings.extend(more)
            case.evidence["boundaries"] = evidence
            if not cand.boundaries and deployment.sites:
                case.findings.append(failure("no_boundary_calls",
                                             "boundaries were required but no replaced site ran"))
    return report


def _matched_checks(policy: Any, sample: Sample, reference: Execution, candidate: Execution,
                    case: CaseReport) -> list[str]:
    """``matched_inputs``: the fresh executions must be on the calibrated sample's
    exact content. Returns the codes that withhold this sample's thresholds."""
    codes: list[str] = []
    stored = (policy.sample_identities or {}).get(sample.sample_id)
    if stored is None:
        return ["sample_not_calibrated"]          # reported by policy_findings
    differ = identity_differences(stored, reference.sample_identity)
    if differ or reference.sample_identity.get("opaque"):
        codes.append("sample_identity_mismatch")
        case.findings.append(missing(
            "sample_identity_mismatch",
            f"the native execution's content differs from calibrated sample "
            f"{sample.sample_id!r} in {differ or 'opaque inputs'}; its thresholds are not used",
            differing=differ, opaque=reference.sample_identity.get("opaque")))
    # The candidate is run on the same sample: same inputs, context, cotangents
    # and seed. (Different initial weights are a candidate failure, judged by
    # compare_executions as initial_state_mismatch.)
    cand_differ = identity_differences(reference.sample_identity, candidate.sample_identity,
                                       [k for k in IDENTITY_COMPONENTS if k != "initial_state"])
    if cand_differ:
        codes.append("candidate_sample_mismatch")
        case.findings.append(missing("candidate_sample_mismatch",
                                     f"the candidate ran on different content: {cand_differ}",
                                     differing=cand_differ))
    case.evidence["matched_sample"] = {
        "calibrated_fingerprint": stored.get("fingerprint"),
        "reference_fingerprint": reference.sample_identity.get("fingerprint"),
        "candidate_fingerprint": candidate.sample_identity.get("fingerprint"),
        "identical_content": not differ and not cand_differ,
        # Evidence, not a condition: does the fresh native reproduce the
        # calibration native's results bit for bit?
        "reference_reproduces_calibration": (
            stored.get("reference_results_digest") is not None
            and stored.get("reference_results_digest") == reference.results_digest)}
    return codes


def combine_scopes(required: Sequence[str], reports: Mapping[str, VerificationReport]
                   ) -> VerificationReport:
    """One verdict over the scopes an experiment declared *before* running.

    Every scope in ``required`` must have a report; a missing one leaves the
    result incomplete. Reports for undeclared scopes are kept as evidence and
    do not decide anything -- a scope cannot be added or dropped after seeing
    its result.
    """
    combined = VerificationReport(gate=GATE, protocol=PROTOCOL,
                                  subject={"required_scopes": list(required),
                                           "reported_scopes": sorted(reports)})
    if not required:
        combined.findings.append(missing("no_required_scopes", "no scope was declared"))
    for scope in required:
        if scope not in SCOPES:
            combined.findings.append(missing("unknown_scope", f"{scope!r} is not a Gate 2 scope "
                                                              f"{SCOPES}", subject=scope))
    for scope in required:
        report = reports.get(scope)
        if report is None:
            combined.findings.append(missing("required_scope_missing", f"no {scope} report",
                                             subject=scope))
            continue
        for case in report.cases:
            case.evidence.setdefault("scope", scope)
            combined.cases.append(case)
        combined.findings.extend(report.findings)
    combined.evidence["undeclared_scopes"] = {
        name: report.to_dict() for name, report in reports.items() if name not in required}
    return combined
