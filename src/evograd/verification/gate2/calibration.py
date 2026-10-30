"""Gate 2 calibration: thresholds from trusted controls only, frozen before use.

Two sample protocols, one per scope, recorded in the policy
(:data:`SAMPLE_PROTOCOLS`):

* ``disjoint_holdout`` (operator scope). For every required result ``z`` and
  metric ``m in {E2, Emax}``, pooled over the calibration samples ``c``::

      T[z, m] = gamma * max( max_c E_m(native repeat_c, native_c),
                             max_c E_m(trusted compile_c, native_c),
                             floor[z, m] )

  and judged on validation samples disjoint from them (id, seed, content).

* ``matched_inputs`` (model scope). Pooled over the fixed samples ``x``::

      T[z, m] = gamma * max( max_x max_r E_m(native repeat_r[x], native[x]),
                             max_x E_m(trusted compile[x], native[x]),
                             floor[z, m] )

  and judged on fresh executions of the *same* samples: a sample is used only
  when its content fingerprint (inputs, labels and other context, cotangents,
  initial weights, module mode, seed) equals the calibrated one. The budget of
  a tensor is the largest drift the controls showed on it over the calibrated
  inputs (the operator scope's formula over its calibration payloads): a
  per-sample budget was a single draw of compile's rounding on that tensor,
  and a candidate whose rounding differs from compile's exceeded it on a few
  scattered tensors per sample for no fault of its own.

``gamma`` is the named safety factor
:data:`SAFETY_FACTOR_GAMMA` (initially 2); floors are the starting values
discussed in review -- outputs ``(1e-5, 1e-3)``, gradients ``(1e-4, 1e-2)`` --
and are design points to be checked by measurement, not bounds.

Rules this module enforces:

* the calibration API never takes a candidate: it builds the native and
  compiled deployments itself, and :func:`derive_thresholds` accepts only
  :class:`ControlMeasurement` objects;
* a control must qualify (executes, finite, the native contract, the same
  parameters and weights, every replaced site called as declared, and -- for
  ``torch.compile`` -- the compiler backend actually invoked at every
  replaced site and a second run reproducing the first bit for bit). A
  disqualified or missing control yields no policy. Whether the native
  repeats were bitwise is recorded per sample (``repeatability``);
* each tensor keeps its own thresholds, matched by stable name (and, for
  ``matched_inputs``, by sample);
* the policy binds scope, sample protocol, case identity, numerics,
  environment, patch set, output contract, reference and control identities,
  the calibration samples with their content identities, and the calibration
  execution ids; :func:`policy_findings` refuses any mismatch; for
  ``disjoint_holdout`` also any overlap between calibration and validation
  samples, for ``matched_inputs`` any sample that was not calibrated.
* every threshold entry is re-derived from its recorded components when a
  policy is loaded; an entry that does not follow the formula is refused.
"""

from __future__ import annotations

import itertools
import json
import math
import time
import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import CONTROL, Deployment, Sample
from evograd.verification.common.reports import Finding, jsonable, missing, summarize
from evograd.verification.common.tensors import EPSILON, tensors_in, isolated_rng
from evograd.verification.gate2.runner import (
    BOUNDARY,
    LIVE,
    METRICS,
    Execution,
    identity_differences,
    ScopeAdapter,
    boundary_contract,
    boundary_name,
    boundary_results,
    compare_executions,
    execute,
    integration_contract,
    judge,
    live_gradient_pairs,
    run_callable,
)

#: /4 records the sample protocol (and per-sample thresholds and content
#: identities for ``matched_inputs``). /3 policies are never reinterpreted.
SCHEMA = "evograd-verification-gate2-policy/4"

#: Sample protocols. The scope decides which one applies; the policy records it.
DISJOINT = "disjoint_holdout"
MATCHED = "matched_inputs"
SAMPLE_PROTOCOLS = {"operator": DISJOINT, "model": MATCHED}

#: Findings that make a policy's thresholds inapplicable to the whole run:
#: verify_integration measures but withholds every numeric judgement.
BINDING_CODES = frozenset({
    "policy_schema", "policy_protocol_mismatch", "policy_scope_mismatch", "policy_case_mismatch",
    "policy_numerics_mismatch", "policy_patch_set_mismatch", "policy_reference_mismatch",
    "reference_unidentifiable", "policy_environment_mismatch", "sample_overlap", "unknown_scope",
})

#: The named safety factor. A design choice, not a guarantee.
SAFETY_FACTOR_GAMMA = 2.0

#: Initial floors per result family. To be re-measured, not trusted blindly.
DEFAULT_FLOORS: dict[str, dict[str, float]] = {
    "output": {"E2": 1e-5, "Emax": 1e-3},
    "gradient": {"E2": 1e-4, "Emax": 1e-2},
}

#: The calibration procedure, for stage caches that judged under an earlier one.
CALIBRATION_PROCEDURE = "gate2-calibration/2: compile repeat under the first run's boundary recording"
NATIVE_REPEAT = "native_repeat"
TRUSTED_COMPILE = "trusted_compile"
CONTROL_SOURCES = (NATIVE_REPEAT, TRUSTED_COMPILE)


class CalibrationError(ValueError):
    """The evidence is not sufficient to derive a threshold."""


def family_of(kind: str) -> str:
    try:
        return C.FLOOR_FAMILY[kind]
    except KeyError:
        raise CalibrationError(f"result kind {kind!r} has no floor family") from None


def validate_gamma(gamma: Any) -> float:
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise CalibrationError(f"gamma must be a number, got {gamma!r}")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 1.0:
        raise CalibrationError(f"gamma must be finite and >= 1 (a safety factor), got {gamma}")
    return gamma


def validate_floors(floors: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for family in ("output", "gradient"):
        if family not in floors:
            raise CalibrationError(f"floors need a {family!r} entry")
        out[family] = {}
        for metric in METRICS:
            value = floors[family].get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise CalibrationError(f"floor {family}/{metric} must be finite and > 0, "
                                       f"got {value!r}")
            out[family][metric] = float(value)
    return out


@dataclass(frozen=True)
class ControlMeasurement:
    """One trusted control's error on one tensor of one calibration sample."""

    source: str
    sample_id: str
    result: str
    kind: str
    E2: float
    Emax: float
    #: Largest elementwise |control - native|; Gate 1's envelope is built from it.
    max_abs: float = 0.0

    def __post_init__(self) -> None:
        if not (isinstance(self.max_abs, float) and math.isfinite(self.max_abs) and self.max_abs >= 0):
            raise CalibrationError(f"{self.result}: max_abs={self.max_abs!r} is not a finite "
                                   "non-negative measurement")
        if self.source not in CONTROL_SOURCES:
            raise CalibrationError(f"{self.result}: {self.source!r} is not a trusted control "
                                   f"source {CONTROL_SOURCES}")
        for metric in METRICS:
            value = getattr(self, metric)
            if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                raise CalibrationError(f"{self.result}: {metric}={value!r} is not a finite "
                                       "non-negative measurement")


def derive_thresholds(measurements: Sequence[ControlMeasurement], *,
                      results: Mapping[str, str], gamma: float = SAFETY_FACTOR_GAMMA,
                      floors: Mapping[str, Mapping[str, float]] = DEFAULT_FLOORS
                      ) -> dict[str, dict[str, Any]]:
    """Per-tensor thresholds from trusted control measurements.

    ``results`` maps every result that needs a threshold to its kind. Each
    needs at least one native-repeat and one trusted-compile measurement; a
    result with either missing is an error, never a fallback. Measurements for
    names outside ``results`` are refused rather than pooled.
    """
    gamma = validate_gamma(gamma)
    floors = validate_floors(floors)
    by_result: dict[str, dict[str, list[ControlMeasurement]]] = {
        name: {s: [] for s in CONTROL_SOURCES} for name in results}
    for m in measurements:
        if type(m) is not ControlMeasurement:
            raise TypeError(f"only ControlMeasurement may calibrate a threshold, got {type(m).__name__}")
        if m.result not in by_result:
            raise CalibrationError(f"measurement for {m.result!r}, which is not a calibrated result")
        if m.kind != results[m.result]:
            raise CalibrationError(f"{m.result}: measured as {m.kind!r}, contract says "
                                   f"{results[m.result]!r}")
        by_result[m.result][m.source].append(m)
    out: dict[str, dict[str, Any]] = {}
    for name, kind in results.items():
        groups = by_result[name]
        lacking = [s for s in CONTROL_SOURCES if not groups[s]]
        if lacking:
            raise CalibrationError(f"{name}: no {' or '.join(lacking)} measurement")
        floor = floors[family_of(kind)]
        entry: dict[str, Any] = {"kind": kind, "components": {}}
        for metric in METRICS:
            native = max(getattr(m, metric) for m in groups[NATIVE_REPEAT])
            compiled = max(getattr(m, metric) for m in groups[TRUSTED_COMPILE])
            entry[metric] = gamma * max(native, compiled, floor[metric])
            entry["components"][metric] = {"native_repeat": native, "trusted_compile": compiled,
                                           "floor": floor[metric], "gamma": gamma,
                                           "binding": max((native, "native_repeat"),
                                                          (compiled, "trusted_compile"),
                                                          (floor[metric], "floor"))[1]}
        # gamma x compile's largest elementwise deviation; Gate 1 rescales it to its own
        # factor (integrations.gates.gate1_envelope).
        entry["envelope"] = gamma * max(m.max_abs for m in groups[TRUSTED_COMPILE])
        entry["samples"] = {s: sorted({m.sample_id for m in groups[s]}) for s in CONTROL_SOURCES}
        out[name] = entry
    return out


# ── the frozen policy ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gate2Policy:
    """A calibrated, frozen Gate 2 policy. Immutable: the content is canonical JSON."""

    text: str

    def __post_init__(self) -> None:
        try:
            data = json.loads(self.text)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"policy is not JSON: {exc}") from None
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            schema = data.get("schema") if isinstance(data, dict) else None
            raise CalibrationError(f"policy schema {schema!r} is not {SCHEMA!r}; "
                                   "recalibrate under the current protocol")
        absent = [k for k in REQUIRED_FIELDS if k not in data]
        protocol = data.get("sample_protocol")
        if protocol not in (DISJOINT, MATCHED):
            raise CalibrationError(f"unknown sample protocol {protocol!r}")
        absent += [k for k in PROTOCOL_FIELDS[protocol] if k not in data]
        if absent:
            raise CalibrationError(f"policy lacks {absent}")
        if not isinstance(data["reference"], dict) or not data["reference"].get("implementation"):
            raise CalibrationError("policy lacks the reference implementation identity")
        if data.get("policy_hash") != _content_hash(data):
            raise CalibrationError("policy_hash does not match the policy content")
        _check_thresholds(data)

    @staticmethod
    def build(content: Mapping[str, Any]) -> "Gate2Policy":
        data = jsonable(dict(content))
        data["schema"] = SCHEMA
        data["policy_hash"] = _content_hash(data)
        return Gate2Policy(json.dumps(data, sort_keys=True, allow_nan=False))

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Gate2Policy":
        try:
            text = json.dumps(dict(data), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"policy cannot be frozen: {exc}") from None
        return Gate2Policy(text)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.text)

    def thresholds_for(self, sample_id: str) -> dict[str, Any] | None:
        """The thresholds that judge ``sample_id``: the pooled ones under
        ``disjoint_holdout``; that sample's own under ``matched_inputs``
        (``None`` for a sample that was not calibrated)."""
        data = self.to_dict()
        if data["sample_protocol"] == MATCHED:
            return data["thresholds_by_sample"].get(sample_id)
        return data["thresholds"]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "text":
            raise AttributeError(name)
        data = json.loads(object.__getattribute__(self, "text"))
        if name in data:
            return data[name]
        raise AttributeError(name)


#: What every policy must carry; a hash-valid policy without them is refused.
REQUIRED_FIELDS = ("scope", "sample_protocol", "adapter", "numerics", "environment", "patch_set",
                   "reference", "contract", "contract_digest", "calibration_samples",
                   "sample_identities", "calibration_executions", "weights", "gamma", "floors",
                   "epsilon", "boundaries", "formula", "policy_hash")
#: And per sample protocol.
PROTOCOL_FIELDS = {DISJOINT: ("thresholds",), MATCHED: ("thresholds_by_sample",)}

FORMULAS = {
    DISJOINT: "T[z,m] = gamma * max(max_c E_m(native_repeat_c, native_c), "
              "max_c E_m(compile_c, native_c), floor[z,m])",
    MATCHED: "T[z,m] = gamma * max(max_x max_r E_m(native_repeat_r[x], native[x]), "
             "max_x E_m(compile[x], native[x]), floor[z,m]), judged on the calibrated samples x",
}


def _content_hash(data: Mapping[str, Any]) -> str:
    return C.stable_hash({k: v for k, v in data.items() if k != "policy_hash"})


def _check_thresholds(data: Mapping[str, Any]) -> None:
    """Every entry must follow the policy's formula from its own components."""
    protocol = data["sample_protocol"]
    if data.get("formula") != FORMULAS[protocol]:
        raise CalibrationError("the policy formula is not the protocol's")
    gamma = validate_gamma(data["gamma"])
    floors = validate_floors(data["floors"])
    if protocol == MATCHED:
        ids = [s.get("sample_id") for s in data["calibration_samples"]]
        if sorted(data["thresholds_by_sample"]) != sorted(ids) or \
                sorted(data["sample_identities"]) != sorted(ids) or len(set(ids)) != len(ids):
            raise CalibrationError("per-sample thresholds and identities must cover exactly the "
                                   "calibration samples")
        groups = list(data["thresholds_by_sample"].items())
    else:
        groups = [("pooled", data["thresholds"])]
    for label, table in groups:
        if not isinstance(table, Mapping) or not table:
            raise CalibrationError(f"{label}: no thresholds")
        for name, entry in table.items():
            _check_entry(f"{label}/{name}", entry, gamma, floors)


def _check_entry(name: str, entry: Any, gamma: float, floors: Mapping[str, Mapping[str, float]]
                 ) -> None:
    def number(value: Any) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)

    if not isinstance(entry, Mapping):
        raise CalibrationError(f"{name}: malformed threshold entry")
    floor = floors[family_of(entry.get("kind"))]
    for metric in METRICS:
        parts = (entry.get("components") or {}).get(metric) or {}
        values = [parts.get(k) for k in ("native_repeat", "trusted_compile", "floor")]
        if not (number(entry.get(metric)) and all(number(v) for v in values)):
            raise CalibrationError(f"{name}/{metric}: missing or non-finite threshold or component")
        if parts.get("floor") != floor[metric] or parts.get("gamma") != gamma:
            raise CalibrationError(f"{name}/{metric}: floor or gamma differs from the policy's")
        if entry[metric] != gamma * max(values):
            raise CalibrationError(f"{name}/{metric}: {entry[metric]!r} is not gamma * max of its "
                                   "components")


def policy_findings(policy: Any, adapter: ScopeAdapter, deployment: Deployment,
                    samples: Sequence[Sample]) -> list[Finding]:
    """Everything that stops ``policy`` from judging this run. All ``incomplete``."""
    out: list[Finding] = []
    if not samples:
        out.append(missing("no_samples", "no validation sample"))
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        out.append(missing("duplicate_samples", "validation sample ids repeat"))
    if policy is None:
        out.append(missing("no_policy", "no calibrated policy; measurements are recorded, "
                                        "nothing can pass"))
        return out
    if isinstance(policy, Mapping):
        try:
            policy = Gate2Policy.from_dict(policy)
        except CalibrationError as exc:
            out.append(missing("policy_schema", str(exc)))
            return out
    if not isinstance(policy, Gate2Policy):
        out.append(missing("policy_schema", f"not a Gate 2 policy: {type(policy).__name__}"))
        return out
    data = policy.to_dict()
    checks = (
        ("policy_scope_mismatch", data.get("scope"), adapter.scope),
        ("policy_protocol_mismatch", data.get("sample_protocol"),
         SAMPLE_PROTOCOLS.get(adapter.scope)),
        ("policy_case_mismatch", data.get("adapter"), jsonable(dict(adapter.identity()))),
        ("policy_numerics_mismatch", data.get("numerics"), jsonable(dict(adapter.numerics()))),
        ("policy_patch_set_mismatch", data.get("patch_set"), list(deployment.patch_set)),
    )
    for code, stored, live in checks:
        if stored != live:
            out.append(missing(code, f"policy has {stored!r}, this run has {live!r}"))
    stored_ref = (data.get("reference") or {}).get("implementation") or {}
    try:
        live_ref = jsonable(adapter.reference_identity())
    except Exception as exc:  # UnidentifiableReference, or an adapter that declares none
        out.append(missing("reference_unidentifiable", f"{type(exc).__name__}: {exc}"))
    else:
        # ``fingerprint_version`` too: a digest computed under another version
        # of the fingerprint is a different identity, never reinterpreted (a
        # version-1 policy predates the field and is refused here).
        changed = sorted(k for k in set(stored_ref) | set(live_ref)
                         if any((stored_ref.get(k) or {}).get(f) != (live_ref.get(k) or {}).get(f)
                                for f in ("digest", "opaque", "fingerprint_version")))
        if changed:
            out.append(missing("policy_reference_mismatch",
                               f"the reference implementation changed since calibration: {changed}; "
                               "its thresholds are not used", parts=changed))
    env = C.environment_fingerprint(adapter.device)
    differences = C.environment_differences(data.get("environment") or {}, jsonable(env))
    if differences:
        out.append(missing("policy_environment_mismatch", "; ".join(differences[:6]),
                           differences=differences))
    calibration = data.get("calibration_samples") or []
    if data.get("sample_protocol") == MATCHED:
        # The same samples, re-executed: each must be a calibrated sample (same
        # id, seed and recipe; its content is checked when it runs), and every
        # calibrated sample is required -- none may be left out.
        known = {s["sample_id"]: s for s in calibration}
        for sample in samples:
            stored = known.get(sample.sample_id)
            if stored is None or int(stored["seed"]) != int(sample.seed) or \
                    dict(stored.get("config") or {}) != jsonable(dict(sample.config)):
                out.append(missing("sample_not_calibrated",
                                   f"{sample.sample_id!r} (seed {sample.seed}) is not a calibrated "
                                   "sample of this matched-input policy; no budget applies",
                                   subject=sample.sample_id))
        absent = sorted(set(known) - set(ids))
        if absent:
            out.append(missing("required_sample_missing",
                               f"calibrated samples not verified: {absent}", samples=absent))
        return out
    used_ids = {s["sample_id"] for s in calibration}
    used_seeds = {int(s["seed"]) for s in calibration}
    overlap = sorted({s.sample_id for s in samples if s.sample_id in used_ids} |
                     {s.sample_id for s in samples if int(s.seed) in used_seeds})
    if overlap:
        out.append(missing("sample_overlap", "validation samples were used for calibration "
                                             f"(by id or seed): {overlap}", samples=overlap))
    return out


# ── controls ─────────────────────────────────────────────────────────────────


def compile_control(adapter: ScopeAdapter, patch_set: Sequence[str], *, backend: str = "inductor",
                    fullgraph: bool = True, options: Mapping[str, Any] | None = None
                    ) -> tuple[Deployment, dict[str, int]]:
    """The native implementation of every replaced site under ``torch.compile``.

    Installed at exactly the candidate's replacement sites and run through the
    same autograd path; where the adapter's deployment takes only some of a
    site's configurations (``control_site``), at exactly those calls. The
    returned counter records how many graphs the backend compiled per site; a
    site with zero did not run compiled code and the control does not qualify.
    """
    from torch._dynamo import lookup_backend

    real = lookup_backend(backend)
    graphs: dict[str, int] = {site: 0 for site in patch_set}
    sites = {}
    for site in patch_set:
        def counting(gm, example_inputs, _site=site):
            graphs[_site] += 1
            return real(gm, example_inputs)

        compiled = torch.compile(_private_entry(adapter.native_site(site)), backend=counting,
                                 fullgraph=fullgraph, **dict(options or {}))
        install = getattr(adapter, "control_site", None)
        sites[site] = install(site, compiled) if install is not None else compiled
    identity = {"kind": "torch_compile", "backend": backend, "fullgraph": fullgraph,
                "options": dict(options or {}), "torch": torch.__version__,
                "compiles": "the native implementation of each replaced site"}
    return Deployment(f"torch_compile[{backend}]", CONTROL, sites, identity), graphs


def _private_entry(fn):
    """``fn`` behind a function with its own code object.

    Dynamo keeps compiled entries on the code object it traces. Compiling the
    same native function again (another calibration in the same process, a
    new counting backend) would pile entries onto one code object until the
    recompile limit turns the next control into a hard failure. A private
    code object gives each control its own cache, freed with the control.
    It also gets its own name: dynamo keeps automatic-dynamic shape state per
    (file, line, name), so entries sharing a name would let one site's shapes
    turn the next site's dynamic (a SymBool then reaches SDPA's bool arguments).
    """
    def site_entry(*args, **kwargs):
        return fn(*args, **kwargs)

    name = f"site_entry_{next(_ENTRY_IDS)}"
    return types.FunctionType(site_entry.__code__.replace(co_name=name, co_qualname=name),
                              site_entry.__globals__, name, None, site_entry.__closure__)


_ENTRY_IDS = itertools.count()


@dataclass
class CalibrationResult:
    policy: Gate2Policy | None
    findings: list[Finding] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.policy is not None and not self.findings:
            return "calibrated"
        return "failed" if summarize(self.findings)[0] == "failed" else "incomplete"

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"status": self.status,
                         "policy_hash": None if self.policy is None else self.policy.policy_hash,
                         "findings": [f.to_dict() for f in self.findings],
                         "evidence": self.evidence})


def _qualify(contract: C.ResultContract, reference: Execution, control: Execution, source: str
             ) -> tuple[list[ControlMeasurement], list[Finding]]:
    """A control's measurements, or the reasons it does not qualify."""
    if control.error:
        return [], [missing("control_failed", f"{source} failed at {control.error_stage}",
                            subject=control.deployment.name, trace=control.error)]
    checks, findings = compare_executions(contract, reference, control, thresholds=None)
    problems = [f for f in findings]
    for check in checks:
        problems.extend(f for f in check.findings if f.code != "no_threshold")
    if problems:
        return [], [missing("control_disqualified",
                            f"{source} control did not qualify: {problems[0].code} "
                            f"({problems[0].message})", subject=control.deployment.name,
                            reasons=sorted({p.code for p in problems}))]
    out = []
    for check in checks:
        if check.metrics:
            out.append(ControlMeasurement(source, control.sample.sample_id, check.name, check.kind,
                                          float(check.metrics["E2"]), float(check.metrics["Emax"]),
                                          float(check.metrics.get("diff_max", 0.0))))
    return out, []


def _boundary_measurements(adapter: ScopeAdapter, control: Deployment, execution: Execution,
                           native_repeat_fn) -> tuple[list[ControlMeasurement], dict[str, str], list[Finding]]:
    """Site-level controls on the calls recorded under the trusted control."""
    out: list[ControlMeasurement] = []
    kinds: dict[str, str] = {}
    problems: list[Finding] = []
    sample_id = execution.sample.sample_id
    for call in execution.boundaries:
        # Do not retain the previous call's replay tensors while allocating the
        # next call. A CE gradient alone is several GiB at a model vocabulary.
        native = compiled = native_again = other = pairs = None
        if call.unattributable_outputs:
            problems.append(missing("boundary_cotangent_unattributable",
                                    f"{call.key}: {call.unattributable_outputs}", subject=call.key))
            continue
        if not any(True for _ in tensors_in(call.outputs)):
            problems.append(missing("boundary_uncaptured",
                                    f"{call.key} returned {type(call.outputs).__name__}; no tensor "
                                    "to calibrate", subject=call.key))
            continue
        try:
            native, compiled, compiled_repeat = boundary_results(adapter, control, call)
            del compiled_repeat  # not part of the calibration measurements
            # Only the repeated native result is used below. Running the full
            # triple again unnecessarily retained two extra compiled VJPs.
            with isolated_rng(call.index, adapter.device):
                native_again = run_callable(native_repeat_fn(call.site), call.inputs,
                                            call.grad_paths, call.upstream,
                                            forward_context=adapter.numerics_context)
        except Exception as exc:
            problems.append(missing("control_failed", f"boundary {call.key}: {exc}"))
            continue
        contract = boundary_contract(call, native)
        for source, other in ((NATIVE_REPEAT, native_again), (TRUSTED_COMPILE, compiled)):
            pairs, presence = C.match_results(contract, other, native)
            if presence:
                problems.append(missing("control_disqualified", f"boundary {call.key}: "
                                        f"{presence[0].code}", subject=call.key))
            for pair in pairs:
                name = boundary_name(call, pair.spec.name)
                check = judge(C.ResultSpec(name, pair.spec.kind, exact=pair.spec.exact),
                              pair.candidate, pair.reference, None)
                bad = [f for f in check.findings if f.code != "no_threshold"]
                if bad:
                    problems.append(missing("control_disqualified", f"{name}: {bad[0].code}"))
                    continue
                if check.metrics:
                    kinds[name] = pair.spec.kind
                    out.append(ControlMeasurement(source, sample_id, name, pair.spec.kind,
                                                  float(check.metrics["E2"]),
                                                  float(check.metrics["Emax"])))
        # The control's own live backward is trusted-compile evidence for the
        # same thresholds the candidate's live gradients are held to.
        pairs, findings, _status = live_gradient_pairs(call, native)
        problems.extend(missing("control_disqualified", f"{f.code}: {f.message}", subject=f.subject)
                        if f.outcome == "failed" else f for f in findings)
        for pair in pairs:
            check = judge(C.ResultSpec(pair.spec.name, C.INPUT_GRAD), pair.candidate,
                          pair.reference, None)
            bad = [f for f in check.findings if f.code != "no_threshold"]
            if bad:
                problems.append(missing("control_disqualified", f"{pair.spec.name}: {bad[0].code}"))
                continue
            if check.metrics:
                base = pair.spec.name[:-len(LIVE)]
                kinds[base] = C.INPUT_GRAD
                out.append(ControlMeasurement(TRUSTED_COMPILE, sample_id, base, C.INPUT_GRAD,
                                              float(check.metrics["E2"]),
                                              float(check.metrics["Emax"])))
    return out, kinds, problems


def calibrate(adapter: ScopeAdapter, *, patch_set: Sequence[str], samples: Sequence[Sample],
              native_repeats: int = 2, compile_backend: str = "inductor",
              compile_fullgraph: bool = True, compile_options: Mapping[str, Any] | None = None,
              gamma: float = SAFETY_FACTOR_GAMMA,
              floors: Mapping[str, Mapping[str, float]] = DEFAULT_FLOORS,
              boundaries: bool = False) -> CalibrationResult:
    """Derive and freeze a Gate 2 policy for ``patch_set`` on this adapter.

    Per calibration sample: one native reference, ``native_repeats - 1``
    further independent native runs, and one ``torch.compile`` control at the
    replaced sites (run twice; the second must reproduce the first bit for
    bit), every one on the reference's exact content. With
    ``boundaries=True`` every call recorded at a replaced site under the
    compile control gets its own thresholds too, from the native site
    implementation run twice and the compiled one, on the call's own inputs
    and upstream gradient.

    The scope's sample protocol (:data:`SAMPLE_PROTOCOLS`) decides how the
    measurements become thresholds: pooled over samples (``disjoint_holdout``)
    or per sample (``matched_inputs``).

    Returns no policy, with the reasons, if any control fails to qualify or
    any required threshold lacks evidence.
    """
    result = CalibrationResult(policy=None)
    started = time.perf_counter()
    try:
        gamma = validate_gamma(gamma)
        floors = validate_floors(floors)
    except CalibrationError as exc:
        result.findings.append(missing("invalid_calibration_parameters", str(exc)))
        return result
    protocol = SAMPLE_PROTOCOLS.get(adapter.scope)
    if protocol is None:
        result.findings.append(missing("unknown_scope", f"scope {adapter.scope!r} is not a Gate 2 "
                                                        f"scope {tuple(SAMPLE_PROTOCOLS)}"))
    patch_set = tuple(sorted(patch_set))
    if not patch_set:
        result.findings.append(missing("empty_patch_set", "a policy is calibrated for replaced sites"))
    unknown = sorted(set(patch_set) - set(adapter.sites()))
    if unknown:
        result.findings.append(missing("unknown_sites", f"{unknown} are not replaceable here"))
    if not samples:
        result.findings.append(missing("no_samples", "no calibration sample"))
    if len({s.sample_id for s in samples}) != len(samples) or \
            len({int(s.seed) for s in samples}) != len(samples):
        result.findings.append(missing("duplicate_samples", "calibration sample ids/seeds repeat"))
    if not isinstance(native_repeats, int) or native_repeats < 2:
        result.findings.append(missing("invalid_calibration_parameters",
                                       "native_repeats must be an int >= 2"))
    if result.findings:
        return result
    try:
        adapter.reference_identity()
    except Exception as exc:  # UnidentifiableReference, or an adapter that declares none
        result.findings.append(missing("reference_unidentifiable", f"{type(exc).__name__}: {exc}"))
        return result
    try:
        control, graphs = compile_control(adapter, patch_set, backend=compile_backend,
                                          fullgraph=compile_fullgraph, options=compile_options)
    except Exception as exc:
        result.findings.append(missing("control_unavailable", f"torch.compile control: {exc}"))
        return result

    measurements: list[ControlMeasurement] = []
    kinds: dict[str, dict[str, str]] = {}            # sample id -> result -> kind
    contract: C.ResultContract | None = None
    input_digests: dict[str, str | None] = {}
    identities: dict[str, dict[str, Any]] = {}
    executions: list[str] = []
    weight_digests: set[str] = set()
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        record: dict[str, Any] = {"sample": sample.to_dict()}
        per_sample.append(record)
        sid = sample.sample_id
        kinds[sid] = {}
        reference = execute(adapter, Deployment.native(), sample)
        executions.append(reference.execution_id)
        record["native"] = reference.evidence()
        if reference.error:
            result.findings.append(missing("reference_runtime_error",
                                           f"native run failed at {reference.error_stage}",
                                           subject=sid, trace=reference.error))
            continue
        if reference.sample_identity.get("opaque"):
            result.findings.append(missing(
                "sample_unidentifiable", f"inputs {reference.sample_identity['opaque']} cannot be "
                "identified by content; a policy cannot be bound to them", subject=sid))
            continue
        input_digests[sid] = reference.input_digest
        weight_digests.add(C.stable_hash(reference.initial_state))
        this = integration_contract(reference)
        bad_ref = []
        for spec in this.compared():
            value = reference.results.get(spec.name)
            if C.is_float_tensor(value) and not bool(torch.isfinite(value).all()):
                bad_ref.append(spec.name)
        if bad_ref:
            result.findings.append(missing("invalid_reference", f"non-finite native results {bad_ref}",
                                           subject=sid))
            continue
        if contract is None:
            contract = this
        elif this.digest() != contract.digest():
            result.findings.append(missing("contract_mismatch",
                                           "calibration samples disagree on the result contract",
                                           subject=sid))
            continue
        for spec in contract.compared():
            if not spec.exact:
                kinds[sid][spec.name] = spec.kind
        identity = {**reference.sample_identity, "seed": int(sample.seed),
                    "config": jsonable(dict(sample.config)),
                    "reference_results_digest": reference.results_digest,
                    "executions": {"native": reference.execution_id, "native_repeats": [],
                                   "compile": None},
                    "control_results_digests": {"native_repeats": [], "compile": None}}
        identities[sid] = identity
        for repeat in range(1, native_repeats):
            again = execute(adapter, Deployment.native(), sample)
            executions.append(again.execution_id)
            identity["executions"]["native_repeats"].append(again.execution_id)
            identity["control_results_digests"]["native_repeats"].append(again.results_digest)
            result.findings.extend(_same_content(reference, again, NATIVE_REPEAT))
            got, problems = _qualify(contract, reference, again, NATIVE_REPEAT)
            measurements.extend(got)
            result.findings.extend(problems)
            del again  # its evidence and scalar measurements have been recorded
        ctl = execute(adapter, control, sample, record_boundaries=boundaries)
        executions.append(ctl.execution_id)
        identity["executions"]["compile"] = ctl.execution_id
        identity["control_results_digests"]["compile"] = ctl.results_digest
        record["control"] = ctl.evidence()
        result.findings.extend(_same_content(reference, ctl, TRUSTED_COMPILE))
        got, problems = _qualify(contract, reference, ctl, TRUSTED_COMPILE)
        measurements.extend(got)
        result.findings.extend(problems)
        # The compile term has no repeat of its own in the budget, so the
        # control must reproduce itself bit for bit. A native repeat that is
        # not bitwise is kept in the budget and recorded here. The repeat runs
        # under the same boundary recording as the first run: recording hands
        # the site identity views of its inputs, and a run without it makes
        # Dynamo recompile; in a process that has compiled another site the
        # recompiled graph can differ at rounding level (Qwen3-0.6B qkv_norm_rope
        # after residual_rmsnorm, 2026-10-23), which is the compiler's cache
        # behaviour, not the kernels' determinism this check is about.
        again = execute(adapter, control, sample, record_boundaries=boundaries)
        executions.append(again.execution_id)
        identity["executions"]["compile_repeat"] = again.execution_id
        identity["repeatability"] = {
            "native_bitwise": all(d == reference.results_digest
                                  for d in identity["control_results_digests"]["native_repeats"]),
            "compile_bitwise": not again.error and again.results_digest == ctl.results_digest}
        if not ctl.error and not identity["repeatability"]["compile_bitwise"]:
            result.findings.append(missing("control_disqualified",
                                           "the torch.compile control did not reproduce its own "
                                           "results bit for bit", subject=sid))
        del again
        del reference  # boundary checks use the control's captured local payloads
        uncompiled = sorted(site for site, n in graphs.items() if n < 1)
        if uncompiled and not ctl.error:
            result.findings.append(missing("control_disqualified",
                                           f"torch.compile compiled no graph at sites {uncompiled}",
                                           subject=sid))
        if boundaries and not ctl.error:
            if not ctl.boundaries:
                result.findings.append(missing("control_disqualified",
                                               "no boundary call was recorded at the replaced sites",
                                               subject=sid))
            got, bkinds, problems = _boundary_measurements(
                adapter, control, ctl, native_repeat_fn=adapter.native_site)
            measurements.extend(got)
            kinds[sid].update(bkinds)
            result.findings.extend(problems)
        del ctl
    seen = [d for d in input_digests.values() if d is not None]
    if len(seen) != len(set(seen)):
        result.findings.append(missing("duplicate_samples",
                                       "calibration samples with different ids/seeds carry "
                                       "identical inputs and cotangents"))
    result.evidence = {"samples": per_sample, "compile_graphs": dict(graphs),
                       "sample_protocol": protocol, "measurements": len(measurements),
                       # The raw control errors every threshold is derived from.
                       "control_measurements": [
                           {"source": m.source, "sample_id": m.sample_id, "result": m.result,
                            "kind": m.kind, "E2": m.E2, "Emax": m.Emax} for m in measurements],
                       "seconds": time.perf_counter() - started}
    if result.findings or contract is None:
        if contract is None and not result.findings:
            result.findings.append(missing("no_reference", "no calibration sample produced a reference"))
        return result
    try:
        union: dict[str, str] = {}
        for table in kinds.values():
            union.update(table)
        table = derive_thresholds(measurements, results=union, gamma=gamma, floors=floors)
        if protocol == MATCHED:
            # Pooled over the matched samples; the per-sample layout says which
            # samples the policy judges (each by its content identity).
            by_sample = {sid: table for sid in kinds}
            pooled = None
        else:
            pooled, by_sample = table, None
    except CalibrationError as exc:
        result.findings.append(missing("insufficient_calibration", str(exc)))
        return result
    result.policy = freeze_policy(
        adapter, patch_set=patch_set, contract=contract, samples=samples,
        thresholds=pooled, thresholds_by_sample=by_sample, gamma=gamma, floors=floors,
        boundaries=boundaries,
        controls={"native_repeats": native_repeats, "compile": control.describe(),
                  "compile_graphs": dict(graphs)},
        input_digests=input_digests, weight_digests=sorted(weight_digests),
        sample_identities=identities, executions=executions)
    return result


def _same_content(reference: Execution, control: Execution, source: str) -> list[Finding]:
    """A control is evidence about a sample only if it ran on that sample's content."""
    if control.error or not reference.sample_identity or not control.sample_identity:
        return []
    differ = identity_differences(reference.sample_identity, control.sample_identity)
    if differ:
        return [missing("control_disqualified", f"{source} ran on different content than the "
                        f"native reference: {differ}", subject=reference.sample.sample_id,
                        reasons=["sample_identity_mismatch"])]
    return []


def freeze_policy(adapter: ScopeAdapter, *, patch_set: Sequence[str], contract: C.ResultContract,
                  samples: Sequence[Sample], thresholds: Mapping[str, Mapping[str, Any]] | None,
                  controls: Mapping[str, Any], gamma: float, floors: Mapping[str, Any],
                  boundaries: bool, input_digests: Mapping[str, str | None] | None = None,
                  weight_digests: Sequence[str] = (),
                  thresholds_by_sample: Mapping[str, Mapping[str, Any]] | None = None,
                  sample_identities: Mapping[str, Mapping[str, Any]] | None = None,
                  executions: Sequence[str] = ()) -> Gate2Policy:
    """Bind thresholds to everything they were measured under, and freeze them.

    The sample protocol is the scope's. ``thresholds`` (pooled) for
    ``disjoint_holdout``; ``thresholds_by_sample`` and the samples' content
    ``sample_identities`` for ``matched_inputs``.

    :func:`calibrate` is the only producer of real policies. Tests use this to
    freeze explicitly labelled synthetic thresholds through the same binding.
    """
    protocol = SAMPLE_PROTOCOLS.get(adapter.scope)
    if protocol is None:
        raise CalibrationError(f"scope {adapter.scope!r} is not a Gate 2 scope")
    content: dict[str, Any] = {
        "scope": adapter.scope,
        "sample_protocol": protocol,
        "adapter": dict(adapter.identity()),
        "numerics": dict(adapter.numerics()),
        "environment": C.environment_fingerprint(adapter.device),
        "patch_set": sorted(patch_set),
        "reference": {"kind": "native_eager", "deployment": Deployment.native().describe(),
                      "implementation": adapter.reference_identity()},
        "controls": dict(controls),
        "contract": contract.to_dict(),
        "contract_digest": contract.digest(),
        "calibration_samples": [{**s.to_dict(),
                                 "input_digest": (input_digests or {}).get(s.sample_id)}
                                for s in samples],
        "sample_identities": dict(sample_identities or {}),
        "calibration_executions": list(executions),
        # One native initial state across all calibration samples binds the
        # policy to those weights; weights drawn per sample cannot be bound
        # this way (under matched_inputs each sample's identity binds its own).
        "weights": {"initial_state_digests": list(weight_digests),
                    "fixed": weight_digests[0] if len(weight_digests) == 1 else None},
        "gamma": gamma,
        "floors": dict(floors),
        "epsilon": EPSILON,
        "boundaries": bool(boundaries),
        "formula": FORMULAS[protocol],
    }
    if protocol == MATCHED:
        content["thresholds_by_sample"] = dict(thresholds_by_sample or {})
    else:
        content["thresholds"] = dict(thresholds or {})
    return Gate2Policy.build(content)
