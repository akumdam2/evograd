"""Gate 3 acceptance: one calibrated rule over weighted training-loss windows.

Calibration consumes complete paired native/native/compile training records,
never candidate distances. Verification compares every window to a single
frozen limit. Per-step losses remain available but have no separate numerical
threshold; invalid execution and nonfinite losses cannot pass.

The caller supplies a versioned protocol context, including the reference and
data recipe identities and numerical settings. Data order/run seeds may differ
between calibration and holdout, but must match within each comparison pair.
This is a finite-horizon training-behaviour gate, not an all-input proof.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from evograd.verification.common.reports import Finding, failure, jsonable, missing, summarize
from evograd.verification.gate3.metrics import compare_curves

SCHEMA = "evograd-verification-gate3-window-policy/1"
METRIC = "max_window_abs_delta"
SAFETY_FACTOR_GAMMA = 2.0
# No universal loss floor: NLL, BCE and MSE have different units/scales.
BINDING_FIELDS = ("plan", "initial_state", "optimizer", "adapter", "environment", "protocol_context")
CONTEXT_FIELDS = ("reference", "data_recipe", "patch_set", "numerics")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _payload(run: Any) -> dict[str, Any]:
    return run.to_dict() if hasattr(run, "to_dict") else dict(run)


def _binding(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = run["identity"]
    binding = {key: identity[key] for key in BINDING_FIELDS}
    context = binding["protocol_context"]
    if not isinstance(context, dict) or any(not context.get(k) for k in CONTEXT_FIELDS):
        raise ValueError(f"protocol_context requires {CONTEXT_FIELDS}")
    patches = context["patch_set"]
    if (not isinstance(patches, list) or not all(isinstance(p, str) and p for p in patches)
            or patches != sorted(set(patches))):
        raise ValueError("context patch_set must be a nonempty sorted unique list")
    if not binding["environment"] or not binding["initial_state"] or not binding["optimizer"]:
        raise ValueError("environment, initial state and optimizer identity are required")
    plan = binding["plan"]
    if plan.get("validation_enabled") is not False or plan.get("checkpoints") != []:
        raise ValueError("window-loss acceptance requires a training-only plan")
    if not identity.get("data") or not isinstance(identity.get("seed"), int):
        raise ValueError("data identity and training seed are required")
    _json(binding)
    return binding


def _sample(run: Mapping[str, Any]) -> dict[str, Any]:
    return {"seed": run["identity"]["seed"], "data_digest": _hash(run["identity"]["data"])}


def _problems(candidate: Mapping[str, Any], native: Mapping[str, Any]):
    comparison = compare_curves(candidate, native)
    findings = [f for f in comparison.findings if f.code != "acceptance_not_configured"]
    for label, run in (("native", native), ("candidate", candidate)):
        for record in run.get("findings", []):
            if record.get("code") == "acceptance_not_configured":
                continue
            note = failure if label == "candidate" and record.get("outcome") == "failed" else missing
            findings.append(note("recorded_" + label + "_finding", str(record.get("message", record))))
    return comparison.distances, findings


@dataclass(frozen=True)
class TrainingControl:
    """One sample: complete native, repeated native, and trusted compile runs."""

    native: Any
    repeated_native: Any
    compiled: Any


@dataclass(frozen=True)
class WindowLossPolicy:
    """Immutable canonical JSON; old/malformed/mismatched policies never pass."""

    text: str

    def __post_init__(self):
        data = json.loads(self.text)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("metric") != METRIC:
            raise ValueError("not a window-loss policy of the current schema")
        required = ("binding", "calibration_samples", "controls", "gamma", "floor", "threshold", "policy_hash")
        if any(k not in data for k in required):
            raise ValueError("policy fields are missing")
        if data["policy_hash"] != _hash({k: v for k, v in data.items() if k != "policy_hash"}):
            raise ValueError("policy content hash does not match")
        for name in ("gamma", "floor", "threshold"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        if data["gamma"] < 1 or data["floor"] < 0 or data["threshold"] < 0:
            raise ValueError("invalid policy budget")
        samples, controls = data["calibration_samples"], data["controls"]
        if not samples or len(samples) != len(controls):
            raise ValueError("policy requires matched calibration samples and controls")
        digests = [s["data_digest"] for s in samples]
        if len(set(digests)) != len(digests):
            raise ValueError("calibration data orders repeat")
        distances = []
        for control in controls:
            if not control.get("compile_identity"):
                raise ValueError("compile control identity missing")
            for key in ("native_repeat", "trusted_compile"):
                value = control[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("invalid calibration distance")
                distances.append(value)
        expected = data["gamma"] * max(data["floor"], *distances)
        if not math.isfinite(expected) or data["threshold"] != expected:
            raise ValueError("threshold disagrees with frozen calibration formula")
        if any(k not in data["binding"] for k in BINDING_FIELDS):
            raise ValueError("policy binding incomplete")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(_json(dict(data)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.text)

    @property
    def threshold(self) -> float:
        return self.to_dict()["threshold"]


@dataclass
class WindowCalibration:
    policy: WindowLossPolicy | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self):
        return jsonable({"status": "calibrated" if self.policy is not None else "incomplete",
                         "findings": [f.to_dict() for f in self.findings],
                         "policy": None if self.policy is None else self.policy.to_dict()})


def calibrate_window_loss(controls: Sequence[TrainingControl], *, floor: float,
                          gamma: float = SAFETY_FACTOR_GAMMA) -> WindowCalibration:
    """Freeze gamma * max(native/native, compile/native, loss floor).

    Controls are trusted evidence supplied by the experiment, whose manifest
    must record qualification of the compiled implementation. Role labels alone
    do not constitute an independent proof that a compiler is correct.
    """
    result = WindowCalibration()
    try:
        for name, value, lower in (("gamma", gamma, 1), ("floor", floor, 0)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < lower):
                raise ValueError(f"{name} must be finite and >= {lower}")
        if not controls:
            raise ValueError("at least one paired calibration sample is required")
        binding = None
        samples, evidence, seen = [], [], set()
        for group in controls:
            n, r, c = map(_payload, (group.native, group.repeated_native, group.compiled))
            if n["deployment"]["role"] != "native" or r["deployment"]["role"] != "native":
                raise ValueError("native repeat controls must be native runs")
            deployment = c["deployment"]
            if deployment["role"] != "control" or deployment.get("identity", {}).get("kind") != "torch_compile":
                raise ValueError("compiled control must be identified torch_compile, never candidate")
            this = _binding(n)
            if binding is None:
                binding = this
            if any(_binding(run) != binding for run in (n, r, c)):
                raise ValueError("calibration runs disagree on the experiment binding")
            if deployment["patch_set"] != binding["protocol_context"]["patch_set"]:
                raise ValueError("compiled patch set differs from the experiment")
            sample = _sample(n)
            if sample["data_digest"] in seen:
                raise ValueError("calibration samples reuse a training data order")
            seen.add(sample["data_digest"])
            samples.append(sample)
            row = {"compile_identity": deployment["identity"]}
            for label, other in (("native_repeat", r), ("trusted_compile", c)):
                distances, problems = _problems(other, n)
                if problems or distances is None:
                    raise ValueError(f"invalid {label} training evidence: " +
                                     ", ".join(f.code for f in problems))
                row[label] = distances[METRIC]
            evidence.append(row)
        threshold = gamma * max(floor, *(e[k] for e in evidence
                                          for k in ("native_repeat", "trusted_compile")))
        data = {"schema": SCHEMA, "metric": METRIC, "binding": binding,
                "calibration_samples": samples, "controls": evidence,
                "gamma": gamma, "floor": floor, "threshold": threshold}
        data["policy_hash"] = _hash(data)
        result.policy = WindowLossPolicy.from_dict(data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.findings.append(missing("calibration_unavailable", str(exc)))
    return result


@dataclass
class TrainingVerdict:
    findings: list[Finding] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, Any] | None = None
    distances: dict[str, Any] | None = None

    @property
    def status(self):
        return summarize(self.findings)[0]

    @property
    def complete(self):
        return summarize(self.findings)[1]

    def to_dict(self):
        return jsonable({"gate": "gate3", "protocol": SCHEMA, "status": self.status,
                         "complete": self.complete, "metric": METRIC, "policy": self.policy,
                         "distances": self.distances, "windows": self.windows,
                         "findings": [f.to_dict() for f in self.findings]})


def _is_step_policy(policy: Any) -> bool:
    from evograd.verification.gate3 import stepwise

    return isinstance(policy, stepwise.StepLossPolicy) or (
        isinstance(policy, Mapping) and policy.get("schema") == stepwise.SCHEMA)


def _is_global_policy(policy: Any) -> bool:
    from evograd.verification.gate3 import global_loss

    return isinstance(policy, global_loss.GlobalLossPolicy) or (
        isinstance(policy, Mapping) and policy.get("schema") in (global_loss.SCHEMA, global_loss.SHARED_SCHEMA))


def verify_training(candidate: Any, native: Any, *, policy: WindowLossPolicy | Mapping | None):
    """Every weighted window must meet the frozen loss-difference threshold.

    Any missing evidence or binding/sample mismatch withholds *all* numerical
    judgements. Measured differences can still be reported. Execution failures
    remain failures. Validation-loss and per-step difference gates are absent.

    Dispatch is by explicit policy schema, and nothing is reinterpreted:

    * ``gate3.global_loss`` schema -> :func:`gate3.global_loss.verify_global_loss`
      (whole-horizon RMS and max of the loss difference; returns a ``GlobalVerdict``);
    * ``gate3.stepwise`` schema -> :func:`gate3.stepwise.verify_step_loss`
      (one limit per step; returns a ``StepVerdict``);
    * anything else -> the historical window rule below.
    """
    if _is_global_policy(policy):
        from evograd.verification.gate3.global_loss import verify_global_loss

        return verify_global_loss(candidate, native, policy=policy)
    if _is_step_policy(policy):
        from evograd.verification.gate3.stepwise import verify_step_loss

        return verify_step_loss(candidate, native, policy=policy)
    verdict = TrainingVerdict()
    try:
        c, n = _payload(candidate), _payload(native)
        distances, problems = _problems(c, n)
        verdict.distances = distances
        verdict.findings.extend(problems)
        if c.get("deployment", {}).get("role") not in ("candidate", "control", "native"):
            verdict.findings.append(missing("invalid_deployment", "unknown tested deployment role"))
        binding = _binding(n)
        if _binding(c) != binding:
            verdict.findings.append(missing("run_binding_mismatch", "paired runs use different conditions"))
        if policy is None:
            verdict.findings.append(missing("no_policy", "no frozen window-loss policy"))
            return verdict
        if not isinstance(policy, WindowLossPolicy):
            policy = WindowLossPolicy.from_dict(policy)
        data = policy.to_dict()
        verdict.policy = {"policy_hash": data["policy_hash"], "threshold": data["threshold"],
                          "gamma": data["gamma"], "floor": data["floor"]}
        if data["binding"] != binding:
            verdict.findings.append(missing("policy_binding_mismatch", "policy does not describe this run"))
        if (c["deployment"]["role"] != "native"
                and c["deployment"].get("patch_set") != binding["protocol_context"]["patch_set"]):
            verdict.findings.append(missing("patch_set_mismatch", "tested patch set differs from policy"))
        used = {s["data_digest"] for s in data["calibration_samples"]}
        if _sample(n)["data_digest"] in used:
            verdict.findings.append(missing("calibration_overlap", "validation reuses a calibration data order"))
        if verdict.findings or distances is None:
            return verdict
        for record, delta in zip(n["train_windows"], distances["window_abs_deltas"]):
            passed = delta <= policy.threshold
            verdict.windows.append({"window": record["window"], "steps": record["steps"],
                                    "abs_delta": delta, "threshold": policy.threshold,
                                    "status": "passed" if passed else "failed"})
            if not passed:
                verdict.findings.append(failure("window_loss_exceeded",
                                                f"window {record['window']}: {delta} > {policy.threshold}",
                                                subject=f"window:{record['window']}"))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        verdict.findings.append(missing("invalid_verification_evidence", str(exc)))
    return verdict
