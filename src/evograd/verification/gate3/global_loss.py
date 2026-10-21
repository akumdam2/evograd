"""Gate 3 global-loss acceptance: whole-horizon RMS and maximum of the loss difference.

For a fixed workload (one exact batch sequence and training seed), with
``ell[p, t]`` the scalar objective run ``p`` differentiated at step ``t``
(recorded before update ``t``, step-record field ``objective``):

    d[t] = ell[test, t] - ell[reference, t]          (same step first)
    R    = sqrt(sum_t d[t]^2 / N)
    M    = max_t |d[t]|

Calibration computes R and M for every trusted control trajectory -- each
native repeat against its native, each compile control against its native --
and freezes, per metric,

    T_R = gamma * max(max_c R_repeat[c], max_c R_compile[c], floor_R)
    T_M = gamma * max(max_c M_repeat[c], max_c M_compile[c], floor_M)

A test passes the loss check only if ``R <= T_R`` AND ``M <= T_M`` (equality
passes). There is no window, mean-of-losses difference, permitted violation
count, excluded step or averaging over repetitions; each test trajectory is
judged on its own. Candidate and fault runs never enter the budget.

The policy binds the stepwise binding (plan, initial state, optimizer,
adapter, environment, protocol context, exact data order, seed, evaluator):
the same batches are reused on purpose, and a mismatch withholds every
numerical judgement. A tested run that was a calibration run is not
independent evidence. Raw-step validation, the binding and run identity are
the stepwise module's own (:mod:`evograd.verification.gate3.stepwise`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from evograd.verification.common.reports import Finding, failure, jsonable, missing, summarize
from evograd.verification.gate3.policy import SAFETY_FACTOR_GAMMA, TrainingControl, _hash, _json, _payload
from evograd.verification.gate3.stepwise import objectives, run_digest, step_binding

SCHEMA = "evograd-verification-gate3-global-loss-policy/1"
VERDICT = "evograd-verification-gate3-global-loss/1"
METRICS = ("rms", "max_abs")
FORMULA = ("d[t] = objective_test[t] - objective_reference[t]; R = sqrt(sum_t d[t]^2 / N); "
           "M = max_t |d[t]|; T_m = gamma * max(max_c m(native_b, native_a), "
           "max_c m(compile, native_a), floor_m) for m in (R, M); pass iff R <= T_R and M <= T_M")
REQUIRED = ("schema", "metrics", "steps", "binding", "calibration_runs", "controls", "gamma",
            "floors", "thresholds", "formula", "policy_hash")


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def loss_metrics(test: Sequence[float], reference: Sequence[float]) -> dict[str, Any]:
    """``R``, ``M`` and where ``M`` occurs, from the same-step differences.

    RMS is computed scaled by ``M`` so finite large differences cannot
    overflow into a wrong verdict; a difference that itself overflows is
    reported unbounded (``inf``) and can only fail.
    """
    if len(test) != len(reference) or not test:
        raise ValueError(f"trajectories of {len(test)} and {len(reference)} steps")
    deltas = [float(a) - float(b) for a, b in zip(test, reference)]
    magnitudes = [abs(d) for d in deltas]
    peak = max(magnitudes)
    step = 1 + magnitudes.index(peak)
    if not math.isfinite(peak):
        return {"rms": math.inf, "max_abs": math.inf, "max_abs_step": step, "steps": len(deltas),
                "overflow": True}
    rms = 0.0 if peak == 0 else peak * math.sqrt(
        math.fsum((d / peak) ** 2 for d in deltas) / len(deltas))
    return {"rms": rms, "max_abs": peak, "max_abs_step": step, "steps": len(deltas),
            "overflow": False}


@dataclass(frozen=True)
class GlobalLossPolicy:
    """Immutable canonical JSON of the two frozen whole-horizon limits."""

    text: str

    def __post_init__(self):
        data = json.loads(self.text)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("not a global-loss policy of the current schema")
        if any(k not in data for k in REQUIRED):
            raise ValueError("policy fields are missing")
        if data["policy_hash"] != _hash({k: v for k, v in data.items() if k != "policy_hash"}):
            raise ValueError("policy content hash does not match")
        if data["formula"] != FORMULA or list(data["metrics"]) != list(METRICS):
            raise ValueError("unknown formula or metrics")
        steps, gamma, floors, thresholds = (data["steps"], data["gamma"], data["floors"],
                                            data["thresholds"])
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("invalid step count")
        if not _number(gamma) or gamma < 1:
            raise ValueError("gamma must be finite and >= 1")
        if not isinstance(floors, dict) or set(floors) != set(METRICS) \
                or not all(_number(floors[m]) and floors[m] > 0 for m in METRICS):
            raise ValueError("floors must be finite and > 0 for both metrics")
        controls, runs = data["controls"], data["calibration_runs"]
        if not controls or not isinstance(controls, list) or len(runs) != len(controls):
            raise ValueError("policy requires calibration controls and their runs")
        worst = {m: floors[m] for m in METRICS}
        for control in controls:
            if not control.get("compile_identity"):
                raise ValueError("compile control identity missing")
            for source in ("native_repeat", "trusted_compile"):
                entry = control.get(source)
                if not isinstance(entry, dict):
                    raise ValueError(f"missing {source} metrics")
                for m in METRICS:
                    value = entry.get(m)
                    if not _number(value) or value < 0:
                        raise ValueError(f"invalid {source} {m}")
                    worst[m] = max(worst[m], value)
        if not isinstance(thresholds, dict) or set(thresholds) != set(METRICS):
            raise ValueError("thresholds must name both metrics")
        for m in METRICS:
            if not _number(thresholds[m]) or thresholds[m] != gamma * worst[m]:
                raise ValueError(f"threshold {m} disagrees with the frozen formula")
        digests = [d for row in runs for d in row.values()]
        if len(set(digests)) != len(digests) or len(digests) != 3 * len(runs):
            raise ValueError("calibration runs must be three distinct executions per control")
        binding = data["binding"]
        if not isinstance(binding, dict) or any(k not in binding for k in (
                "plan", "initial_state", "optimizer", "adapter", "environment",
                "protocol_context", "data", "seed", "evaluator")):
            raise ValueError("policy binding incomplete")
        if binding["plan"].get("steps") != steps:
            raise ValueError("the bound plan has another step count")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(_json(dict(data)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.text)

    @property
    def thresholds(self) -> dict[str, float]:
        return self.to_dict()["thresholds"]


@dataclass
class GlobalCalibration:
    policy: GlobalLossPolicy | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self):
        return jsonable({"status": "calibrated" if self.policy is not None else "incomplete",
                         "findings": [f.to_dict() for f in self.findings],
                         "policy": None if self.policy is None else self.policy.to_dict()})


def calibrate_global_loss(controls: Sequence[TrainingControl], *, floor_rms: float,
                          floor_max: float, gamma: float = SAFETY_FACTOR_GAMMA) -> GlobalCalibration:
    """Freeze ``T_R``/``T_M`` from complete native/native/compile trajectories.

    Each control's metrics are computed on its own whole trajectory first,
    then the maximum is taken across controls. Every run must share the full
    binding (same data order and seed); the compiled control must be an
    identified ``torch_compile`` deployment at the experiment's patch set.
    """
    result = GlobalCalibration()
    try:
        if not _number(gamma) or gamma < 1:
            raise ValueError("gamma must be finite and >= 1")
        for name, value in (("floor_rms", floor_rms), ("floor_max", floor_max)):
            if not _number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not controls:
            raise ValueError("at least one native/native/compile control group is required")
        binding, steps = None, None
        runs, evidence = [], []
        for group in controls:
            n, r, c = map(_payload, (group.native, group.repeated_native, group.compiled))
            if n["deployment"]["role"] != "native" or r["deployment"]["role"] != "native":
                raise ValueError("native repeat controls must be native runs")
            deployment = c["deployment"]
            if deployment["role"] != "control" or \
                    deployment.get("identity", {}).get("kind") != "torch_compile":
                raise ValueError("compiled control must be identified torch_compile, never a "
                                 "candidate or fault")
            this = step_binding(n)
            binding = binding if binding is not None else this
            if any(step_binding(run) != binding for run in (n, r, c)):
                raise ValueError("calibration runs disagree on the workload binding")
            if deployment["patch_set"] != binding["protocol_context"]["patch_set"]:
                raise ValueError("compiled patch set differs from the experiment")
            series = {}
            for label, run in (("native_a", n), ("native_b", r), ("compile", c)):
                values, problems = objectives(run, label=label, reference=True)
                if problems or values is None:
                    raise ValueError(f"invalid {label} calibration record: " +
                                     ", ".join(f"{f.code} ({f.message})" for f in problems))
                series[label] = values
            steps = len(series["native_a"])
            digests = [run_digest(run) for run in (n, r, c)]
            if len(set(digests)) != 3:
                raise ValueError("a calibration execution was supplied twice")
            runs.append({"native_a": digests[0], "native_b": digests[1], "compile": digests[2]})
            repeat = loss_metrics(series["native_b"], series["native_a"])
            compiled = loss_metrics(series["compile"], series["native_a"])
            if repeat["overflow"] or compiled["overflow"]:
                raise ValueError("a control difference overflowed")
            evidence.append({"compile_identity": deployment["identity"],
                             "native_repeat_bitwise": series["native_b"] == series["native_a"],
                             "native_repeat": {m: repeat[m] for m in METRICS}
                             | {"max_abs_step": repeat["max_abs_step"]},
                             "trusted_compile": {m: compiled[m] for m in METRICS}
                             | {"max_abs_step": compiled["max_abs_step"]}})
        flat = [d for row in runs for d in row.values()]
        if len(set(flat)) != len(flat):
            raise ValueError("calibration groups share an execution")
        floors = {"rms": float(floor_rms), "max_abs": float(floor_max)}
        thresholds = {m: gamma * max(floors[m], *(e[s][m] for e in evidence
                                                   for s in ("native_repeat", "trusted_compile")))
                      for m in METRICS}
        data = {"schema": SCHEMA, "metrics": list(METRICS), "steps": steps, "binding": binding,
                "calibration_runs": runs, "controls": evidence, "gamma": gamma,
                "floors": floors, "thresholds": thresholds, "formula": FORMULA,
                "objective": "step record field 'objective' (scalar the backward differentiated)"}
        data["policy_hash"] = _hash(data)
        result.policy = GlobalLossPolicy.from_dict(data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.findings.append(missing("calibration_unavailable", str(exc)))
    return result


@dataclass
class GlobalVerdict:
    findings: list[Finding] = field(default_factory=list)
    policy: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return summarize(self.findings)[0]

    @property
    def complete(self) -> bool:
        return summarize(self.findings)[1]

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"gate": "gate3", "protocol": VERDICT, "status": self.status,
                         "complete": self.complete, "metrics": list(METRICS),
                         "policy": self.policy, "summary": self.summary,
                         "findings": [f.to_dict() for f in self.findings]})


def verify_global_loss(candidate: Any, native: Any, *, policy: GlobalLossPolicy | Mapping | None
                       ) -> GlobalVerdict:
    """``R <= T_R`` and ``M <= T_M`` over the whole horizon, one test at a time.

    A missing, foreign (window/stepwise) or mismatched policy, invalid records
    or a tested run that was a calibration execution withhold the numerical
    judgement; the metrics are still measured and reported, never judged.
    """
    verdict = GlobalVerdict()
    try:
        c, n = _payload(candidate), _payload(native)
        if n.get("deployment", {}).get("role") != "native":
            verdict.findings.append(missing("not_native_reference", "the reference is not a native run"))
        if c.get("deployment", {}).get("role") not in ("candidate", "control", "native"):
            verdict.findings.append(missing("invalid_deployment", "unknown tested deployment role"))
        test_values, problems = objectives(c, label="candidate", reference=False)
        verdict.findings.extend(problems)
        ref_values, problems = objectives(n, label="native", reference=True)
        verdict.findings.extend(problems)
        binding = step_binding(n)
        if step_binding(c) != binding:
            verdict.findings.append(missing("run_binding_mismatch",
                                            "the tested and reference runs differ in workload binding"))
        measured = None
        if test_values is not None and ref_values is not None and len(test_values) == len(ref_values):
            measured = loss_metrics(test_values, ref_values)
            verdict.summary.update(measured)
        if policy is None:
            verdict.findings.append(missing("no_policy", "no frozen global-loss policy"))
        elif not isinstance(policy, GlobalLossPolicy):
            data = dict(policy)
            if data.get("schema") != SCHEMA:
                verdict.findings.append(missing(
                    "wrong_policy_schema", f"{data.get('schema')!r} is not {SCHEMA!r}; a window or "
                    "stepwise policy is never reinterpreted as global limits"))
                policy = None
            else:
                policy = GlobalLossPolicy.from_dict(data)
        if policy is not None:
            data = policy.to_dict()
            verdict.policy = {"policy_hash": data["policy_hash"], "gamma": data["gamma"],
                              "floors": data["floors"], "thresholds": data["thresholds"],
                              "steps": data["steps"]}
            if data["binding"] != binding:
                differ = sorted(k for k in set(data["binding"]) | set(binding)
                                if data["binding"].get(k) != binding.get(k))
                verdict.findings.append(missing("policy_binding_mismatch",
                                                f"the policy does not describe this run: {differ}"))
            if c["deployment"]["role"] != "native" and \
                    c["deployment"].get("patch_set") != binding["protocol_context"]["patch_set"]:
                verdict.findings.append(missing("patch_set_mismatch", "tested patch set differs"))
            used = {d for row in data["calibration_runs"] for d in row.values()}
            if run_digest(c) in used:
                verdict.findings.append(missing("tested_run_used_in_calibration",
                                                "a calibration execution is not independent evidence"))
            if measured is not None and measured["steps"] != data["steps"]:
                verdict.findings.append(missing("step_count_mismatch", "policy and runs differ"))
        if verdict.findings or policy is None or measured is None:
            verdict.summary["judged"] = False
            return verdict
        limits = policy.thresholds
        ratios = {m: measured[m] / limits[m] for m in METRICS}
        passed = {m: measured[m] <= limits[m] for m in METRICS}
        verdict.summary.update({"judged": True, "thresholds": limits, "ratios": ratios,
                                "passed": passed})
        failing = [m for m in METRICS if not passed[m]]
        if failing:
            verdict.findings.append(failure(
                "global_loss_exceeded",
                "; ".join(f"{m} {measured[m]!r} > {limits[m]!r}" for m in failing),
                subject="whole_horizon", metrics=failing))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        verdict.findings.append(missing("invalid_verification_evidence", str(exc)))
        verdict.summary["judged"] = False
    return verdict
