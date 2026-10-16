"""Gate 3 stepwise acceptance: one frozen loss limit per training step.

For a fixed workload (one exact batch sequence, one training seed), every
control and test replays the same steps. With ``ell[p, t]`` the scalar
objective that run ``p`` differentiated at step ``t`` (before update ``t``,
field ``objective`` of the step record):

    repeat[t]  = max_c |ell[native_a, c, t] - ell[native_b, c, t]|
    compile[t] = max_c |ell[compile,  c, t] - ell[native_a, c, t]|
    T[t]       = gamma * max(repeat[t], compile[t], floor)

and a test passes only if ``|ell[test, t] - ell[reference, t]| <= T[t]`` at
every step (equality passes). There is no window, mean, percentile, permitted
violation count or excluded region. Calibration sets an error budget from
trusted controls; candidate values never enter it, and it fits nothing.

Unlike the window rule, the policy binds the exact data order and training
seed: verification reuses the calibration batches by design, and a different
order, reference, setting or initial state withholds every numerical
judgement. A test run that was itself a calibration run is not independent
evidence and is reported incomplete.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from evograd.verification.common.reports import Finding, failure, jsonable, missing, summarize
from evograd.verification.gate3.policy import (
    SAFETY_FACTOR_GAMMA,
    TrainingControl,
    _binding,
    _hash,
    _json,
    _payload,
)

SCHEMA = "evograd-verification-gate3-step-loss-policy/1"
METRIC = "per_step_abs_objective_delta"
VERDICT = "evograd-verification-gate3-step-loss/1"
#: The step-record field compared, as the measurement runner defines it.
OBJECTIVE_FIELD = "objective"
REQUIRED = ("schema", "metric", "steps", "binding", "calibration_runs", "controls", "gamma",
            "floor", "thresholds", "formula", "policy_hash")
FORMULA = "T[t] = gamma * max(max_c |native_a - native_b|[t], max_c |compile - native_a|[t], floor)"


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def step_binding(run: Mapping[str, Any]) -> dict[str, Any]:
    """The window-policy binding plus the exact data sequence, seed and evaluator."""
    binding = _binding(run)
    identity = run["identity"]
    binding.update({"data": identity["data"], "seed": identity["seed"],
                    "evaluator": identity.get("evaluator")})
    _json(binding)
    return binding


def run_digest(run: Mapping[str, Any]) -> str:
    """Identity of one measured execution: its ``run_id`` and its content.

    Deterministic repeats can be bitwise equal in every record, so content
    alone cannot tell a fresh execution from the same one supplied twice; the
    runner's per-execution ``run_id`` does. (Non-finite values are hashed as
    such; they are judged elsewhere.)
    """
    content = {"run_id": run.get("run_id"), "deployment": run.get("deployment"),
               "identity": run.get("identity"), "train_steps": run.get("train_steps")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=True, default=str).encode()).hexdigest()


def objectives(run: Mapping[str, Any], *, label: str, reference: bool
               ) -> tuple[list[float] | None, list[Finding]]:
    """Every step's objective, after checking the raw records are complete.

    Missing, truncated, reordered, inconsistent or non-numeric records leave
    the check incomplete; a non-finite objective is a failure of the tested
    run and an invalid reference otherwise.
    """
    side = missing if reference else failure
    findings: list[Finding] = []
    if not run.get("completed"):
        findings.append(side(f"{label}_incomplete", f"the {label} run did not complete",
                             reasons=[f.get("code") for f in run.get("findings", [])]))
    for record in run.get("findings", []):
        note = failure if not reference and record.get("outcome") == "failed" else missing
        findings.append(note(f"recorded_{label}_finding", str(record.get("message", record))))
    try:
        steps = int(run["identity"]["plan"]["steps"])
        raw = run["train_steps"]
        if len(raw) != steps:
            return None, findings + [missing("incomplete_step_records",
                                             f"{label}: {len(raw)} of {steps} steps recorded")]
        values = []
        for index, row in enumerate(raw, 1):
            if isinstance(row.get("step"), bool) or row.get("step") != index:
                return None, findings + [missing("step_misaligned",
                                                 f"{label}: record {index} is step {row.get('step')}")]
            if OBJECTIVE_FIELD not in row:
                return None, findings + [missing("objective_not_recorded",
                                                 f"{label}: step {index} has no {OBJECTIVE_FIELD!r} "
                                                 "(a record from before the stepwise measurement)")]
            value, total, weight = row[OBJECTIVE_FIELD], row.get("loss_sum"), row.get("weight")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (value, total, weight)):
                return None, findings + [missing("invalid_step_record", f"{label}: step {index}")]
            if not (math.isfinite(value) and math.isfinite(total) and math.isfinite(weight)) \
                    or weight <= 0:
                return None, findings + [side("non_finite_step_record",
                                              f"{label}: step {index} objective {value}")]
            # The objective is the float32 quotient the backward used; it agrees
            # with the float64 quotient of the stored fields to float32 rounding.
            if not math.isclose(value, total / weight, rel_tol=1e-6, abs_tol=1e-12):
                return None, findings + [missing("inconsistent_step_record",
                                                 f"{label}: step {index} objective {value} vs "
                                                 f"loss_sum/weight {total / weight}")]
            values.append(float(value))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return None, findings + [missing("invalid_step_record", f"{label}: {exc}")]
    return values, findings


@dataclass(frozen=True)
class StepLossPolicy:
    """Immutable canonical JSON of a calibrated step-limit vector."""

    text: str

    def __post_init__(self):
        data = json.loads(self.text)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("metric") != METRIC:
            raise ValueError("not a stepwise loss policy of the current schema")
        if any(k not in data for k in REQUIRED):
            raise ValueError("policy fields are missing")
        if data["policy_hash"] != _hash({k: v for k, v in data.items() if k != "policy_hash"}):
            raise ValueError("policy content hash does not match")
        if data["formula"] != FORMULA:
            raise ValueError("unknown threshold formula")
        steps, gamma, floor = data["steps"], data["gamma"], data["floor"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("invalid step count")
        if not (_number(gamma) and _number(floor)) or gamma < 1 or floor <= 0:
            raise ValueError("gamma must be finite >= 1 and the floor finite > 0")
        thresholds = data["thresholds"]
        controls = data["controls"]
        if not controls or len(data["calibration_runs"]) != len(controls):
            raise ValueError("policy requires calibration controls and their runs")
        repeat = [0.0] * steps
        compiled = [0.0] * steps
        for control in controls:
            if not control.get("compile_identity"):
                raise ValueError("compile control identity missing")
            for key, into in (("native_repeat", repeat), ("trusted_compile", compiled)):
                vector = control.get(key)
                if not isinstance(vector, list) or len(vector) != steps \
                        or not all(_number(v) and v >= 0 for v in vector):
                    raise ValueError(f"invalid {key} error vector")
                for t, v in enumerate(vector):
                    into[t] = max(into[t], v)
        if not isinstance(thresholds, list) or len(thresholds) != steps:
            raise ValueError("threshold vector length differs from the step count")
        for t in range(steps):
            expected = gamma * max(repeat[t], compiled[t], floor)
            if not _number(thresholds[t]) or thresholds[t] != expected:
                raise ValueError(f"threshold at step {t + 1} disagrees with the frozen formula")
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
    def thresholds(self) -> list[float]:
        return self.to_dict()["thresholds"]


@dataclass
class StepCalibration:
    policy: StepLossPolicy | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self):
        return jsonable({"status": "calibrated" if self.policy is not None else "incomplete",
                         "findings": [f.to_dict() for f in self.findings],
                         "policy": None if self.policy is None else self.policy.to_dict()})


def calibrate_step_loss(controls: Sequence[TrainingControl], *, floor: float,
                        gamma: float = SAFETY_FACTOR_GAMMA) -> StepCalibration:
    """Freeze ``T[t]`` from complete native/native/compile runs of one workload.

    Every control run must share the full binding, including the exact data
    order and seed. The compiled control must be an identified
    ``torch_compile`` deployment at the experiment's patch set; its
    qualification remains the experiment's evidence to supply.
    """
    result = StepCalibration()
    try:
        for name, value, lower in (("gamma", gamma, 1), ("floor", floor, 0)):
            if not _number(value) or value < lower or (name == "floor" and value <= 0):
                raise ValueError(f"{name} must be finite and {'>' if name == 'floor' else '>='} {lower}")
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
                raise ValueError("compiled control must be identified torch_compile, never a candidate")
            this = step_binding(n)
            binding = binding if binding is not None else this
            if any(step_binding(run) != binding for run in (n, r, c)):
                raise ValueError("calibration runs disagree on the workload binding "
                                 "(data order, seed, plan, initial state, settings)")
            if deployment["patch_set"] != binding["protocol_context"]["patch_set"]:
                raise ValueError("compiled patch set differs from the experiment")
            values = {}
            for label, run in (("native_a", n), ("native_b", r), ("compile", c)):
                series, problems = objectives(run, label=label, reference=True)
                if problems or series is None:
                    raise ValueError(f"invalid {label} calibration record: " +
                                     ", ".join(f"{f.code} ({f.message})" for f in problems))
                values[label] = series
            steps = len(values["native_a"])
            digests = [run_digest(run) for run in (n, r, c)]
            if len(set(digests)) != 3:
                raise ValueError("a calibration run was supplied twice")
            runs.append({"native_a": digests[0], "native_b": digests[1], "compile": digests[2]})
            evidence.append({
                "compile_identity": deployment["identity"],
                "native_repeat": [abs(a - b) for a, b in zip(values["native_a"], values["native_b"])],
                "trusted_compile": [abs(c_ - a) for a, c_ in zip(values["native_a"], values["compile"])]})
        flat = [d for row in runs for d in row.values()]
        if len(set(flat)) != len(flat):
            raise ValueError("calibration groups share a run")
        thresholds = [gamma * max(max(e["native_repeat"][t] for e in evidence),
                                  max(e["trusted_compile"][t] for e in evidence), floor)
                      for t in range(steps)]
        data = {"schema": SCHEMA, "metric": METRIC, "steps": steps, "binding": binding,
                "calibration_runs": runs, "controls": evidence, "gamma": gamma,
                "floor": floor, "thresholds": thresholds, "formula": FORMULA,
                "objective": "step record field 'objective' (scalar the backward differentiated)"}
        data["policy_hash"] = _hash(data)
        result.policy = StepLossPolicy.from_dict(data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.findings.append(missing("calibration_unavailable", str(exc)))
    return result


@dataclass
class StepVerdict:
    findings: list[Finding] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
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
                         "complete": self.complete, "metric": METRIC, "policy": self.policy,
                         "summary": self.summary, "steps": self.steps,
                         "findings": [f.to_dict() for f in self.findings]})


def verify_step_loss(candidate: Any, native: Any, *, policy: StepLossPolicy | Mapping | None
                     ) -> StepVerdict:
    """Every step's objective difference against its own frozen limit.

    A missing, foreign (e.g. window) or mismatched policy, invalid records or a
    tested run that was used for calibration withhold every numerical
    judgement: the errors are still measured and reported, never judged.
    """
    verdict = StepVerdict()
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
        errors = None
        if test_values is not None and ref_values is not None and len(test_values) == len(ref_values):
            errors = [abs(a - b) for a, b in zip(test_values, ref_values)]
        if policy is None:
            verdict.findings.append(missing("no_policy", "no frozen stepwise policy"))
        else:
            if not isinstance(policy, StepLossPolicy):
                data = dict(policy)
                if data.get("schema") != SCHEMA:
                    verdict.findings.append(missing(
                        "wrong_policy_schema", f"{data.get('schema')!r} is not {SCHEMA!r}; a window "
                        "policy is never reinterpreted as step limits"))
                    policy = None
                else:
                    policy = StepLossPolicy.from_dict(data)
        if policy is not None:
            data = policy.to_dict()
            verdict.policy = {"policy_hash": data["policy_hash"], "gamma": data["gamma"],
                              "floor": data["floor"], "steps": data["steps"]}
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
                                                "a calibration run is not independent evidence"))
            if errors is not None and len(errors) != data["steps"]:
                verdict.findings.append(missing("step_count_mismatch", "policy and runs differ"))
        if errors is not None:
            verdict.summary = {"steps": len(errors), "max_abs_error": max(errors),
                               "max_abs_error_step": 1 + errors.index(max(errors))}
        if verdict.findings or policy is None or errors is None:
            verdict.summary["judged"] = False
            verdict.summary["measured_errors"] = errors
            return verdict
        thresholds = policy.thresholds
        violations = []
        for t, (error, limit) in enumerate(zip(errors, thresholds), 1):
            passed = error <= limit
            verdict.steps.append({"step": t, "abs_error": error, "threshold": limit,
                                  "ratio": error / limit, "status": "passed" if passed else "failed"})
            if not passed:
                violations.append(t)
        ratios = [row["ratio"] for row in verdict.steps]
        worst = max(range(len(ratios)), key=ratios.__getitem__)
        verdict.summary.update({"judged": True, "violations": len(violations),
                                "first_failing_step": violations[0] if violations else None,
                                "failing_steps": violations, "max_ratio": ratios[worst],
                                "max_ratio_step": worst + 1})
        if violations:
            verdict.findings.append(failure(
                "step_loss_exceeded", f"{len(violations)} of {len(errors)} steps exceed their limit; "
                f"first at step {violations[0]}", subject=f"step:{violations[0]}",
                steps=violations[:100]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        verdict.findings.append(missing("invalid_verification_evidence", str(exc)))
        verdict.summary["judged"] = False
        verdict.steps = []
    return verdict
