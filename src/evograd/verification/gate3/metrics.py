"""Gate 3 measurement arithmetic: the plan, weighted windows, curve distances.

This module computes distances; ``policy.verify_training`` applies a frozen
training-window policy. The distances are

    D_train = max_w |mean_loss_candidate(w) - mean_loss_native(w)|
    D_val   = max_k |validation_loss_candidate(k) - validation_loss_native(k)|

over fixed training windows ``w`` and checkpoints ``k``. A window mean is a
weighted mean -- the sum of its steps' loss sums over the sum of their weights
(tokens or samples) -- never a mean of per-step means. Curves are compared only
when their plans, run seeds, data, initial state and optimizer configuration
are identical and every window and checkpoint lines up; otherwise no distance is
reported. A comparison without a policy is diagnostic, never ``passed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from evograd.verification.common.reports import Finding, failure, jsonable, missing

ACCEPTANCE_NOT_CONFIGURED = (
    "This is a training measurement, not an acceptance verdict. Use "
    "gate3.verify_training with a native reference and a frozen window-loss policy.")


@dataclass(frozen=True)
class TrainingPlan:
    """Steps, window length and checkpoints. Part of every measurement's identity.

    Nothing defaults: a 5-step smoke run and a 1000-step study use the same
    core and differ only in this object.

    ``validation=False`` is a training-only measurement: no evaluator is built
    and no held-out data is read, so it has no checkpoints.
    """

    steps: int
    window: int
    checkpoints: tuple[int, ...]
    validation: bool = True
    #: Training batches per update. Each step differentiates the mean of the
    #: objectives of ``accumulation`` consecutive batches (each backward is
    #: scaled by ``1/accumulation``), so a model whose batch must keep its
    #: shape can still take each update from more data.
    accumulation: int = 1

    def __post_init__(self) -> None:
        for name in ("steps", "window", "accumulation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.steps % self.window:
            raise ValueError(f"steps {self.steps} is not a multiple of window {self.window}")
        if not isinstance(self.validation, bool):
            raise ValueError(f"validation must be a bool, got {self.validation!r}")
        checkpoints = tuple(self.checkpoints)
        if not self.validation and checkpoints:
            raise ValueError(f"a training-only plan has no validation checkpoints, got {checkpoints}")
        if list(checkpoints) != sorted(set(checkpoints)):
            raise ValueError(f"checkpoints must be strictly increasing, got {checkpoints}")
        bad = [c for c in checkpoints if isinstance(c, bool) or not isinstance(c, int)
               or c < 0 or c > self.steps]
        if bad:
            raise ValueError(f"checkpoints {bad} fall outside 0..{self.steps}")
        object.__setattr__(self, "checkpoints", checkpoints)

    @property
    def windows(self) -> int:
        return self.steps // self.window

    def to_dict(self) -> dict[str, Any]:
        plan = {"steps": self.steps, "window": self.window, "checkpoints": list(self.checkpoints),
                "validation_enabled": self.validation}
        if self.accumulation != 1:      # only when set, so a plain plan keeps its identity
            plan["accumulation"] = self.accumulation
        return plan


class WindowAccumulator:
    """Per-step ``(loss_sum, weight)`` into fixed windows of the plan."""

    def __init__(self, plan: TrainingPlan):
        self.plan = plan
        self.windows: list[dict[str, Any]] = []
        self._sum = 0.0
        self._weight = 0.0

    def add(self, step: int, loss_sum: float, weight: float) -> None:
        if not (math.isfinite(loss_sum) and math.isfinite(weight)) or weight <= 0:
            raise ValueError(f"step {step}: loss_sum={loss_sum}, weight={weight}")
        self._sum += float(loss_sum)
        self._weight += float(weight)
        if step % self.plan.window == 0:
            index = step // self.plan.window
            self.windows.append({"window": index,
                                 "steps": [step - self.plan.window + 1, step],
                                 "loss_sum": self._sum, "weight": self._weight,
                                 "mean_loss": self._sum / self._weight})
            self._sum, self._weight = 0.0, 0.0


@dataclass
class CurveComparison:
    findings: list[Finding]
    distances: dict[str, Any] | None

    @property
    def status(self) -> str:
        # Measurement is the most this phase can conclude.
        return "incomplete"

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"status": self.status,
                         "acceptance": {"configured": False, "reason": ACCEPTANCE_NOT_CONFIGURED},
                         "findings": [f.to_dict() for f in self.findings],
                         "distances": self.distances})


#: Identity fields two measurements must share before their curves mean anything.
ALIGNED_FIELDS = ("plan", "seed", "data", "initial_state", "optimizer", "evaluator", "adapter")


def training_record_problems(run: Mapping[str, Any], *, reference: bool) -> list[Finding]:
    """Validate complete raw steps and their weighted windows, not just summaries.

    An empty, truncated, nonfinite or edited curve cannot establish agreement.
    Invalid numeric candidate results fail; missing records make checking incomplete.
    """
    side = missing if reference else failure
    label = "native" if reference else "candidate"
    try:
        config = run["identity"]["plan"]
        plan = TrainingPlan(steps=config["steps"], window=config["window"],
                            checkpoints=tuple(config["checkpoints"]),
                            validation=config["validation_enabled"])
        raw = run["train_steps"]
        stored = run["train_windows"]
        if len(stored) != plan.windows:
            return [missing("window_count_mismatch", f"{label}: expected {plan.windows} windows")]
        if len(raw) != plan.steps:
            return [missing("incomplete_training_records",
                            f"{label}: expected {plan.steps} steps and {plan.windows} windows")]
        accumulator = WindowAccumulator(plan)
        for index, row in enumerate(raw, 1):
            if isinstance(row["step"], bool) or row["step"] != index:
                return [missing("step_misaligned", f"{label}: expected step {index}")]
            values = [row[key] for key in ("loss_sum", "weight", "mean_loss")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                return [missing("invalid_training_record", f"{label}: nonnumeric step {index}")]
            total, weight, mean = values
            if not all(math.isfinite(v) for v in values) or weight <= 0:
                return [side("non_finite_training_record", f"{label}: invalid step {index}")]
            if not math.isclose(mean, total / weight, rel_tol=1e-12, abs_tol=1e-12):
                return [missing("inconsistent_training_record", f"{label}: step {index}")]
            accumulator.add(index, total, weight)
        for expected, actual in zip(accumulator.windows, stored):
            if actual["window"] != expected["window"] or actual["steps"] != expected["steps"]:
                return [missing("window_misaligned", f"{label}: window layout differs")]
            for key in ("loss_sum", "weight", "mean_loss"):
                value = actual[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not math.isfinite(expected[key])
                        or not math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12)):
                    return [missing("inconsistent_window", f"{label}: window {actual['window']} {key}")]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return [missing("invalid_training_record", f"{label}: {exc}")]
    return []


def compare_curves(candidate: Mapping[str, Any], native: Mapping[str, Any]) -> CurveComparison:
    """Distances between a candidate's and the native run's curves, if they align.

    Both arguments are :meth:`TrainingMeasurement.to_dict` results (or the
    measurements themselves). Misalignment of any kind -- a different plan,
    data stream, initial state or optimizer, an unfinished run, a missing or
    extra window or checkpoint, different window weights -- yields findings
    and no distances. Nothing is truncated to the shorter curve.
    """
    c = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
    n = native.to_dict() if hasattr(native, "to_dict") else dict(native)
    findings: list[Finding] = [missing("acceptance_not_configured", ACCEPTANCE_NOT_CONFIGURED)]
    if n.get("deployment", {}).get("role") != "native":
        findings.append(missing("not_native_reference", "the reference curve is not the native run"))
    for name, run, side in (("candidate", c, failure), ("native", n, missing)):
        if not run.get("completed"):
            side_code = "candidate_incomplete" if name == "candidate" else "native_incomplete"
            findings.append(side(side_code, f"the {name} run did not complete",
                                 reasons=[f.get("code") for f in run.get("findings", [])]))
    for field in ALIGNED_FIELDS:
        if c.get("identity", {}).get(field) != n.get("identity", {}).get(field):
            findings.append(missing("misaligned_" + field,
                                    f"{field} differs between the candidate and native runs"))
    findings.extend(training_record_problems(c, reference=False))
    findings.extend(training_record_problems(n, reference=True))
    if any(f.code != "acceptance_not_configured" for f in findings):
        return CurveComparison(findings, None)
    cw, nw = c.get("train_windows") or [], n.get("train_windows") or []
    if len(cw) != len(nw):
        findings.append(missing("window_count_mismatch", f"{len(cw)} candidate windows vs {len(nw)}"))
    else:
        for a, b in zip(cw, nw):  # lengths equal: nothing is dropped
            if a["window"] != b["window"] or a["steps"] != b["steps"]:
                findings.append(missing("window_misaligned", f"window {a['window']} vs {b['window']}"))
            elif a["weight"] != b["weight"]:
                findings.append(missing("window_weight_mismatch",
                                        f"window {a['window']}: weight {a['weight']} vs {b['weight']}"))
    cv, nv = c.get("validation") or {}, n.get("validation") or {}
    if set(cv) != set(nv):
        findings.append(missing("checkpoint_mismatch",
                                f"checkpoints {sorted(cv, key=int)} vs {sorted(nv, key=int)}"))
    else:
        for key in nv:
            if cv[key]["weight"] != nv[key]["weight"]:
                findings.append(missing("checkpoint_weight_mismatch", f"checkpoint {key}"))
    blocking = [f for f in findings if f.code != "acceptance_not_configured"]
    if blocking:
        return CurveComparison(findings, None)
    window_deltas = [abs(a["mean_loss"] - b["mean_loss"]) for a, b in zip(cw, nw)]
    checkpoint_deltas = {k: abs(cv[k]["mean_loss"] - nv[k]["mean_loss"]) for k in sorted(nv, key=int)}
    return CurveComparison(findings, {
        "max_window_abs_delta": max(window_deltas) if window_deltas else None,
        "max_checkpoint_abs_delta": max(checkpoint_deltas.values()) if checkpoint_deltas else None,
        "window_abs_deltas": window_deltas,
        "checkpoint_abs_deltas": checkpoint_deltas,
        "windows": len(window_deltas), "checkpoints": sorted(checkpoint_deltas, key=int),
    })
