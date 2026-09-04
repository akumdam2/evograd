"""A small, patch-set-matched whole-model correctness gate.

The detailed policy in :mod:`numerics` gates 54 groups on two metrics each, and
every one of them is a hard failure. That is hard to act on -- a run fails on
``optimizer.exp_avg_sq|k_proj`` and nothing about the number says whether the
kernel is wrong -- and one of the 54 is derived in a way that does not describe
the run being judged: the S/B reference patches *every* site, so a candidate
that replaces one site is measured against the integration drift of four.

This module keeps the elementwise layer where it belongs -- every live
invocation is still checked against its declared ``atol + rtol * |reference|``
by :mod:`boundary`, unchanged -- and shrinks the *model-level* gate to four
questions:

    is every output and gradient present
    is everything finite
    how far did the logits move          (relative L2)
    how far did all gradients move       (one global relative L2)

Each threshold is calibrated for the exact ``(workload, dtype, patch set)`` the
candidate will be judged in, against a trusted replacement that patches exactly
the sites the candidate patches. Everything else the detailed policy measured is
still collected and reported -- as a diagnostic, not a gate.

The candidate never contributes to a threshold. That is the whole point of a
reference calibration, and it is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

SCHEMA_VERSION = "evograd-qwen3-t3-numerics/3"

#: The metrics this gate actually enforces, in the order a failure reports them.
HARD_METRICS = ("logits_rel_l2", "global_grad_rel_l2")

#: Everything else the step produces. Collected, reported, never gated here.
DIAGNOSTIC_METRICS = (
    "loss_abs_delta", "loss_rel_delta", "logits_max_abs_over_rms",
    "global_grad_max_abs_over_rms", "per_role_gradient", "parameter_update",
    "exp_avg", "exp_avg_sq", "loss_trajectory",
)

#: Same floors the detailed policy uses, for the same reason: a configuration
#: whose reference noise happens to be exactly zero must not derive a gate that
#: demands bitwise equality of something with no reason to be bitwise. Sized as
#: a small fraction of bfloat16's unit roundoff (2^-8 = 3.9e-3).
FLOORS: dict[str, float] = {
    "logits_rel_l2": 1e-5,
    "global_grad_rel_l2": 1e-4,
}

#: Matches the detailed policy. Frozen before any candidate is measured.
SAFETY_MARGIN = 2.0


class PolicyMismatch(RuntimeError):
    """A policy was applied to something it was not calibrated for."""


# ── patch-set identity ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatchSet:
    """Which sites a provider replaces, which come along, how often each runs.

    Two providers may only be compared at model level when these agree. A
    QKV-only candidate and an all-sites bound-pair reference differ in three
    sites' worth of arithmetic, and calling the difference between them "the
    candidate's drift" attributes to the kernel what the reference did.
    """

    patched: tuple[str, ...]
    supporting: tuple[str, ...]
    expected_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, kernels, *, layers: int) -> PatchSet:
        """Read the patch set off a live ``KernelSet``."""
        from .boundary import SitePlan

        plan = SitePlan.build(kernels.patched, layers=layers)
        return cls(patched=plan.patched, supporting=plan.supporting,
                   expected_counts=dict(plan.expected))

    @property
    def key(self) -> str:
        """A stable name for this patch set, for filenames and report keys."""
        return "+".join(self.patched) if self.patched else "eager"

    def matches(self, other: PatchSet) -> bool:
        return (tuple(self.patched) == tuple(other.patched)
                and tuple(self.supporting) == tuple(other.supporting)
                and dict(self.expected_counts) == dict(other.expected_counts))

    def to_dict(self) -> dict[str, Any]:
        return {"patched": list(self.patched), "supporting": list(self.supporting),
                "expected_counts": dict(self.expected_counts), "key": self.key}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PatchSet:
        return cls(patched=tuple(payload["patched"]),
                   supporting=tuple(payload["supporting"]),
                   expected_counts=dict(payload.get("expected_counts") or {}))


def matched_trusted_kernels(ops, patch_set: PatchSet, registry):
    """The trusted replacement for one patch set: the same sites, bound.

    Deliberately *not* ``bound_pair_identity_kernels(ops, None, ...)``, whose
    ``sites=None`` means every registered site. The supporting adapters are not
    passed either -- they install themselves, because patching any member of an
    adapter group installs that group's adapter, which is exactly what happens
    for the candidate too.
    """
    from .sites import bound_pair_identity_kernels

    return bound_pair_identity_kernels(ops, tuple(patch_set.patched),
                                       registry=registry)


# ── the two gated metrics ────────────────────────────────────────────────────


def _rel_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    difference = (candidate.to(torch.float64) - reference.to(torch.float64))
    denominator = float(reference.to(torch.float64).norm())
    return float(difference.norm()) / max(denominator, 1e-30)


def _max_abs_over_rms(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    reference64 = reference.to(torch.float64)
    rms = float((reference64 ** 2).mean().sqrt())
    peak = float((candidate.to(torch.float64) - reference64).abs().max())
    return peak / max(rms, 1e-30)


def global_grad_rel_l2(candidate: dict[str, torch.Tensor],
                       reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    """One relative L2 over every identically named parameter gradient.

        sqrt(sum ||c - r||^2) / sqrt(sum ||r||^2)

    Streamed: each parameter contributes two float64 scalars and is released,
    so the peak is one gradient rather than a concatenation of all 310. The
    accumulators are float64 on the host because the summands span many orders
    of magnitude across a 0.6B model and a float32 running sum loses the small
    ones.
    """
    numerator = 0.0
    denominator = 0.0
    peak = 0.0
    rms_accumulator = 0.0
    elements = 0
    missing: list[str] = []
    extra = sorted(set(candidate) - set(reference))
    non_finite: list[str] = []
    shape_mismatch: list[str] = []

    for name in sorted(reference):
        reference_grad = reference[name]
        candidate_grad = candidate.get(name)
        if candidate_grad is None:
            missing.append(name)
            continue
        if tuple(candidate_grad.shape) != tuple(reference_grad.shape):
            shape_mismatch.append(name)
            continue
        r64 = reference_grad.detach().to(torch.float64)
        c64 = candidate_grad.detach().to(torch.float64)
        if not bool(torch.isfinite(c64).all()):
            non_finite.append(name)
        difference = c64 - r64
        numerator += float((difference ** 2).sum())
        denominator += float((r64 ** 2).sum())
        rms_accumulator += float((r64 ** 2).sum())
        elements += r64.numel()
        peak = max(peak, float(difference.abs().max()))
        del r64, c64, difference

    rms = (rms_accumulator / elements) ** 0.5 if elements else 0.0
    return {
        "rel_l2": (numerator ** 0.5) / max(denominator ** 0.5, 1e-30),
        "max_abs_over_rms": peak / max(rms, 1e-30),
        "parameters": len(reference),
        "compared": len(reference) - len(missing) - len(shape_mismatch),
        "missing": missing[:16],
        "missing_count": len(missing),
        "extra": extra[:16],
        "shape_mismatch": shape_mismatch[:16],
        "non_finite": non_finite[:16],
        "ok": not missing and not shape_mismatch and not non_finite,
    }


def measure(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """The simplified metrics for one candidate capture against one reference."""
    candidate_logits = candidate["logits"]
    reference_logits = reference["logits"]
    grads = global_grad_rel_l2(candidate["grads"], reference["grads"])

    candidate_loss = float(candidate["loss"])
    reference_loss = float(reference["loss"])
    logits_finite = bool(torch.isfinite(candidate_logits).all())
    loss_finite = bool(candidate_loss == candidate_loss and abs(candidate_loss) != float("inf"))

    return {
        # hard
        "logits_rel_l2": _rel_l2(candidate_logits, reference_logits),
        "global_grad_rel_l2": grads["rel_l2"],
        # presence and finiteness, also hard
        "missing_grads": list(candidate.get("missing_grads") or []),
        "grad_presence": {k: grads[k] for k in
                          ("parameters", "compared", "missing", "missing_count",
                           "extra", "shape_mismatch", "non_finite", "ok")},
        "finite": {
            "logits": logits_finite,
            "loss": loss_finite,
            "gradients": not grads["non_finite"],
            "ok": logits_finite and loss_finite and not grads["non_finite"],
        },
        # diagnostics
        "logits_max_abs_over_rms": _max_abs_over_rms(candidate_logits, reference_logits),
        "global_grad_max_abs_over_rms": grads["max_abs_over_rms"],
        "loss_abs_delta": abs(candidate_loss - reference_loss),
        "loss_rel_delta": abs(candidate_loss - reference_loss) / max(abs(reference_loss), 1e-30),
        "candidate_loss": candidate_loss,
        "reference_loss": reference_loss,
    }


# ── the policy ───────────────────────────────────────────────────────────────


@dataclass
class SimplePolicy:
    """Thresholds for one exact (workload, dtype, patch set), and their origin."""

    workload_id: str
    workload_hash: str
    dtype: str
    environment_hash: str
    patch_set: PatchSet
    thresholds: dict[str, float]
    derivation: dict[str, dict[str, float]]
    margin: float = SAFETY_MARGIN
    schema: str = SCHEMA_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "workload_id": self.workload_id,
            "workload_hash": self.workload_hash,
            "dtype": self.dtype,
            "environment_hash": self.environment_hash,
            "patch_set": self.patch_set.to_dict(),
            "thresholds": dict(self.thresholds),
            "derivation": {k: dict(v) for k, v in self.derivation.items()},
            "margin": self.margin,
            "hard_metrics": list(HARD_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimplePolicy:
        if payload.get("schema") != SCHEMA_VERSION:
            raise PolicyMismatch(
                f"policy schema {payload.get('schema')!r} is not {SCHEMA_VERSION!r}"
            )
        return cls(
            workload_id=payload["workload_id"],
            workload_hash=payload["workload_hash"],
            dtype=payload["dtype"],
            environment_hash=payload["environment_hash"],
            patch_set=PatchSet.from_dict(payload["patch_set"]),
            thresholds=dict(payload["thresholds"]),
            derivation={k: dict(v) for k, v in payload["derivation"].items()},
            margin=payload.get("margin", SAFETY_MARGIN),
            notes=dict(payload.get("notes") or {}),
        )

    def require_binding(self, *, workload_id: str, workload_hash: str, dtype: str,
                        environment_hash: str, patch_set: PatchSet) -> None:
        """Refuse to judge anything this policy was not calibrated for."""
        problems = []
        if workload_id != self.workload_id:
            problems.append(f"workload {workload_id!r} != {self.workload_id!r}")
        if workload_hash != self.workload_hash:
            problems.append(f"workload hash {workload_hash!r} != {self.workload_hash!r}")
        if dtype != self.dtype:
            problems.append(f"dtype {dtype!r} != {self.dtype!r}")
        if environment_hash != self.environment_hash:
            problems.append(
                f"environment {environment_hash!r} != {self.environment_hash!r}")
        if not patch_set.matches(self.patch_set):
            problems.append(
                f"patch set {patch_set.to_dict()} != {self.patch_set.to_dict()}")
        if problems:
            raise PolicyMismatch(
                "this calibration does not describe the run being judged: "
                + "; ".join(problems)
            )


def derive_simple_policy(
    *,
    reference_noise: Iterable[dict[str, Any]],
    trusted_drift: Iterable[dict[str, Any]],
    workload_id: str,
    workload_hash: str,
    dtype: str,
    environment_hash: str,
    patch_set: PatchSet,
    margin: float = SAFETY_MARGIN,
    notes: dict[str, Any] | None = None,
) -> SimplePolicy:
    """threshold = max(reference noise, trusted drift, floor) * margin.

    ``reference_noise`` is eager against an independently rebuilt eager -- the
    device's own run-to-run drift. ``trusted_drift`` is eager against the
    *matched* trusted replacement -- the cost of going through the declared
    pair at exactly the sites the candidate will replace. Neither involves a
    candidate, and this function has no way to see one.
    """
    noise = list(reference_noise)
    drift = list(trusted_drift)
    if not noise or not drift:
        raise ValueError(
            "both reference-noise and trusted-drift samples are required; "
            f"got {len(noise)} and {len(drift)}"
        )
    thresholds: dict[str, float] = {}
    derivation: dict[str, dict[str, float]] = {}
    for metric in HARD_METRICS:
        noise_max = max(float(s[metric]) for s in noise)
        drift_max = max(float(s[metric]) for s in drift)
        floor = FLOORS[metric]
        base = max(noise_max, drift_max, floor)
        thresholds[metric] = base * margin
        derivation[metric] = {
            "reference_noise_max": noise_max,
            "trusted_drift_max": drift_max,
            "floor": floor,
            "binding_term": ("trusted_drift" if base == drift_max else
                             "reference_noise" if base == noise_max else "floor"),
            "base": base,
            "margin": margin,
            "threshold": base * margin,
            "noise_samples": len(noise),
            "drift_samples": len(drift),
        }
    return SimplePolicy(
        workload_id=workload_id, workload_hash=workload_hash, dtype=dtype,
        environment_hash=environment_hash, patch_set=patch_set,
        thresholds=thresholds, derivation=derivation, margin=margin,
        notes=dict(notes or {}),
    )


def check(policy: SimplePolicy, metrics: dict[str, Any]) -> dict[str, Any]:
    """The verdict, with the first failure named.

    Order is presence, then finiteness, then the two magnitudes -- because a
    missing gradient makes every magnitude below it meaningless, and a NaN
    makes them arbitrary.
    """
    failures: list[dict[str, Any]] = []

    missing = list(metrics.get("missing_grads") or [])
    presence = metrics.get("grad_presence") or {}
    missing = missing or list(presence.get("missing") or [])
    if missing or presence.get("shape_mismatch"):
        failures.append({
            "metric": "presence",
            "reason": (f"{len(missing) or len(presence.get('shape_mismatch', []))} "
                       f"gradients missing or mis-shaped, first "
                       f"{(missing or presence.get('shape_mismatch'))[0]}"),
        })
    finite = metrics.get("finite") or {}
    if not finite.get("ok", False):
        bad = [k for k in ("logits", "loss", "gradients") if not finite.get(k, True)]
        failures.append({"metric": "finite",
                         "reason": f"non-finite {', '.join(bad) or 'values'}"})
    for metric in HARD_METRICS:
        value = float(metrics[metric])
        threshold = policy.thresholds[metric]
        if value > threshold:
            failures.append({
                "metric": metric, "value": value, "threshold": threshold,
                "ratio": value / threshold if threshold else float("inf"),
                "reason": (f"{metric} = {value:.4e} > {threshold:.4e} "
                           f"({value / threshold:.2f}x)"),
            })

    return {
        "schema": SCHEMA_VERSION,
        "ok": not failures,
        "failed_at": failures[0]["metric"] if failures else None,
        "reason": failures[0]["reason"] if failures else None,
        "failures": failures,
        "measured": {m: float(metrics[m]) for m in HARD_METRICS},
        "thresholds": dict(policy.thresholds),
        "ratios": {m: float(metrics[m]) / policy.thresholds[m]
                   if policy.thresholds[m] else float("inf")
                   for m in HARD_METRICS},
        "patch_set": policy.patch_set.to_dict(),
        "diagnostics": {k: metrics[k] for k in metrics
                        if k not in HARD_METRICS and k != "grad_presence"},
        "grad_presence": presence,
    }
