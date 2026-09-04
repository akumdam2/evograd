"""Whole-model numerical metrics, grouping, and the calibrated noise envelope.

Tier 3's site preflight proves a kernel correct at its declared shapes. It says
nothing about the model those kernels are assembled into, and a whole-model
check needs a threshold -- which is where a single global number like
``atol=0.08`` comes from, and why it is indefensible: it is not derived from
anything, it is the same for a 151936x1024 embedding gradient and a 128-element
per-head norm, and it cannot tell a wrong kernel from the GPU's own irreducible
run-to-run drift.

This module is the machinery for replacing it with a measured one.

**Three comparisons, never conflated.** ``E/E`` is the unmodified model against
an independently rebuilt unmodified model, and it measures the hardware. ``E/S``
is eager against the structural adapters, and it asks whether restructuring the
modules added anything on top. ``S/B`` is structural against the bound pair, and
it measures what ``opdecl.bind`` and the declared runtime spellings cost. A
single "identity" number would hide which of the three moved.

**Statistics that survive near-zero elements.** Maximum *relative* error is
useless here: a gradient element that should be 1e-9 and comes out 2e-9 is a
100% relative error and means nothing. The primary statistics are the relative
L2 error over the whole tensor and the maximum absolute error normalized by a
documented scale floor, with cosine agreement and above-floor sign disagreement
alongside. Everything is computed in one streaming pass and only the summary is
kept.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

SCHEMA_VERSION = "evograd-qwen3-t3-numerics/2"

# ── tensor kinds ─────────────────────────────────────────────────────────────
#
# A gradient, the update one optimizer step applied, and Adam's two moments are
# four different quantities with four different scales, and pooling them gives a
# threshold set by whichever is loudest -- in practice the gradient, which is
# ~30x looser than an update needs. Then a wrong update hides under a gradient's
# noise. Each kind gets its own namespace, and a role only ever compares against
# samples of its own kind.
KIND_OUTPUT = "output"
KIND_GRADIENT = "gradient"
KIND_UPDATE = "parameter_update"
KIND_EXP_AVG = "optimizer.exp_avg"
KIND_EXP_AVG_SQ = "optimizer.exp_avg_sq"
KIND_STEP = "optimizer.step"

TENSOR_KINDS = (KIND_OUTPUT, KIND_GRADIENT, KIND_UPDATE, KIND_EXP_AVG,
                KIND_EXP_AVG_SQ, KIND_STEP)

#: Kinds compared exactly rather than against an envelope. Adam's step counter
#: is an integer count of optimizer steps; "close enough" is not a thing it can
#: be, and a provider that has taken a different number of steps than the
#: reference is not a numerical question.
EXACT_KINDS = (KIND_STEP,)


def group_key(kind: str, role: str) -> str:
    """The namespace a sample is judged in. Kind first: it is the coarser split."""
    return f"{kind}|{role}"

#: Below this fraction of a tensor's RMS an element carries no information about
#: correctness -- it is the sum of cancellations. Sign disagreement and relative
#: statistics are only counted above it.
NOISE_FLOOR_FRACTION = 1e-3

#: The absolute floor under a scale, so a genuinely all-zero tensor divides by
#: something rather than by nothing. Chosen at bfloat16's smallest normal, which
#: no meaningful gradient is below.
SCALE_FLOOR = 1e-30


# ── parameter roles ──────────────────────────────────────────────────────────

#: Which role each parameter plays, matched on its name. Grouping by role rather
#: than per-tensor is what makes the envelope a policy instead of 310 constants:
#: a role has one shape and one scale across every layer, so its per-layer
#: samples describe one distribution. Roles whose scales differ -- an embedding
#: is 151936x1024, a per-head norm is 128 -- are never pooled.
#:
#: These are the HuggingFace decoder naming convention, not one model's: Llama,
#: Mistral and Qwen all spell ``self_attn.q_proj`` and ``mlp.down_proj`` the same
#: way. ``q_norm``/``k_norm`` are Qwen3's per-head normalizations, which simply
#: do not appear in an architecture that lacks them. An architecture that names
#: something else is not mis-pooled: :func:`role_of` falls through to
#: ``other:<name>``, giving the parameter a group of its own, which is the safe
#: direction -- an ungrouped tensor gets its own envelope rather than borrowing
#: a threshold measured on a different scale.
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"embed_tokens\.weight$", "embedding"),
    (r"^lm_head\.weight$", "lm_head"),
    (r"^model\.norm\.weight$", "final_norm"),
    (r"input_layernorm\.weight$", "input_layernorm"),
    (r"post_attention_layernorm\.weight$", "post_attention_layernorm"),
    (r"self_attn\.q_norm\.weight$", "q_norm"),
    (r"self_attn\.k_norm\.weight$", "k_norm"),
    (r"self_attn\.q_proj\.weight$", "q_proj"),
    (r"self_attn\.k_proj\.weight$", "k_proj"),
    (r"self_attn\.v_proj\.weight$", "v_proj"),
    (r"self_attn\.o_proj\.weight$", "o_proj"),
    (r"mlp\.gate_proj\.weight$", "gate_proj"),
    (r"mlp\.up_proj\.weight$", "up_proj"),
    (r"mlp\.down_proj\.weight$", "down_proj"),
)

#: Results that are not parameters but still need a group of their own.
SCALAR_ROLES = ("logits", "loss")


#: Prefixes a result name can carry to say *which* quantity it is. The role is a
#: property of the parameter, not of whether this sample is its gradient, its
#: post-step value or its optimizer moment, so they are stripped before matching.
#: Two of the role patterns are anchored at the start of the name, and without
#: this a stepped ``model.norm.weight`` fell outside every envelope and failed
#: the gate on a provider that was perfectly correct.
RESULT_PREFIXES = ("step:", "update:", "exp_avg:", "exp_avg_sq:",
                   "step_count:")


def base_name(name: str) -> str:
    for prefix in RESULT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def role_of(name: str) -> str:
    """The group a named result belongs to. Unmatched names get their own."""
    if name in SCALAR_ROLES:
        return name
    stripped = base_name(name)
    for pattern, role in ROLE_PATTERNS:
        if re.search(pattern, stripped):
            return role
    return f"other:{stripped}"


def roles_present(names: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(role_of(name), []).append(name)
    return groups


# ── one comparison of one tensor ─────────────────────────────────────────────


@dataclass(frozen=True)
class TensorStats:
    """Everything worth knowing about one pair of tensors, and no tensor."""

    name: str
    role: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    finite: bool
    reference_finite: bool
    bitwise: bool
    max_abs_err: float
    #: ``||a-b||_2 / max(||b||_2, floor)``. The primary statistic: it is a whole
    #: -tensor quantity, so one cancelling element cannot dominate it.
    rel_l2: float
    #: ``max|a-b| / max(rms(b), floor)``. The secondary statistic, in units of
    #: the tensor's own typical magnitude rather than of an arbitrary constant.
    max_abs_over_rms: float
    ref_rms: float
    ref_absmax: float
    cosine: float
    #: Elements whose sign flipped, counted only where ``|b|`` is above the
    #: noise floor. Below it a sign is not information.
    sign_flips_above_floor: int
    elements_above_floor: int

    @property
    def group(self) -> str:
        return group_key(self.kind, self.role)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shape": list(self.shape), "group": self.group}


def compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor,
                   *, kind: str = KIND_GRADIENT) -> TensorStats:
    """One streaming pass. Neither tensor is retained by this function."""
    a = actual.detach().to(torch.float32)
    b = expected.detach().to(torch.float32)
    diff = a - b
    ref_norm = float(b.norm())
    ref_rms = float(b.pow(2).mean().sqrt()) if b.numel() else 0.0
    floor = max(ref_rms, SCALE_FLOOR)
    above = b.abs() > NOISE_FLOOR_FRACTION * floor
    n_above = int(above.sum())
    flips = int(((a * b) < 0).logical_and(above).sum()) if n_above else 0
    denom = float(a.norm()) * ref_norm
    return TensorStats(
        name=name,
        role=role_of(name),
        kind=kind,
        shape=tuple(actual.shape),
        dtype=str(actual.dtype).removeprefix("torch."),
        elements=int(b.numel()),
        finite=bool(torch.isfinite(a).all()),
        reference_finite=bool(torch.isfinite(b).all()),
        bitwise=bool(torch.equal(actual.detach(), expected.detach())),
        max_abs_err=float(diff.abs().max()) if b.numel() else 0.0,
        rel_l2=float(diff.norm()) / max(ref_norm, SCALE_FLOOR),
        max_abs_over_rms=(float(diff.abs().max()) / floor) if b.numel() else 0.0,
        ref_rms=ref_rms,
        ref_absmax=float(b.abs().max()) if b.numel() else 0.0,
        cosine=float((a * b).sum() / denom) if denom > 0 else 1.0,
        sign_flips_above_floor=flips,
        elements_above_floor=n_above,
    )


# ── the environment a calibration is only valid in ───────────────────────────


def environment_fingerprint() -> dict[str, Any]:
    """Everything that can move a number without anyone editing a file.

    A calibration is a measurement of *this* machine. Re-using it on another
    driver, another cuDNN, or with TF32 flipped would be re-using someone else's
    noise floor, so all of it is recorded and hashed.
    """
    from importlib.metadata import PackageNotFoundError, version

    def package(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    fingerprint: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": package("transformers"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "sdpa_backends": _sdpa_backends(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        fingerprint.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "driver": _driver_version(),
        })
    else:
        fingerprint.update({"gpu_name": None, "compute_capability": None, "driver": None})
    return fingerprint


def _sdpa_backends() -> dict[str, bool]:
    backend = torch.backends.cuda
    return {
        name: bool(getattr(backend, f"{name}_sdp_enabled")())
        for name in ("flash", "mem_efficient", "math")
        if hasattr(backend, f"{name}_sdp_enabled")
    }


def _driver_version() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return None


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


#: Fields whose change invalidates a calibration outright. A different GPU or a
#: different SDPA backend is a different noise floor; a different transformers
#: patch release might be, so it is included and the operator can recalibrate.
BINDING_FIELDS = (
    "gpu_name", "compute_capability", "driver", "torch", "cuda", "cudnn",
    "transformers", "tf32_matmul", "float32_matmul_precision",
    "deterministic_algorithms", "sdpa_backends",
)


def environment_mismatch(stored: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Which binding fields differ. Empty means the calibration still applies."""
    return [
        f"{field}: calibrated on {stored.get(field)!r}, running on {live.get(field)!r}"
        for field in BINDING_FIELDS
        if stored.get(field) != live.get(field)
    ]


# ── the envelope ─────────────────────────────────────────────────────────────

#: Multiplied onto the observed calibration maximum. Two, because the statistic
#: being bounded is itself a maximum over a finite sample and the holdout has to
#: fit under it without being what set it.
SAFETY_MARGIN = 2.0

#: Metrics the gate actually enforces, in the order a failure reports them.
GATED_METRICS = ("rel_l2", "max_abs_over_rms")

#: A floor under every derived threshold, so a configuration whose E/E noise
#: happens to be zero cannot produce a gate that demands bitwise equality of
#: something that has no reason to be bitwise. Sized at a small fraction of
#: bfloat16's unit roundoff (2^-8 = 3.9e-3): a whole-tensor relative L2 below
#: 1e-4, or a peak below 1% of the tensor's own RMS, is not a disagreement any
#: correct implementation can be held to have avoided on purpose.
THRESHOLD_FLOOR: dict[str, float] = {"rel_l2": 1e-4, "max_abs_over_rms": 1e-2}

#: Per-kind floors, because the kinds do not share a scale. A parameter update
#: after one AdamW step is ~lr in every element, so its noise is far below a
#: gradient's and a gradient-sized floor would be the whole threshold.
KIND_THRESHOLD_FLOOR: dict[str, dict[str, float]] = {
    KIND_OUTPUT: {"rel_l2": 1e-5, "max_abs_over_rms": 1e-3},
    KIND_GRADIENT: {"rel_l2": 1e-4, "max_abs_over_rms": 1e-2},
    KIND_UPDATE: {"rel_l2": 1e-5, "max_abs_over_rms": 1e-3},
    KIND_EXP_AVG: {"rel_l2": 1e-5, "max_abs_over_rms": 1e-3},
    KIND_EXP_AVG_SQ: {"rel_l2": 1e-5, "max_abs_over_rms": 1e-3},
}


def floor_for(group: str) -> dict[str, float]:
    kind = group.split("|", 1)[0]
    return KIND_THRESHOLD_FLOOR.get(kind, THRESHOLD_FLOOR)


@dataclass
class GroupEnvelope:
    """One role's bound on each gated metric, and what it was derived from."""

    role: str
    tensors: int
    samples: int
    observed_max: dict[str, float] = field(default_factory=dict)
    observed_p99: dict[str, float] = field(default_factory=dict)
    threshold: dict[str, float] = field(default_factory=dict)
    margin: float = SAFETY_MARGIN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def derive_envelope(samples: Iterable[dict[str, Any]], *, margin: float = SAFETY_MARGIN,
                    floor: dict[str, float] | None = None
                    ) -> dict[str, GroupEnvelope]:
    """Per-role bounds from reference-only samples.

    ``threshold = max(observed maximum, THRESHOLD_FLOOR) * margin``. The maximum
    rather than a quantile because the gate has to accept every correct run, not
    99% of them; the margin because the maximum of a finite sample
    underestimates the maximum of the process, and the holdout seeds -- which
    did not contribute here -- have to fit underneath. The floor because a
    configuration small enough to be deterministic would otherwise derive a
    zero threshold and demand bitwise equality of something that has no reason
    to be bitwise; at the canonical size every observed maximum is well above
    it, so it binds nothing there.
    """
    by_group: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        # Exactly-compared kinds have a verdict, not a distribution. An
        # optimizer step counter has no envelope to derive.
        if sample.get("kind") in EXACT_KINDS:
            continue
        by_group.setdefault(
            sample.get("group") or group_key(sample.get("kind", KIND_GRADIENT),
                                             sample["role"]),
            [],
        ).append(sample)

    envelopes: dict[str, GroupEnvelope] = {}
    for role, group in by_group.items():
        envelope = GroupEnvelope(
            role=role,
            tensors=len({s["name"] for s in group}),
            samples=len(group),
        )
        for metric in GATED_METRICS:
            values = [float(s[metric]) for s in group]
            observed = max(values) if values else 0.0
            envelope.observed_max[metric] = observed
            envelope.observed_p99[metric] = _percentile(values, 0.99)
            limits = floor if floor is not None else floor_for(role)
            base = max(observed, limits.get(metric, 0.0))
            envelope.threshold[metric] = base * margin
        envelopes[role] = envelope
    return envelopes


def combined_envelope(hardware: dict[str, GroupEnvelope],
                      integration: dict[str, GroupEnvelope]) -> dict[str, GroupEnvelope]:
    """The bound a provider is actually held to: noise plus known drift.

    A provider reaching the *eager* model crosses two things it did not choose:
    the device's run-to-run drift (E/E) and the integration cost of going
    through ``bind`` and the declared runtime spellings (S/B). Gating on E/E
    alone would fail a numerically perfect kernel for the second; gating on S/B
    alone would forgive a real error the size of the first. The bound is their
    sum, per role and per metric, and both halves stay visible.
    """
    roles = set(hardware) | set(integration)
    combined: dict[str, GroupEnvelope] = {}
    for role in roles:
        left = hardware.get(role)
        right = integration.get(role)
        base = left or right
        merged = GroupEnvelope(
            role=role,
            tensors=base.tensors,
            samples=(left.samples if left else 0) + (right.samples if right else 0),
            margin=base.margin,
        )
        for metric in GATED_METRICS:
            merged.observed_max[metric] = max(
                (left.observed_max.get(metric, 0.0) if left else 0.0),
                (right.observed_max.get(metric, 0.0) if right else 0.0),
            )
            merged.observed_p99[metric] = merged.observed_max[metric]
            merged.threshold[metric] = (
                (left.threshold.get(metric, 0.0) if left else 0.0)
                + (right.threshold.get(metric, 0.0) if right else 0.0)
            )
        combined[role] = merged
    return combined


def check_against(envelopes: dict[str, GroupEnvelope],
                  samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Do these samples fit inside that envelope? Returns every exceedance."""
    exceed: list[dict[str, Any]] = []
    checked = 0
    for sample in samples:
        checked += 1
        group = sample.get("group") or group_key(
            sample.get("kind", KIND_GRADIENT), sample["role"]
        )
        # Exactly-compared kinds carry their own verdict and never meet an
        # envelope: an optimizer step counter is right or it is not.
        if sample.get("kind") in EXACT_KINDS:
            if not sample.get("exact", False):
                exceed.append({"name": sample["name"], "role": sample["role"],
                               "group": group, "metric": "exact",
                               "reason": "optimizer step counter differs"})
            continue
        envelope = envelopes.get(group)
        if envelope is None:
            exceed.append({"name": sample["name"], "role": sample["role"],
                           "group": group,
                           "reason": f"no envelope for {group}"})
            continue
        for metric in GATED_METRICS:
            value = float(sample[metric])
            limit = envelope.threshold.get(metric)
            if limit is not None and value > limit:
                exceed.append({
                    "name": sample["name"], "role": sample["role"],
                    "group": group, "kind": sample.get("kind"),
                    "metric": metric, "value": value, "threshold": limit,
                    "ratio": value / limit if limit else math.inf,
                })
    return {"checked": checked, "exceeded": exceed, "ok": not exceed}


# ── loss trajectories ────────────────────────────────────────────────────────


@dataclass
class TrajectoryPolicy:
    """How far a loss curve may move, over exactly how many steps.

    The horizon is part of the policy and not a footnote: divergence compounds
    through the optimizer, so a bound measured over five steps says nothing
    about fifty, and a policy that did not carry its horizon would be quoted for
    both.
    """

    horizon: int
    optimizer: str
    learning_rate: float
    max_abs_delta: float
    max_rel_delta: float
    margin: float = SAFETY_MARGIN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, reference: list[float], candidate: list[float]) -> dict[str, Any]:
        if len(reference) != len(candidate):
            return {"ok": False, "reason": "trajectory lengths differ"}
        if len(reference) != self.horizon:
            return {"ok": False, "reason":
                    f"policy is for {self.horizon} steps, got {len(reference)}"}
        deltas = [abs(a - b) for a, b in zip(candidate, reference)]
        rel = [
            abs(a - b) / abs(b) if abs(b) > 0 else 0.0
            for a, b in zip(candidate, reference)
        ]
        return {
            "ok": bool(max(deltas) <= self.max_abs_delta
                       and max(rel) <= self.max_rel_delta),
            "max_abs_delta": max(deltas),
            "max_rel_delta": max(rel),
            "limits": {"abs": self.max_abs_delta, "rel": self.max_rel_delta},
            "horizon": self.horizon,
        }


def derive_trajectory_policy(deltas: list[tuple[float, float]], *, horizon: int,
                             optimizer: str, learning_rate: float,
                             margin: float = SAFETY_MARGIN) -> TrajectoryPolicy:
    absolute = max((d for d, _ in deltas), default=0.0)
    relative = max((r for _, r in deltas), default=0.0)
    return TrajectoryPolicy(
        horizon=horizon, optimizer=optimizer, learning_rate=learning_rate,
        max_abs_delta=absolute * margin, max_rel_delta=relative * margin,
        margin=margin,
    )


# ── the stored policy ────────────────────────────────────────────────────────


@dataclass
class NumericsPolicy:
    """A calibration, loadable, environment-bound, and refusing to travel."""

    schema_version: str
    workload_id: str
    workload_hash: str
    environment: dict[str, Any]
    environment_hash: str
    envelopes: dict[str, GroupEnvelope]
    trajectory: TrajectoryPolicy
    #: What S/B measured, kept separate from the hardware floor: it is a known
    #: integration drift, not noise, and a candidate is compared against both.
    bound_pair_envelopes: dict[str, GroupEnvelope] = field(default_factory=dict)
    bound_pair_trajectory: TrajectoryPolicy | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "workload_hash": self.workload_hash,
            "environment": self.environment,
            "environment_hash": self.environment_hash,
            "envelopes": {k: v.to_dict() for k, v in self.envelopes.items()},
            "trajectory": self.trajectory.to_dict(),
            "bound_pair_envelopes": {
                k: v.to_dict() for k, v in self.bound_pair_envelopes.items()
            },
            "bound_pair_trajectory": (
                self.bound_pair_trajectory.to_dict()
                if self.bound_pair_trajectory else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumericsPolicy":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"calibration schema {payload.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION!r}; recalibrate rather than reinterpreting it"
            )
        return cls(
            schema_version=payload["schema_version"],
            workload_id=payload["workload_id"],
            workload_hash=payload["workload_hash"],
            environment=payload["environment"],
            environment_hash=payload["environment_hash"],
            envelopes={k: GroupEnvelope(**v) for k, v in payload["envelopes"].items()},
            trajectory=TrajectoryPolicy(**payload["trajectory"]),
            bound_pair_envelopes={
                k: GroupEnvelope(**v)
                for k, v in (payload.get("bound_pair_envelopes") or {}).items()
            },
            bound_pair_trajectory=(
                TrajectoryPolicy(**payload["bound_pair_trajectory"])
                if payload.get("bound_pair_trajectory") else None
            ),
            notes=payload.get("notes", {}),
        )

    def applies_here(self, live: dict[str, Any] | None = None) -> list[str]:
        """Empty when this calibration is valid on the running machine."""
        return environment_mismatch(self.environment, live or environment_fingerprint())
