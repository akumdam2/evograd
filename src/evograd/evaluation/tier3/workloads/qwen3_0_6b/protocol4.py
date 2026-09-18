"""The four-part Qwen3 correctness protocol, schema evograd-qwen3-t3-protocol/4.

Four questions, each with its own metric, floor and threshold:

    A  local computation     every live invocation, every output and gradient,
                             against its declared atol/rtol   (boundary.py, unchanged)
    B  model prediction      mean valid-token KL(P_eager || P_provider), nats
    C  model gradients       global parameter-gradient vector relative L2
    D  training behaviour    max |Δ| of fixed-window training NLL, and of
                             validation NLL at fixed checkpoints, nats/token

B and C are measured on the first training batch before the first optimizer
update. D is measured over a predeclared horizon with a trusted eager
evaluator on disjoint held-out text. Raw-logit error, parameter updates, Adam
moments and per-role errors are collected and reported as diagnostics.

Every threshold is

    T_m = MARGIN * max( max compile-vs-eager distance over calibration seeds,
                        max repeated-eager distance over calibration seeds,
                        FLOOR_m )

derived from references only, frozen with its full binding, and then applied to
holdout seeds the calibration never saw -- to the trusted compile provider as
well as to the candidate. A candidate's numbers cannot reach the derivation:
the calibration takes a patch set, not a program.

**Design provenance.** The local elementwise layer follows the practice of
Liger-Kernel's and FlashAttention's correctness tests; comparing trained
models rather than operators follows Cut Cross-Entropy and FlashMask. The KL
anchor, the compile-vs-eager calibration and the automated trajectory
thresholds are this project's design choices and are not published standards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .prediction import IGNORE_INDEX, mean_token_kl
from .simple import PatchSet, PolicyMismatch, global_grad_rel_l2

SCHEMA_VERSION = "evograd-qwen3-t3-protocol/4"

HARD_METRICS = (
    "kl_mean",
    "global_grad_rel_l2",
    "train_window_nll_max_abs_delta",
    "val_nll_max_abs_delta",
)

MARGIN = 2.0

#: The two single-step metrics on their own. A *screening* policy calibrates and
#: enforces only these; part D (training behaviour) is recorded as a diagnostic
#: because the 1,000-step reference experiment of 2026-09-05 was itself unstable
#: and its events did not reproduce. A screening policy carries this plan so it
#: can never bind to a run that expects four enforced parts.
SCREENING_METRICS = ("kl_mean", "global_grad_rel_l2")
SCREENING_PLAN = {"mode": "screening_abc", "steps": 0,
                  "note": "A/B/C-gated Tier-3 screening; part D diagnostic only"}
TRAINING_METRICS = ("train_window_nll_max_abs_delta", "val_nll_max_abs_delta")

#: Independent floors, one per metric, each justified by a numerical check and
#: a reference-only measurement rather than inherited from another metric.
FLOORS: dict[str, float] = {
    # nats/token. Two identical bfloat16 logit tensors up-cast to float32 give a
    # KL of exactly 0.0 (checked in tests). Differing only by float32 summation
    # order gives |KL| ~1e-7 per position. 1e-6 is 10x above that arithmetic and
    # four orders below the KL a provider difference produces at this model.
    # NOT derived from the logits floor: KL and logit rel_l2 have different
    # units and different sensitivities.
    "kl_mean": 1e-6,
    # unitless relative L2 of a 0.6B-element vector. Same quantity and same
    # justification as the tensor-level floor in numerics.py: a whole-vector
    # relative L2 below 1e-4 is below bfloat16 unit roundoff (2^-8 = 3.9e-3) by
    # a factor of 39 and cannot be attributed to a kernel on purpose.
    "global_grad_rel_l2": 1e-4,
    # nats/token, a 50-step token-weighted window mean. The loss is a float32
    # scalar ~2-4 nats whose ULP is ~4e-7; summation over ~2e5 tokens per
    # window adds roundoff well below 1e-6. 1e-4 is the smallest change in a
    # window mean this pipeline would treat as a difference at all; the
    # repeated-eager control measures the true process noise and binds instead
    # whenever it is larger, which at bfloat16 it is expected to be.
    "train_window_nll_max_abs_delta": 1e-4,
    # nats/token over ~2.6e5 held-out tokens. Same reasoning; same expectation
    # that the repeated-eager term rather than this floor is what binds.
    "val_nll_max_abs_delta": 1e-4,
}

DIAGNOSTIC_METRICS = (
    "logits_rel_l2", "logits_rel_l2_valid", "kl_std", "kl_max_position",
    "train_window_deltas", "validation_deltas", "perplexity",
    "parameter_update", "exp_avg", "exp_avg_sq", "per_role_gradient",
)

METRIC_DEFINITIONS = {
    "kl_mean": ("mean over valid next-token positions of sum_v P_eager(v) "
                "[log P_eager(v) - log P_provider(v)]; float32 log_softmax, "
                "temperature 1, float64 accumulation, causal shift "
                f"(position t predicts t+1), ignore_index={IGNORE_INDEX}; measured on "
                "the first training batch before the first optimizer update"),
    "global_grad_rel_l2": ("sqrt(sum_p ||g_provider,p - g_eager,p||^2) / "
                           "max(sqrt(sum_p ||g_eager,p||^2), 1e-30), float64 "
                           "accumulation over identically named parameters, "
                           "before the first optimizer update"),
    "train_window_nll_max_abs_delta": ("max over fixed windows of |mean training "
                                       "NLL_provider - mean training NLL_eager|, "
                                       "nats per valid token, token-weighted within "
                                       "a window"),
    "val_nll_max_abs_delta": ("max over fixed checkpoints of |validation "
                              "NLL_provider - validation NLL_eager|, nats per valid "
                              "token, token-weighted, both through the trusted eager "
                              "evaluator on the same held-out text"),
}


# ── B and C on one batch, in one process ─────────────────────────────────────


def capture_first_step(workload, kernels, ids, labels) -> dict[str, Any]:
    """Forward and backward on one batch; logits to host, gradients on device.

    No optimizer step is taken: B and C are defined *before* the first update.
    """
    model, provenance = workload.build_patched(kernels)
    model.train()
    logits = model(input_ids=ids, use_cache=False).logits
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    loss = torch.nn.functional.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum") / int(mask.sum())
    loss.backward()
    # A torch.compile'd model reports its parameters as ``_orig_mod.<name>``;
    # the gradient comparison matches by name, so read them off the original
    # module (found 2026-09-11: the whole-model compile baseline compared zero
    # parameters and reported a relative L2 of exactly 0.0).
    named = getattr(model, "_orig_mod", model).named_parameters
    grads = {n: p.grad.detach().clone() for n, p in named() if p.grad is not None}
    missing = [n for n, p in named() if p.grad is None]
    built = workload.last_build
    capture = {
        "logits": logits.detach().to("cpu"),      # bfloat16 on the host
        "loss": float(loss.detach()),
        "grads": grads,
        "missing_grads": missing,
        "provenance": provenance.to_dict(),
        "counts": built.observed() if built else {},
    }
    del model, logits, shifted, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return capture


def step_distances(provider: dict[str, Any], reference: dict[str, Any],
                   labels: torch.Tensor, *, device: str = "cuda") -> dict[str, Any]:
    """B and C for one provider capture against one eager capture."""
    kl = mean_token_kl(reference["logits"], provider["logits"], labels.cpu(), device=device)
    grads = global_grad_rel_l2(provider["grads"], reference["grads"])
    finite_logits = bool(torch.isfinite(provider["logits"]).all())
    return {
        "kl_mean": kl["kl_for_gate"],
        "kl_mean_raw": kl["kl_mean"],
        "kl_roundoff_negative": kl["kl_roundoff_negative"],
        "kl_std": kl["kl_std"], "kl_max_position": kl["kl_max_position"],
        "valid_positions": kl["valid_positions"],
        "global_grad_rel_l2": grads["rel_l2"],
        "grad_presence": {k: grads[k] for k in ("parameters", "compared", "missing",
                                                "missing_count", "extra",
                                                "shape_mismatch", "non_finite", "ok",
                                                "worst_by_squared_error", "worst_by_rel_l2")},
        "missing_grads": list(provider.get("missing_grads") or []),
        "finite": {"logits": finite_logits, "gradients": not grads["non_finite"],
                   "loss": provider["loss"] == provider["loss"],
                   "ok": finite_logits and not grads["non_finite"]},
        # diagnostics
        "logits_rel_l2_valid": kl["logits_rel_l2_valid"],
        "loss_provider": provider["loss"], "loss_reference": reference["loss"],
        "loss_abs_delta": abs(provider["loss"] - reference["loss"]),
    }


# ── the policy ───────────────────────────────────────────────────────────────


@dataclass
class Protocol4Policy:
    workload_id: str
    workload_hash: str
    dtype: str
    environment_hash: str
    patch_set: PatchSet
    data_identity_digest: str
    training_plan: dict[str, Any]
    thresholds: dict[str, float]
    derivation: dict[str, dict[str, Any]]
    margin: float = MARGIN
    schema: str = SCHEMA_VERSION
    metric_definitions: dict[str, str] = field(default_factory=lambda: dict(METRIC_DEFINITIONS))
    floors: dict[str, float] = field(default_factory=lambda: dict(FLOORS))
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "workload_id": self.workload_id,
            "workload_hash": self.workload_hash, "dtype": self.dtype,
            "environment_hash": self.environment_hash,
            "patch_set": self.patch_set.to_dict(),
            "data_identity_digest": self.data_identity_digest,
            "training_plan": dict(self.training_plan),
            "thresholds": dict(self.thresholds),
            "derivation": {k: dict(v) for k, v in self.derivation.items()},
            "margin": self.margin, "floors": dict(self.floors),
            "metric_definitions": dict(self.metric_definitions),
            "hard_metrics": list(self.thresholds),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Protocol4Policy":
        if payload.get("schema") != SCHEMA_VERSION:
            raise PolicyMismatch(
                f"policy schema {payload.get('schema')!r} is not {SCHEMA_VERSION!r}; "
                "older schemas keep their own meaning and are not read here")
        return cls(
            workload_id=payload["workload_id"], workload_hash=payload["workload_hash"],
            dtype=payload["dtype"], environment_hash=payload["environment_hash"],
            patch_set=PatchSet.from_dict(payload["patch_set"]),
            data_identity_digest=payload["data_identity_digest"],
            training_plan=dict(payload["training_plan"]),
            thresholds=dict(payload["thresholds"]),
            derivation={k: dict(v) for k, v in payload["derivation"].items()},
            margin=payload.get("margin", MARGIN),
            metric_definitions=dict(payload.get("metric_definitions") or METRIC_DEFINITIONS),
            floors=dict(payload.get("floors") or FLOORS),
            notes=dict(payload.get("notes") or {}),
        )

    def require_binding(self, *, workload_id, workload_hash, dtype, environment_hash,
                        patch_set: PatchSet, data_identity_digest, training_plan) -> None:
        problems = []
        if workload_id != self.workload_id:
            problems.append(f"workload {workload_id!r} != {self.workload_id!r}")
        if workload_hash != self.workload_hash:
            problems.append("workload hash differs")
        if dtype != self.dtype:
            problems.append(f"dtype {dtype!r} != {self.dtype!r}")
        if environment_hash != self.environment_hash:
            problems.append(f"environment {environment_hash!r} != {self.environment_hash!r}")
        if not patch_set.matches(self.patch_set):
            problems.append(f"patch set {patch_set.to_dict()} != {self.patch_set.to_dict()}")
        if data_identity_digest != self.data_identity_digest:
            problems.append("data identity (dataset revision, tokenizer, packing, masks) differs")
        if dict(training_plan) != dict(self.training_plan):
            problems.append(f"training plan {training_plan} != {self.training_plan}")
        if self.metric_definitions != METRIC_DEFINITIONS:
            problems.append("metric definitions in the policy differ from this code's")
        if problems:
            raise PolicyMismatch(
                "this calibration does not describe the run being judged: "
                + "; ".join(problems))


def derive_policy(*, compile_distances: Iterable[dict[str, Any]],
                  repeat_distances: Iterable[dict[str, Any]],
                  workload_id, workload_hash, dtype, environment_hash,
                  patch_set: PatchSet, data_identity_digest, training_plan,
                  margin: float = MARGIN, notes=None,
                  metrics: tuple[str, ...] = HARD_METRICS) -> Protocol4Policy:
    """T_m = margin * max(max compile-vs-eager, max repeated-eager, floor_m).

    ``metrics`` is ``HARD_METRICS`` for the four-part policy and
    ``SCREENING_METRICS`` for an A/B/C screening policy; a screening policy must
    carry ``SCREENING_PLAN`` so the two can never be mistaken for each other.
    """
    compile_ = list(compile_distances)
    repeat = list(repeat_distances)
    if not compile_ or not repeat:
        raise ValueError(
            f"need compile and repeated-eager samples; got {len(compile_)} and {len(repeat)}")
    if set(metrics) == set(SCREENING_METRICS) and dict(training_plan) != SCREENING_PLAN:
        raise ValueError("a screening policy must carry SCREENING_PLAN as its training plan")
    if set(metrics) == set(HARD_METRICS) and dict(training_plan) == SCREENING_PLAN:
        raise ValueError("the four-part policy cannot carry the screening plan")
    thresholds, derivation = {}, {}
    for metric in metrics:
        c = max(float(s[metric]) for s in compile_)
        r = max(float(s[metric]) for s in repeat)
        floor = FLOORS[metric]
        base = max(c, r, floor)
        binding = "compile_vs_eager" if base == c else "repeated_eager" if base == r else "floor"
        thresholds[metric] = base * margin
        derivation[metric] = {
            "compile_vs_eager_max": c, "repeated_eager_max": r, "floor": floor,
            "base": base, "binding_term": binding, "margin": margin,
            "threshold": base * margin,
            "compile_samples": len(compile_), "repeat_samples": len(repeat),
        }
    return Protocol4Policy(
        workload_id=workload_id, workload_hash=workload_hash, dtype=dtype,
        environment_hash=environment_hash, patch_set=patch_set,
        data_identity_digest=data_identity_digest, training_plan=dict(training_plan),
        thresholds=thresholds, derivation=derivation, margin=margin,
        notes=dict(notes or {}),
    )


def requires_training_part(policy: Protocol4Policy) -> bool:
    """Does this policy enforce part D? A screening policy does not."""
    return any(metric in policy.thresholds for metric in TRAINING_METRICS)


def is_screening(policy: Protocol4Policy) -> bool:
    return dict(policy.training_plan) == SCREENING_PLAN and not requires_training_part(policy)


def check(policy: Protocol4Policy, measured: dict[str, Any]) -> dict[str, Any]:
    """Presence, then finiteness, then every magnitude the policy carries. First failure named."""
    failures = []
    presence = measured.get("grad_presence") or {}
    missing = list(measured.get("missing_grads") or []) or list(presence.get("missing") or [])
    if missing or presence.get("shape_mismatch") or presence.get("extra"):
        failures.append({"metric": "presence",
                         "reason": (f"gradients missing={len(missing)} "
                                    f"mis-shaped={len(presence.get('shape_mismatch') or [])} "
                                    f"extra={len(presence.get('extra') or [])}")})
    finite = measured.get("finite") or {}
    if not finite.get("ok", False):
        failures.append({"metric": "finite", "reason": "non-finite logits, loss or gradients"})
    if measured.get("non_finite_training_steps"):
        failures.append({"metric": "finite",
                         "reason": f"non-finite training loss at steps "
                                   f"{measured['non_finite_training_steps'][:4]}"})
    for metric in policy.thresholds:
        if metric not in measured:
            failures.append({"metric": metric, "reason": f"{metric} was not measured"})
            continue
        value, threshold = float(measured[metric]), policy.thresholds[metric]
        if value > threshold:
            failures.append({"metric": metric, "value": value, "threshold": threshold,
                             "ratio": value / threshold,
                             "reason": f"{metric} = {value:.4e} > {threshold:.4e} "
                                       f"({value / threshold:.2f}x)"})
    return {
        "schema": SCHEMA_VERSION, "ok": not failures,
        "failed_at": failures[0]["metric"] if failures else None,
        "reason": failures[0]["reason"] if failures else None,
        "failures": failures,
        "screening": is_screening(policy),
        "measured": {m: float(measured[m]) for m in policy.thresholds if m in measured},
        "thresholds": dict(policy.thresholds),
        "ratios": {m: float(measured[m]) / policy.thresholds[m]
                   for m in policy.thresholds if m in measured},
        "diagnostics": {k: v for k, v in measured.items()
                        if k not in policy.thresholds and k not in ("grad_presence",)},
    }


def load_policy(path) -> Protocol4Policy:
    return Protocol4Policy.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))["policy"])
