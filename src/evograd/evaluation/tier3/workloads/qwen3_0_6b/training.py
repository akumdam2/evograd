"""Qwen3's training-behaviour stage: the shared loop, plus this model's distances.

The loop itself -- the plan, the token-weighted training NLL, validation through
one trusted unpatched evaluator at fixed checkpoints, and the optional
gradient/update-norm probes -- is workload-neutral and lives in
:mod:`evograd.evaluation.tier3.gate.training`, which Llama uses too. It is
re-exported here so every existing import path and policy keeps working.

What stays here is Qwen-specific: the distances between a provider's curves and
the reference's, in the units protocol 4's part D is stated in.

Design lineage, stated so it is not mistaken for a standard: elementwise local
checks in the style of Liger-Kernel's and FlashAttention's test suites bound the
kernel; training-curve comparison follows the practice in Cut Cross-Entropy and
FlashMask of comparing the trained model rather than the operator. The
thresholds those curves are held to here are this project's design choices.
"""

from __future__ import annotations

from typing import Any

from evograd.evaluation.tier3.gate.training import (  # noqa: F401  (re-exported)
    DEFAULT_BETAS,
    DEFAULT_CHECKPOINTS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STEPS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WINDOW,
    TrainingPlan,
    evaluate_validation,
    run_training,
    token_weighted_nll,
)


def training_distances(provider: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """The two hard training-behaviour distances, and the per-point diagnostics."""
    pw, rw = provider["train_window_nll"], reference["train_window_nll"]
    if len(pw) != len(rw):
        raise ValueError(f"window counts differ: {len(pw)} vs {len(rw)}")
    pv, rv = provider["validation_nll"], reference["validation_nll"]
    if set(pv) != set(rv):
        raise ValueError(f"checkpoints differ: {sorted(pv)} vs {sorted(rv)}")
    window_deltas = [abs(a - b) for a, b in zip(pw, rw)]
    val_deltas = {k: abs(pv[k] - rv[k]) for k in sorted(pv, key=int)}
    return {
        "train_window_nll_max_abs_delta": max(window_deltas),
        "val_nll_max_abs_delta": max(val_deltas.values()),
        "train_window_deltas": window_deltas,
        "validation_deltas": val_deltas,
        "provider_validation_nll": pv, "reference_validation_nll": rv,
    }
