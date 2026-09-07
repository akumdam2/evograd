"""Training behaviour: continuous training NLL and held-out validation NLL.

The single-step gate asks whether one forward and backward agree. This asks
whether *training* does: the same initial checkpoint, the same fresh optimizer
state, the same batch order, the same learning-rate schedule and the same step
count for every provider, and two token-weighted quantities read off the run:

* the mean training NLL over each fixed window of steps -- nats per valid
  token, weighted by tokens, so a window's mean is the sum of its per-token
  losses over its valid-token count and not an average of per-step averages;
* the validation NLL at fixed checkpoint steps, computed by **one trusted
  eager evaluator** over the same held-out text for every provider. The
  trained parameters are copied into an unpatched model for this, so what is
  compared is the *weights training produced*, not the provider's own forward.

Both are diagnostics of training *behaviour* over a declared horizon. They
say nothing about where training ends up after a hundred times as many steps,
and the horizon is stored in the policy so the claim cannot outgrow it.

Design lineage, stated so it is not mistaken for a standard: elementwise local
checks in the style of Liger-Kernel and FlashAttention's test suites bound the
kernel; training-curve comparison follows the practice in Cut Cross-Entropy
and FlashMask of comparing the trained model rather than the operator. The
thresholds those curves are held to here are this project's own choices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .prediction import IGNORE_INDEX, mean_token_nll

#: The predeclared experiment. Changing any of these changes the policy's
#: identity, so a calibration taken at one horizon cannot judge another.
DEFAULT_STEPS = 1000
DEFAULT_WINDOW = 50
DEFAULT_CHECKPOINTS = (0, 100, 500, 1000)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01      # torch.optim.AdamW's default, written down
DEFAULT_BETAS = (0.9, 0.999)


@dataclass(frozen=True)
class TrainingPlan:
    steps: int = DEFAULT_STEPS
    window: int = DEFAULT_WINDOW
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    betas: tuple[float, float] = DEFAULT_BETAS
    optimizer: str = "AdamW"
    schedule: str = "constant"

    def __post_init__(self) -> None:
        if self.steps % self.window:
            raise ValueError(f"steps {self.steps} is not a multiple of window {self.window}")
        bad = [c for c in self.checkpoints if c < 0 or c > self.steps]
        if bad:
            raise ValueError(f"checkpoints {bad} fall outside 0..{self.steps}")

    @property
    def windows(self) -> int:
        return self.steps // self.window

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "window": self.window,
                "checkpoints": list(self.checkpoints),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "betas": list(self.betas),
                "optimizer": self.optimizer, "schedule": self.schedule}


def token_weighted_nll(model, input_ids, labels) -> tuple[torch.Tensor, int]:
    """``(sum of -log P over valid positions, valid count)`` for one batch.

    Computed from the logits explicitly rather than through the model's own
    ``labels=`` path, so the normalisation is the one in ``mean_token_nll`` for
    every provider and for validation alike, and the backward is taken on the
    sum -- the mean is applied after, by the caller, over the window's tokens.
    """
    logits = model(input_ids=input_ids, use_cache=False).logits
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    count = int(mask.sum())
    if count == 0:
        raise ValueError("a training batch with no valid next-token positions")
    loss_sum = torch.nn.functional.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    return loss_sum, count


@torch.no_grad()
def evaluate_validation(evaluator_model, batches, *, steps: int) -> dict[str, Any]:
    """Token-weighted validation NLL over a fixed set of held-out batches.

    ``evaluator_model`` is the trusted eager model with the trained weights
    loaded in. ``batches`` yields ``(input_ids, labels)`` for ``steps`` batches
    in a fixed order; the same object is used for every provider and every
    checkpoint, so the held-out set is literally the same tokens.
    """
    was_training = evaluator_model.training
    evaluator_model.eval()
    nll_sum, tokens = 0.0, 0
    try:
        for index in range(steps):
            ids, labels = batches.batch(index)
            logits = evaluator_model(input_ids=ids, use_cache=False).logits
            part = mean_token_nll(logits, labels, device=logits.device)
            nll_sum += part["nll_sum"]
            tokens += part["valid_tokens"]
            del logits
    finally:
        if was_training:
            evaluator_model.train()
    mean = nll_sum / tokens
    return {"nll_mean": mean, "nll_sum": nll_sum, "valid_tokens": tokens,
            "perplexity": float(torch.exp(torch.tensor(mean))), "batches": steps}


def run_training(
    workload,
    kernels,
    *,
    plan: TrainingPlan,
    train_batches,
    validation_batches,
    validation_steps: int,
    data_seed: int,
    evaluator_factory: Callable[[], torch.nn.Module],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one provider from the initial checkpoint and read off both curves.

    ``evaluator_factory`` builds the trusted eager model used for validation;
    it is built once here and reloaded with the trained weights at each
    checkpoint. Training state -- parameters and optimizer moments -- is
    continuous across the whole run; nothing is reset at a checkpoint.
    """
    say = progress or (lambda _m: None)
    started = time.time()

    model, provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate,
                                  betas=plan.betas, weight_decay=plan.weight_decay)
    evaluator = evaluator_factory()
    evaluator.eval()

    def validate(step: int) -> dict[str, Any]:
        # The trained weights, through the unpatched evaluator. `load_state_dict`
        # copies; the provider's parameters are not aliased.
        evaluator.load_state_dict(model.state_dict(), strict=True)
        result = evaluate_validation(evaluator, validation_batches, steps=validation_steps)
        say(f"    step {step:>5} validation nll {result['nll_mean']:.6f} "
            f"({result['valid_tokens']} tokens)")
        return result

    validation: dict[int, dict[str, Any]] = {}
    if 0 in plan.checkpoints:
        validation[0] = validate(0)

    window_sum, window_tokens = 0.0, 0
    windows: list[dict[str, Any]] = []
    per_step: list[float] = []
    non_finite_steps: list[int] = []
    for step in range(1, plan.steps + 1):
        ids, labels = train_batches.batch(step - 1)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count = token_weighted_nll(model, ids, labels)
        if not torch.isfinite(loss_sum):
            non_finite_steps.append(step)
            raise RuntimeError(f"non-finite training loss at step {step}")
        # Backward on the mean over this batch's tokens, as a trainer would.
        (loss_sum / count).backward()
        optimizer.step()
        value = float(loss_sum.detach())
        window_sum += value
        window_tokens += count
        per_step.append(value / count)
        del loss_sum
        if step % plan.window == 0:
            windows.append({"window": step // plan.window, "steps": [step - plan.window + 1, step],
                            "nll_mean": window_sum / window_tokens,
                            "valid_tokens": window_tokens})
            say(f"    step {step:>5} window {step // plan.window:>2} train nll "
                f"{window_sum / window_tokens:.6f}")
            window_sum, window_tokens = 0.0, 0
        if step in plan.checkpoints:
            validation[step] = validate(step)

    counters = workload.last_build
    result = {
        "plan": plan.to_dict(), "data_seed": data_seed,
        "provenance": provenance.to_dict(),
        "observed_counts": counters.observed() if counters else {},
        "train_windows": windows,
        "train_window_nll": [w["nll_mean"] for w in windows],
        "train_per_step_nll": per_step,
        "validation": {str(k): v for k, v in validation.items()},
        "validation_nll": {str(k): v["nll_mean"] for k, v in validation.items()},
        "non_finite_steps": non_finite_steps,
        "train_batches": train_batches.describe(),
        "validation_batches": validation_batches.describe(),
        "seconds": time.time() - started,
    }
    del model, optimizer, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


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
