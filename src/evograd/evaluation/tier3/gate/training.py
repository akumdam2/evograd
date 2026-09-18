"""The training-behaviour loop, once, for any workload with a site registry.

Moved here from the Qwen3 package when Llama needed the same thing: the loop is
about *training*, not about an architecture. A workload supplies what only it
knows -- how to build a patched model, where its batches come from, and what a
trusted unpatched evaluator is -- and this supplies the rest:

* the plan (steps, window, checkpoints, optimizer settings), which is part of a
  claim's identity and is recorded with every result;
* a token-weighted training NLL per step and per fixed window, in nats per valid
  token, so a window mean is a sum over its tokens rather than a mean of means;
* validation NLL at fixed checkpoint steps, computed by **one trusted evaluator**
  over the same held-out batches for every provider, with the trained weights
  copied into an unpatched model -- what is compared is the weights training
  produced, not the provider's own forward. A wrong forward that lowers its own
  loss (a causal-mask fault reading the next token) therefore cannot flatter
  itself here;
* optional gradient-norm and parameter-update-norm probes at step 1 and at every
  checkpoint, and the run's peak device memory.

Validation never moves the training stream: it runs under ``no_grad``, indexes
its own batches positionally, and uses a separate model.

Both curves are diagnostics over a declared horizon. They say nothing about
where training ends up after a hundred times as many steps, which is why the
horizon travels inside the result.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

#: The label ``cross_entropy`` skips, and the one every workload here uses.
IGNORE_INDEX = -100
#: Rows per chunk when scoring logits. A [2, 2048, 151936] float32 log-softmax
#: is 2.5 GiB; chunking keeps validation inside the memory a training step needs.
DEFAULT_CHUNK = 256

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


def token_nll(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = IGNORE_INDEX,
              chunk: int = DEFAULT_CHUNK) -> dict[str, Any]:
    """``(sum of -log P(label), valid count)`` over valid next-token positions.

    Explicit rather than ``F.cross_entropy(reduction="mean")`` so the
    normalization is one written-down object used by training and validation
    alike: sum over valid positions of -log P(label), divided by the number of
    valid positions. Returns the sum and the count too, so a caller aggregating
    over batches weights by tokens instead of averaging averages. Chunked over
    rows and computed in float32.
    """
    if logits.dim() != 3 or labels.dim() != 2 or labels.shape != logits.shape[:2]:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} disagree")
    shifted = logits[:, :-1]
    targets = labels[:, 1:]
    flat_logits = shifted.reshape(-1, shifted.shape[-1])
    flat_targets = targets.reshape(-1)
    mask = flat_targets != ignore_index
    valid = int(mask.sum())
    if valid == 0:
        raise ValueError("a batch with no valid next-token positions")
    total = 0.0
    for start in range(0, flat_logits.shape[0], max(1, chunk)):
        piece = flat_logits[start:start + chunk].float()
        target = flat_targets[start:start + chunk]
        total += float(torch.nn.functional.cross_entropy(
            piece, target, ignore_index=ignore_index, reduction="sum"))
        del piece
    return {"nll_sum": total, "valid_tokens": valid, "nll_mean": total / valid}


def token_weighted_nll(model, input_ids, labels, *, ignore_index: int = IGNORE_INDEX):
    """``(differentiable sum of -log P, valid count)`` for one training batch.

    Computed from the logits explicitly rather than through a model's own
    ``labels=`` path, so the normalization is the one above for every provider,
    and the backward is taken on the sum -- the caller divides by the batch's
    token count.
    """
    logits = model(input_ids=input_ids, use_cache=False).logits
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != ignore_index
    count = int(mask.sum())
    if count == 0:
        raise ValueError("a training batch with no valid next-token positions")
    loss_sum = torch.nn.functional.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
        ignore_index=ignore_index, reduction="sum",
    )
    return loss_sum, count


@torch.no_grad()
def evaluate_validation(evaluator_model, batches, *, steps: int,
                        ignore_index: int = IGNORE_INDEX) -> dict[str, Any]:
    """Token-weighted validation NLL over a fixed set of held-out batches.

    ``evaluator_model`` is the trusted unpatched model with the trained weights
    loaded in. ``batches`` yields ``(input_ids, labels)`` for ``steps`` batches
    in a fixed order; the same object serves every provider and every
    checkpoint, so the held-out set is literally the same tokens.
    """
    was_training = evaluator_model.training
    evaluator_model.eval()
    nll_sum, tokens = 0.0, 0
    try:
        for index in range(steps):
            ids, labels = batches.batch(index)
            logits = evaluator_model(input_ids=ids, use_cache=False).logits
            part = token_nll(logits, labels, ignore_index=ignore_index)
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
    record_norms: bool = False,
) -> dict[str, Any]:
    """Train one provider from the initial checkpoint and read off both curves.

    ``evaluator_factory`` builds the trusted unpatched model used for
    validation; it is built once here and reloaded with the trained weights at
    each checkpoint. Training state -- parameters and optimizer moments -- is
    continuous across the whole run; nothing is reset at a checkpoint.

    ``record_norms`` additionally records, at step 1 and at every checkpoint
    step, the global gradient norm, the norm of the parameter update that step
    applied, how many parameters carried a gradient, and the run's peak device
    memory -- diagnostics read off the same step, never altering it.
    """
    say = progress or (lambda _m: None)
    started = time.time()
    if record_norms and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate,
                                  betas=plan.betas, weight_decay=plan.weight_decay)
    evaluator = evaluator_factory()
    evaluator.eval()

    def validate(step: int) -> dict[str, Any]:
        # The trained weights, through the unpatched evaluator. `load_state_dict`
        # copies; the provider's parameters are not aliased. A torch.compile'd
        # model keeps the real module under `_orig_mod`.
        evaluator.load_state_dict(getattr(model, "_orig_mod", model).state_dict(), strict=True)
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
    norms: dict[int, dict[str, Any]] = {}
    for step in range(1, plan.steps + 1):
        ids, labels = train_batches.batch(step - 1)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count = token_weighted_nll(model, ids, labels)
        if not torch.isfinite(loss_sum):
            non_finite_steps.append(step)
            raise RuntimeError(f"non-finite training loss at step {step}")
        # Backward on the mean over this batch's tokens, as a trainer would.
        (loss_sum / count).backward()
        probe = record_norms and (step == 1 or step in plan.checkpoints)
        if probe:
            params = [p for p in model.parameters() if p.grad is not None]
            grad_norm = math.sqrt(sum(float(p.grad.detach().to(torch.float64).pow(2).sum())
                                      for p in params))
            before = [p.detach().clone() for p in params]
        optimizer.step()
        if probe:
            update_norm = math.sqrt(sum(float((p.detach().to(torch.float64) - b.to(torch.float64)).pow(2).sum())
                                        for p, b in zip(params, before)))
            norms[step] = {"grad_norm": grad_norm, "update_norm": update_norm,
                           "grad_finite": math.isfinite(grad_norm),
                           "parameters_with_grad": len(params),
                           "parameters_without_grad": sum(1 for p in model.parameters() if p.grad is None)}
            del before
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
    if record_norms:
        result["norms"] = {str(k): v for k, v in norms.items()}
        result["peak_memory_bytes"] = (int(torch.cuda.max_memory_allocated())
                                       if torch.cuda.is_available() else None)
    del model, optimizer, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


__all__ = [
    "DEFAULT_BETAS", "DEFAULT_CHECKPOINTS", "DEFAULT_LEARNING_RATE", "DEFAULT_STEPS",
    "DEFAULT_WEIGHT_DECAY", "DEFAULT_WINDOW", "IGNORE_INDEX", "TrainingPlan",
    "evaluate_validation", "run_training", "token_nll", "token_weighted_nll",
]
