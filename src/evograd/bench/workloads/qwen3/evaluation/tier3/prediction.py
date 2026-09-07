"""Prediction-distribution disagreement: mean valid-token KL(P_eager || P_provider).

Raw-logit relative L2 asks how far two tensors are apart. That is not what a
language model is for. Two logit tensors can differ by a constant per position
and predict identically; two can differ by a small relative L2 concentrated on
the arg-max and predict differently. KL between the two next-token distributions
is the quantity that tracks what training actually consumes, so it is the hard
metric here and the logit error is kept as a diagnostic.

Conventions, all of them checked rather than assumed:

* **Direction.** ``KL(P_eager || P_provider)``: the reference is the
  expectation's measure. Reported per valid next-token position in nats.
* **Precision.** Full-vocabulary ``log_softmax`` in float32, temperature 1.
  The models emit bfloat16 logits; the up-cast is where every evaluator does
  it. Accumulation across positions is float64.
* **Normalisation.** Sum over the vocabulary, then the *token-weighted mean*
  over valid positions -- never PyTorch's default reduction, whose
  normalisation depends on what was passed.
* **Causal shift and masks.** Position ``t`` predicts token ``t+1``. A position
  is valid when the label at ``t+1`` is not ``ignore_index``. The last position
  predicts nothing and is never valid. This is exactly the shift
  ``Qwen3ForCausalLM`` applies to ``labels``.
* **Memory.** Positions are processed in chunks, so no full-size probability
  tensor is ever materialised for the whole sequence. Full-vocabulary
  normalisation is preserved inside every chunk.
* **Roundoff.** KL is non-negative in exact arithmetic. In float32 a per-position
  value can come out at ``-1e-7``-ish when the two distributions are nearly
  identical. A *mean* that is negative by less than ``ROUNDOFF_TOLERANCE`` is
  reported as measured, flagged, and treated as zero for gating. Anything more
  negative is an error and is raised, not clamped: a substantially negative KL
  means the inputs were not what this function was told they were.
"""

from __future__ import annotations

from typing import Any

import torch

IGNORE_INDEX = -100

#: The size below which a negative mean KL is float32 arithmetic rather than a
#: measurement. Two identical bfloat16 logit tensors up-cast to float32 give a
#: per-position KL of exactly 0.0; two that differ only by float32 summation
#: order give |KL| of order 1e-7 per position. 1e-6 nats is an order of
#: magnitude above that and four orders below anything a provider difference
#: produces at this model.
ROUNDOFF_TOLERANCE = 1e-6

#: Positions per chunk. 512 x 151936 x 4 bytes is 311 MB per float32 tensor;
#: two of them plus the elementwise intermediates stay under 1.5 GB.
DEFAULT_CHUNK = 512


class PredictionError(ValueError):
    """The inputs to the KL are not what a next-token comparison needs."""


def valid_positions(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Mask over positions ``[B, T-1]`` that predict a real next token.

    Position ``t`` is valid iff ``labels[:, t+1] != ignore_index``. The final
    position has no next token and is excluded by construction.
    """
    if labels.dim() != 2:
        raise PredictionError(f"labels must be [B, T], got {tuple(labels.shape)}")
    return labels[:, 1:] != ignore_index


def finalize_mean_kl(kl_sum: float, valid: int) -> tuple[float, bool]:
    """Turn the float64 sum into the mean, applying the roundoff policy.

    Negative by less than ``ROUNDOFF_TOLERANCE``: float32 arithmetic, returned as
    measured with the flag set (the gate uses ``max(mean, 0)``). More negative:
    an error, raised -- clamping it would hide a wrong comparison. Non-finite:
    an error.
    """
    mean = kl_sum / valid
    if not (mean == mean and abs(mean) != float("inf")):
        raise PredictionError(f"mean KL is not finite: {mean}")
    roundoff_negative = mean < 0.0
    if roundoff_negative and -mean > ROUNDOFF_TOLERANCE:
        raise PredictionError(
            f"mean KL is substantially negative ({mean:.3e} nats/token); the two "
            "inputs are not distributions over the same vocabulary or the "
            "reference is not the reference")
    return mean, roundoff_negative


def mean_token_kl(
    reference_logits: torch.Tensor,
    provider_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    chunk: int = DEFAULT_CHUNK,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Mean KL(P_reference || P_provider) over valid next-token positions, in nats.

    ``reference_logits`` and ``provider_logits`` are ``[B, T, V]`` in any float
    dtype and on any device; they are moved chunk by chunk to ``device`` and
    up-cast to float32 there. ``labels`` is ``[B, T]``.
    """
    if reference_logits.shape != provider_logits.shape:
        raise PredictionError(
            f"logit shapes differ: {tuple(reference_logits.shape)} vs "
            f"{tuple(provider_logits.shape)}")
    if reference_logits.dim() != 3:
        raise PredictionError(
            f"logits must be [B, T, V], got {tuple(reference_logits.shape)}")
    batch, length, vocab = reference_logits.shape
    if labels.shape != (batch, length):
        raise PredictionError(
            f"labels {tuple(labels.shape)} do not match logits [B, T] = "
            f"{(batch, length)}")
    if chunk < 1:
        raise PredictionError("chunk must be positive")

    mask = valid_positions(labels, ignore_index)          # [B, T-1]
    valid = int(mask.sum())
    if valid == 0:
        raise PredictionError("no valid next-token positions: every label is ignored")

    work = torch.device(device) if device is not None else reference_logits.device
    # Flatten the predicting positions [B, T-1, V] -> [B*(T-1), V] without
    # copying: narrow to :-1 then reshape a contiguous view.
    ref_flat = reference_logits[:, :-1].reshape(-1, vocab)
    prov_flat = provider_logits[:, :-1].reshape(-1, vocab)
    mask_flat = mask.reshape(-1)

    kl_sum = torch.zeros((), dtype=torch.float64)
    kl_sq_sum = torch.zeros((), dtype=torch.float64)
    kl_max = float("-inf")
    kl_min = float("inf")
    logit_diff_sq = torch.zeros((), dtype=torch.float64)
    logit_ref_sq = torch.zeros((), dtype=torch.float64)
    seen = 0
    non_finite_inputs = 0

    total = ref_flat.shape[0]
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        m = mask_flat[start:stop]
        if not bool(m.any()):
            continue
        r = ref_flat[start:stop][m].to(work, torch.float32)
        p = prov_flat[start:stop][m].to(work, torch.float32)
        if not (torch.isfinite(r).all() and torch.isfinite(p).all()):
            non_finite_inputs += int((~torch.isfinite(r)).sum() + (~torch.isfinite(p)).sum())
            raise PredictionError(
                f"non-finite logits in chunk [{start}, {stop}): "
                f"{non_finite_inputs} elements")
        log_p = torch.log_softmax(r, dim=-1)               # reference, float32
        log_q = torch.log_softmax(p, dim=-1)               # provider,  float32
        # sum_v p_v (log p_v - log q_v), per position, float32 -> float64
        per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1).to(torch.float64)
        kl_sum += per_pos.sum().cpu()
        kl_sq_sum += (per_pos * per_pos).sum().cpu()
        kl_max = max(kl_max, float(per_pos.max()))
        kl_min = min(kl_min, float(per_pos.min()))
        # diagnostics: raw logit disagreement on the same valid positions
        d = (p - r).to(torch.float64)
        logit_diff_sq += (d * d).sum().cpu()
        logit_ref_sq += (r.to(torch.float64) ** 2).sum().cpu()
        seen += int(m.sum())
        del r, p, log_p, log_q, per_pos, d

    if seen != valid:  # pragma: no cover - would be a masking bug
        raise PredictionError(f"processed {seen} positions but mask has {valid}")

    mean, roundoff_negative = finalize_mean_kl(float(kl_sum), valid)
    variance = max(float(kl_sq_sum) / valid - mean * mean, 0.0)
    return {
        "kl_mean": mean,
        "kl_for_gate": max(mean, 0.0),
        "kl_roundoff_negative": roundoff_negative,
        "kl_std": variance ** 0.5,
        "kl_max_position": kl_max,
        "kl_min_position": kl_min,
        "valid_positions": valid,
        "total_positions": int(batch * (length - 1)),
        "vocab": int(vocab),
        "chunk": int(chunk),
        "definition": ("mean over valid next-token positions of "
                       "sum_v P_ref(v) [log P_ref(v) - log P_prov(v)]; float32 "
                       "log_softmax, temperature 1, float64 accumulation, causal "
                       f"shift, ignore_index={ignore_index}"),
        # diagnostics only
        "logits_rel_l2_valid": (float(logit_diff_sq) ** 0.5)
                               / max(float(logit_ref_sq) ** 0.5, 1e-30),
    }


def mean_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    chunk: int = DEFAULT_CHUNK,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Token-weighted next-token negative log-likelihood, in nats per valid token.

    Explicit rather than ``F.cross_entropy(reduction="mean")`` so the
    normalisation is the same object for training and validation and is
    written down: ``sum over valid positions of -log P(label) / number of valid
    positions``. Returns the sum and the count as well, so a caller aggregating
    over many batches weights by tokens rather than averaging averages.
    """
    if logits.dim() != 3 or labels.dim() != 2 or labels.shape != logits.shape[:2]:
        raise PredictionError(
            f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} disagree")
    vocab = logits.shape[-1]
    mask = valid_positions(labels, ignore_index)
    valid = int(mask.sum())
    if valid == 0:
        raise PredictionError("no valid next-token positions")
    work = torch.device(device) if device is not None else logits.device
    flat = logits[:, :-1].reshape(-1, vocab)
    targets = labels[:, 1:].reshape(-1)
    mask_flat = mask.reshape(-1)
    nll_sum = torch.zeros((), dtype=torch.float64)
    for start in range(0, flat.shape[0], chunk):
        stop = min(start + chunk, flat.shape[0])
        m = mask_flat[start:stop]
        if not bool(m.any()):
            continue
        z = flat[start:stop][m].to(work, torch.float32)
        y = targets[start:stop][m].to(work)
        if not torch.isfinite(z).all():
            raise PredictionError(f"non-finite logits in chunk [{start}, {stop})")
        log_p = torch.log_softmax(z, dim=-1)
        nll_sum += (-log_p.gather(1, y[:, None]).squeeze(1)).to(torch.float64).sum().cpu()
        del z, y, log_p
    total = float(nll_sum)
    return {"nll_sum": total, "valid_tokens": valid, "nll_mean": total / valid,
            "perplexity": float(torch.exp(torch.tensor(total / valid)))}
