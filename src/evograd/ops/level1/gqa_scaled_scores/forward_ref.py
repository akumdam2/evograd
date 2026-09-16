"""References for scaled grouped-query attention scores: ``s = (q @ k_exp^T) / sqrt(D)``.

Two spellings of one contract.

``gqa_scaled_scores_forward_ref`` is the oracle: the KV heads expanded to the
query heads, the contraction in float32 with IEEE operands, the scale applied
to the float32 product. It materializes ``[B, HQ, T, T]`` float32 -- 512 MiB
at the Qwen3-0.6B decomposition size -- which is exactly what the contract's
output is.

``gqa_scaled_scores_runtime`` is the eager spelling a PyTorch user would write
for bf16 inputs with a float32 score tensor: one tensor-core GEMM with float32
accumulation and a float32 result (``torch.bmm(..., out_dtype=float32)``), the
scale applied afterwards in float32, and a backward whose GEMMs consume the
inputs' dtype (``ds`` is cast to it) with float32 accumulation and a float32
group reduction. It is the timed baseline; ``verify_runtime_forward`` and the
calibration report record how far it sits from the oracle.

The contract is the first third of the attention decomposition
(scores -> causal softmax -> PV) and stops at the score tensor. Float32
output is the contract: a BF16 model does not make every intermediate BF16,
and the softmax that consumes ``s`` is defined on float32 scores.
"""

import math

import torch


def _check(q: torch.Tensor, k: torch.Tensor) -> int:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be [B, heads, T, D]")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k must agree on batch, sequence and head dimension")
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    return n_q // n_kv


def gqa_scaled_scores_forward_ref(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    groups = _check(q, k)
    scale = 1.0 / math.sqrt(q.shape[-1])
    key = k.repeat_interleave(groups, dim=1)
    return torch.matmul(q.float(), key.float().transpose(-2, -1)) * scale


class _RuntimeScores(torch.autograd.Function):
    """bf16 tensor-core GEMMs with float32 accumulation, float32 scores."""

    @staticmethod
    def forward(ctx, q, k):
        groups = _check(q, k)
        batch, n_q, tokens, dim = q.shape
        scale = 1.0 / math.sqrt(dim)
        key = k.repeat_interleave(groups, dim=1)
        q2 = q.reshape(batch * n_q, tokens, dim)
        k2 = key.reshape(batch * n_q, tokens, dim).transpose(-2, -1)
        if q.dtype == torch.float32:
            scores = torch.bmm(q2, k2)
        else:
            scores = torch.bmm(q2, k2, out_dtype=torch.float32)
        scores = scores.mul_(scale).view(batch, n_q, tokens, tokens)
        ctx.save_for_backward(q, key)
        ctx.groups = groups
        ctx.scale = scale
        return scores

    @staticmethod
    def backward(ctx, ds):
        q, key = ctx.saved_tensors
        batch, n_q, tokens, dim = q.shape
        n_kv = n_q // ctx.groups
        # The gradient of a bf16 input is consumed by bf16 tensor cores: cast the
        # float32 cotangent once, accumulate in float32, reduce the KV groups in
        # float32 and cast the results to the inputs' dtype.
        ds2 = ds.to(q.dtype).reshape(batch * n_q, tokens, tokens)
        q2 = q.reshape(batch * n_q, tokens, dim)
        k2 = key.reshape(batch * n_q, tokens, dim)
        if q.dtype == torch.float32:
            dq = torch.bmm(ds2, k2)
            dk_exp = torch.bmm(ds2.transpose(-2, -1), q2)
        else:
            dq = torch.bmm(ds2, k2, out_dtype=torch.float32)
            dk_exp = torch.bmm(ds2.transpose(-2, -1), q2, out_dtype=torch.float32)
        dq = dq.mul_(ctx.scale).view(batch, n_q, tokens, dim).to(q.dtype)
        dk = (
            dk_exp.mul_(ctx.scale)
            .view(batch, n_kv, ctx.groups, tokens, dim)
            .sum(dim=2)
            .to(key.dtype)
        )
        return dq, dk


def gqa_scaled_scores_runtime(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The eager baseline: tensor-core GEMM, float32 accumulate and output."""
    return _RuntimeScores.apply(q, k)
