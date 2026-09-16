"""References for the grouped-query probability-value product ``o = p @ v_exp``.

``gqa_pv_forward_ref`` is the oracle: the KV heads expanded, the contraction
with float32 operands and accumulation, the result cast to the inputs' dtype.
``gqa_pv_runtime`` is the eager spelling a training step would use: one
``torch.matmul`` on the stored dtype (bf16 tensor cores, float32 accumulation,
bf16 result) whose autograd backward is two more such GEMMs and the group
reduction that ``repeat_interleave``'s derivative performs. It is the timed
baseline.

Third of the three primitives the causal grouped-query attention boundary
decomposes into (scores -> causal softmax -> PV). ``p`` arrives already
masked and already in the model's dtype; strictly-future entries are zero.
"""

import torch


def _check(p: torch.Tensor, v: torch.Tensor) -> int:
    if p.ndim != 4 or v.ndim != 4:
        raise ValueError("p must be [B, HQ, T, T] and v [B, HK, T, D]")
    if p.shape[0] != v.shape[0] or p.shape[2] != v.shape[2] or p.shape[3] != v.shape[2]:
        raise ValueError("p and v must agree on batch and sequence length")
    n_q, n_kv = p.shape[1], v.shape[1]
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    return n_q // n_kv


def gqa_pv_forward_ref(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    groups = _check(p, v)
    value = v.repeat_interleave(groups, dim=1)
    return torch.matmul(p.float(), value.float()).to(v.dtype)


def gqa_pv_runtime(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The eager baseline: one matmul in the stored dtype, float32 accumulate."""
    groups = _check(p, v)
    return torch.matmul(p, v.repeat_interleave(groups, dim=1))
