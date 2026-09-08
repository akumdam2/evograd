"""Forward references for Level-2 boundaries that two architectures share.

Three of the four fused boundaries a decoder layer decomposes into are the same
mathematics in Qwen3-0.6B and Llama-3-8B: causal grouped-query attention with
its output projection, the gated MLP, and the residual add followed by RMSNorm.
Only the widths differ, and a width is a case rather than a contract.

Each model still owns its own *task*: its own key, its own observed cases, its
own tolerances and its own provenance, in its own package. What they share is
the definition of the computation, written here once so that two declarations
cannot drift into computing subtly different things while claiming to measure
the same boundary. The bodies below are the ones Qwen3's references already
carried; they moved here unchanged, and each model re-exports them under the
symbol its own declaration points at.

Nothing here is model-specific and nothing here reads a snapshot.

The fourth boundary is not here. Q/K/V projection with rotation differs between
the two -- Qwen3 applies a per-head RMSNorm to q and k and Llama-3 does not --
so each has its own reference, and neither pretends otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# -- causal grouped-query attention plus the output projection --------------


def _check_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o_weight: torch.Tensor) -> int:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be [B, heads, T, D]")
    if k.shape != v.shape:
        raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k must agree on batch, sequence and head dimension")
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    fan_in = n_q * q.shape[3]
    if o_weight.ndim != 2 or o_weight.shape[1] != fan_in:
        raise ValueError(f"o_weight must be [H, {fan_in}], got {tuple(o_weight.shape)}")
    return n_q // n_kv

def _merge_and_project(attn: torch.Tensor, o_weight: torch.Tensor) -> torch.Tensor:
    """``[B, HQ, T, D] -> [B, T, HQ*D] -> [B, T, H]``, exactly as the module does it."""
    batch, _, tokens, _ = attn.shape
    merged = attn.transpose(1, 2).contiguous().reshape(batch, tokens, -1)
    return F.linear(merged, o_weight)

# -- the gated (SwiGLU) MLP block ------------------------------------------


def attention_projection_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    groups = _check_attention(q, k, v, o_weight)
    scale = 1.0 / math.sqrt(q.shape[-1])

    # `enable_gqa=True` broadcasts the KV heads inside the kernel; here the
    # expansion is written out, because this spelling exists to state the
    # mathematics rather than to avoid the memory.
    key = k.repeat_interleave(groups, dim=1)
    value = v.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q.float(), key.float().transpose(-2, -1)) * scale
    tokens, kv_tokens = q.shape[2], key.shape[2]
    causal = torch.ones(tokens, kv_tokens, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    attn = torch.matmul(weights, value)
    return _merge_and_project(attn, o_weight)

def attention_projection_forward_production(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    """The exact branch ``transformers.integrations.sdpa_attention`` takes here.

    ``attn_mask=None`` with ``is_causal=True`` is what the model passes when the
    causal pattern needs no explicit mask, and ``enable_gqa=True`` is what it
    passes when the KV heads can be broadcast rather than materialized. Both are
    read off the harvest, not assumed.
    """
    _check_attention(q, k, v, o_weight)
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=True,
    )
    return _merge_and_project(attn, o_weight)

# -- residual add plus RMSNorm ---------------------------------------------


def gated_mlp_forward_ref(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] != gate_weight.shape[-1]:
        raise ValueError(
            f"x's last dim {x.shape[-1]} must match gate_weight's {gate_weight.shape[-1]}"
        )
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape [I, H]")
    if down_weight.shape != (gate_weight.shape[1], gate_weight.shape[0]):
        raise ValueError(
            f"down_weight must be [H, I] = "
            f"{(gate_weight.shape[1], gate_weight.shape[0])}, got {tuple(down_weight.shape)}"
        )

    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = F.silu(gate.float()) * up.float()
    hidden = hidden.to(x.dtype)
    return F.linear(hidden, down_weight)

def gated_mlp_forward_model_dtype(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """The spelling ``Qwen3MLP.forward`` actually executes: no float32 upcast.

    Not the declared reference -- kept so the verification can report what the
    upcast costs, rather than leaving the two contracts silently different.
    """
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)

def residual_rms_norm_forward_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x + residual
    rstd = torch.rsqrt(summed.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = (summed * rstd.to(summed.dtype)) * weight
    return normalized, summed

def residual_rms_norm_runtime_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The residual add plus PyTorch's fused RMSNorm.

    The definition above spells the normalization out in primitives so AtenIR
    can lower it; ``F.rms_norm`` computes the same thing in one kernel, and is
    what the eager baseline is timed through. The add stays separate -- fusing it
    into the norm is exactly the optimization this task asks a candidate to
    find, so the baseline must not have it for free.

    ``summed`` is returned by both spellings, and it is the same tensor the
    normalization consumed, so a caller gets it at no extra cost.
    """
    summed = x + residual
    normalized = torch.nn.functional.rms_norm(
        summed, (x.shape[-1],), weight=weight, eps=eps
    )
    return normalized, summed


__all__ = [
    "attention_projection_forward_production",
    "attention_projection_forward_ref",
    "gated_mlp_forward_model_dtype",
    "gated_mlp_forward_ref",
    "residual_rms_norm_forward_ref",
    "residual_rms_norm_runtime_ref",
]
