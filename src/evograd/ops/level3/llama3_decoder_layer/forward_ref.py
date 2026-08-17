"""Trusted PyTorch reference for one Llama-3 decoder layer.

    x -> RMSNorm -> Q/K/V proj -> RoPE -> GQA attention -> O proj -> +x
      -> RMSNorm -> SwiGLU MLP -> +residual

This is a level-3 task: a whole architectural block, measured the same way the
individual kernels are, so a candidate can be rewarded for optimizing across the
boundaries between them rather than each one in isolation.

Two deliberate exclusions, both consequences of what an ``OpDecl`` can express
and both stated here rather than left for a reader to discover:

* **No KV cache.** This models the *training* forward pass. A cache is
  per-step mutable state, which the declaration model excludes, and it does not
  exist in a training step anyway.
* **No attention-weight output.** The block returns one tensor.

The attention itself uses ``F.scaled_dot_product_attention``. That is a
correctness-neutral but benchmark-critical choice: the eager and
``torch.compile`` baselines both measure *this* function, and an explicit
``softmax(q @ k.T)`` would materialize a ``[batch, heads, tokens, tokens]``
score matrix that neither HuggingFace nor Inductor would ever produce. Beating
that would say nothing. It also keeps activation memory linear in the token
count instead of quadratic, which is what makes the larger workloads fit on a
40 GB card at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rms_norm(x, weight, eps):
    """RMSNorm with float32 statistics, as every production implementation does."""
    x32 = x.float()
    rrms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return ((x32 * rrms).to(x.dtype)) * weight


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    x32 = x.float()
    return (x32 * cos[None, None].float() + _rotate_half(x32) * sin[None, None].float()).to(
        x.dtype
    )


def _llama3_decoder_layer(
    rms_norm,
    x,
    input_norm_weight,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    post_norm_weight,
    gate_weight,
    up_weight,
    down_weight,
    cos,
    sin,
    eps=1e-5,
):
    """One decoder layer. ``x`` is ``[batch, tokens, hidden]``.

    Projection weights follow the ``nn.Linear`` orientation ``[out, in]``, so
    they drop straight into a HuggingFace or Liger-patched layer without
    transposition. ``cos``/``sin`` are ``[tokens, head_dim]`` with duplicated
    halves.

    The head counts are recovered from the weight shapes rather than passed in:
    a declared dimension that appears in no tensor shape could not be recovered
    from the tensors at deploy time, so the contract carries ``q_out`` and
    ``kv_out`` as widths and divides.
    """
    batch, tokens, hidden = x.shape
    head_dim = cos.shape[-1]
    n_heads = q_weight.shape[0] // head_dim
    n_kv_heads = k_weight.shape[0] // head_dim
    if q_weight.shape[0] % head_dim or k_weight.shape[0] % head_dim:
        raise ValueError("q_out and kv_out must be whole multiples of head_dim")
    if n_heads % n_kv_heads:
        raise ValueError(
            f"GQA needs n_heads divisible by n_kv_heads, got {n_heads} and {n_kv_heads}"
        )

    # ── attention ────────────────────────────────────────────────────────
    h = rms_norm(x, input_norm_weight, eps)
    q = F.linear(h, q_weight).view(batch, tokens, n_heads, head_dim).transpose(1, 2)
    k = F.linear(h, k_weight).view(batch, tokens, n_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(h, v_weight).view(batch, tokens, n_kv_heads, head_dim).transpose(1, 2)

    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    # Grouped-query attention: replicate each kv head across its query group.
    repeat = n_heads // n_kv_heads
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(batch, tokens, n_heads * head_dim)
    x = x + F.linear(attn, o_weight)

    # ── MLP ──────────────────────────────────────────────────────────────
    h = rms_norm(x, post_norm_weight, eps)
    gate = F.linear(h, gate_weight)
    up = F.linear(h, up_weight)
    # SiLU in float32: the activation is where a bfloat16 MLP loses the most.
    swiglu = (F.silu(gate.float()) * up.float()).to(h.dtype)
    return x + F.linear(swiglu, down_weight)


def _rms_norm_fused(x, weight, eps):
    """PyTorch's fused RMSNorm — one kernel instead of pow/mean/rsqrt/mul."""
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def llama3_decoder_layer_forward_ref(*args, **kwargs):
    """The definition: normalization spelled out in primitives.

    Unchanged in signature and behaviour — AtenIR lowers this, and the oracle
    differentiates it.
    """
    return _llama3_decoder_layer(_rms_norm, *args, **kwargs)


def llama3_decoder_layer_runtime_ref(*args, **kwargs):
    """The same layer with PyTorch's fused RMSNorm, for timing the baseline.

    Sharing one body keeps the two from drifting apart: only the normalization
    differs, which is the whole point of the distinction.
    """
    return _llama3_decoder_layer(_rms_norm_fused, *args, **kwargs)
