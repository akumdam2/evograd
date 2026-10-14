"""References for Qwen3.5's Gated DeltaNet input stage.

Everything ``Qwen3_5GatedDeltaNet.forward`` does before the delta rule::

    x -> in_proj_qkv, in_proj_z, in_proj_b, in_proj_a   four GEMMs
      -> conv1d over q|k|v + SiLU                        depthwise, causal, k=4
      -> split q, k, v                                   v stays a view of the conv output
      -> beta = sigmoid(b)
         g = -exp(A_log) * softplus(a + dt_bias)         (float32)
      -> q, k repeated to the value-head count

There is **no** per-key-head deinterleave: ``in_proj_qkv``
lays its output out as ``[q (HK*D) | k (HK*D) | v (HV*D)]`` -- exactly the
convolution's channel order -- and ``z``, ``b``, ``a`` have projections of their
own. ``HK`` appears in no tensor shape and is recovered from the widths:
``C = 2 * HK * D + VD`` with ``VD = HV * D``.

Production is the module's forward with each submodule call replaced by its
tensor equivalent, in the module's operation order (the input gradient is a sum
of four GEMMs' contributions and autograd accumulates them in graph order). The
causal convolution is the torch ``causal_conv1d_fn`` reached through
``__wrapped__``, so an importable ``causal_conv1d`` cannot redirect it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import hf_modules


def _heads(w_qkv, w_z, A_log):
    conv, value_dim = w_qkv.shape[0], w_z.shape[0]
    hv = A_log.shape[0]
    d = value_dim // hv
    hk = (conv - value_dim) // (2 * d) if d else 0
    if d <= 0 or hk <= 0 or hv * d != value_dim or 2 * hk * d + value_dim != conv or hv % hk:
        raise ValueError(f"widths C={conv}, VD={value_dim}, HV={hv} do not describe a Gated "
                         "DeltaNet with Dk == Dv")
    return hk, hv, d


def gdn_in_proj_forward_ref(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
    """The float32 oracle: same layouts, no intermediate rounding except where
    the outputs are cast (the convolution output once, before the split)."""
    hk, hv, d = _heads(w_qkv, w_z, A_log)
    batch, tokens = x.shape[0], x.shape[1]
    xf = x.float()
    mixed = (xf @ w_qkv.float().transpose(0, 1)).transpose(1, 2)
    taps = conv_w.shape[1]
    padded = F.pad(mixed, (taps - 1, 0))
    conv = torch.zeros(mixed.shape, dtype=torch.float32, device=mixed.device)
    for j in range(taps):
        conv = conv + conv_w[:, j].float()[None, :, None] * padded[:, :, j : j + tokens]
    # Cast before the transpose, so ``v`` comes out as the same view of the
    # convolution output the model hands on.
    conv = F.silu(conv.contiguous()).to(x.dtype).transpose(1, 2)
    key_dim = hk * d
    q, k, v = torch.split(conv, [key_dim, key_dim, hv * d], dim=-1)
    q = q.reshape(batch, tokens, -1, d)
    k = k.reshape(batch, tokens, -1, d)
    v = v.reshape(batch, tokens, -1, d)
    z = (xf @ w_z.float().transpose(0, 1)).reshape(batch, tokens, -1, d)
    b = xf @ w_b.float().transpose(0, 1)
    a = xf @ w_a.float().transpose(0, 1)
    beta = torch.sigmoid(b).to(x.dtype)
    g = -A_log.float().exp() * F.softplus(a + dt_bias.float())
    ratio = hv // hk
    if ratio > 1:
        q = q.repeat_interleave(ratio, dim=2)
        k = k.repeat_interleave(ratio, dim=2)
    return q, k, v, z.to(x.dtype), beta, g


def gdn_in_proj_forward_production(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
    """``Qwen3_5GatedDeltaNet.forward`` up to the delta rule, cache-free, mask-free."""
    modeling = hf_modules.modeling
    hk, hv, d = _heads(w_qkv, w_z, A_log)
    batch_size, seq_len = x.shape[0], x.shape[1]
    mixed_qkv = F.linear(x, w_qkv)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = F.linear(x, w_z)
    z = z.reshape(batch_size, seq_len, -1, d)
    b = F.linear(x, w_b)
    a = F.linear(x, w_a)
    mixed_qkv = modeling.causal_conv1d_fn.__wrapped__(mixed_qkv, conv_w, None, activation="silu")
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [hk * d, hk * d, hv * d], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, d)
    key = key.reshape(batch_size, seq_len, -1, d)
    value = value.reshape(batch_size, seq_len, -1, d)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    if hv // hk > 1:
        query = query.repeat_interleave(hv // hk, dim=2)
        key = key.repeat_interleave(hv // hk, dim=2)
    return query, key, value, z, beta, g


__all__ = ["gdn_in_proj_forward_production", "gdn_in_proj_forward_ref"]
