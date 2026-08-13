"""Trusted PyTorch reference for an AlphaFold3 single-representation block.

    x -> LayerNorm -> Q/K/V projections
      -> attention with a trainable pair bias and a residue mask
      -> output projection -> +x
      -> LayerNorm -> SwiGLU transition -> +residual

This exercises all three kernel families MegaFold (arXiv:2506.20686) implements
for AlphaFold3 training — EvoFlash-3D attention, fused LayerNorm+Linear, and the
fused SwiGLU transition — inside one level-3 task.

**What this is not.** A full AlphaFold3 PairformerStack block maps
``(single, pair) -> (single', pair')``: two tensors in, two out. An ``OpDecl``
has a single ``Active`` output, so that block cannot be declared here. This is
the single-representation update *conditioned on* the pair representation, with
``pair_bias`` as a trainable input. The pair path is still fully exercised in
the backward — ``d_pair_bias`` is returned, and it reduces over the MSA axis —
but the triangle-multiplicative update that would produce ``pair'`` is out of
scope. Saying so is the point; a task named for the whole block while computing
half of it would be worse than the restriction it hides.

Unlike the Llama block, attention here is written out explicitly rather than
through ``scaled_dot_product_attention``. That is deliberate: ``pair_bias``
requires a gradient, and SDPA falls back to its math backend whenever the
attention bias does, so routing through it would add a dispatch layer without
changing what runs. The explicit form is also what MegaFold's own reference
does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _layer_norm(x, weight, bias, eps):
    """LayerNorm over the channel axis with float32 statistics."""
    x32 = x.float()
    mean = x32.mean(-1, keepdim=True)
    var = x32.var(-1, unbiased=False, keepdim=True)
    normed = (x32 - mean) * torch.rsqrt(var + eps)
    return (normed.to(x.dtype)) * weight + bias


def af3_single_repr_block_forward_ref(
    x,
    ln1_weight,
    ln1_bias,
    q_weight,
    k_weight,
    v_weight,
    res_mask,
    pair_bias,
    out_weight,
    ln2_weight,
    ln2_bias,
    gate_weight,
    up_weight,
    down_weight,
    eps=1e-5,
):
    """One block. ``x`` is ``[batch, n_seq, n_res, channels]``.

    ``pair_bias`` is ``[batch, 1, heads, n_res, n_res]`` in float32 and is shared
    across the MSA axis, so its gradient sums over that axis. ``res_mask`` is
    additive: 0 keeps a key, a large negative value drops it.

    The head count and head dim are recovered from ``q_weight``'s fan-out and the
    declared head count rather than passed separately — a declared dimension that
    appears in no tensor shape could not be rebuilt from the tensors themselves
    at deployment time.
    """
    if x.ndim != 4:
        raise ValueError(f"expected [batch, n_seq, n_res, channels], got {tuple(x.shape)}")
    batch, n_seq, n_res, _channels = x.shape
    heads = pair_bias.shape[2]
    fan_out = q_weight.shape[-1]
    if fan_out % heads:
        raise ValueError(f"projection fan-out {fan_out} is not divisible by {heads} heads")
    head_dim = fan_out // heads

    # ── attention ────────────────────────────────────────────────────────
    h = _layer_norm(x, ln1_weight, ln1_bias, eps)

    def project(weight):
        # [B, S, N, C] @ [C, E] -> [B, S, N, H, D] -> [B, S, H, N, D]
        projected = torch.matmul(h, weight)
        return projected.view(batch, n_seq, n_res, heads, head_dim).transpose(-2, -3)

    q, k, v = project(q_weight), project(k_weight), project(v_weight)

    scores = torch.matmul(q * head_dim**-0.5, k.transpose(-1, -2))
    scores = scores + pair_bias + res_mask
    probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)

    attended = torch.matmul(probs, v)  # [B, S, H, N, D]
    attended = (
        attended.transpose(-2, -3).contiguous().view(batch, n_seq, n_res, fan_out)
    )
    x = x + torch.matmul(attended, out_weight)

    # ── transition ───────────────────────────────────────────────────────
    h = _layer_norm(x, ln2_weight, ln2_bias, eps)
    gate = torch.matmul(h, gate_weight)
    up = torch.matmul(h, up_weight)
    transition = (F.silu(gate.float()) * up.float()).to(x.dtype)
    return x + torch.matmul(transition, down_weight)
