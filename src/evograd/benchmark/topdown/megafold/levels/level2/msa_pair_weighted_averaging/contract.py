"""``MSAPairWeightedAveraging.forward``: AlphaFold 3's Algorithm 10.

The residual add is outside, and ``gather_async`` is a no-op at ``sp_size`` 1. The
softmax is float32 under autocast, and nothing casts it back until the einsum's own
autocast rule does, which is why the reference spells ``weights.to(values.dtype)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.benchmark.topdown.megafold.contracts import LAYER_NORM_EPS, contract, layernorm_linear
from evograd.opdecl import Active, Inactive


def msa_pair_weighted_averaging_forward(x, ln_weight, ln_bias, linear_weight, pair, pair_ln_weight,
                                        pair_ln_bias, pair_linear_weight, out_weight, mask, keep,
                                        eps=LAYER_NORM_EPS):
    bias = layernorm_linear(pair, pair_ln_weight, pair_ln_bias, pair_linear_weight, None,
                            eps).permute(0, 3, 1, 2)
    heads = pair_linear_weight.shape[1]
    hidden = layernorm_linear(x, ln_weight, ln_bias, linear_weight, None, eps)
    b, s, n, width = hidden.shape
    head_width = width // (2 * heads)
    # Indexed, not unpacked: unpacking a tensor is an unbind, which a seed cannot lower.
    halves = hidden.reshape(b, s, n, 2, heads, head_width).permute(3, 0, 4, 1, 2, 5)
    values, gates = halves[0], halves[1]
    bias = bias.masked_fill(~mask[:, None, None, :], -torch.finfo(bias.dtype).max)
    weights = bias.float().softmax(dim=-1)
    out = torch.einsum("b h i j, b h s j d -> b h s i d", weights.to(values.dtype), values)
    out = (out * gates.sigmoid()).permute(0, 2, 3, 1, 4).reshape(b, s, n, heads * head_width)
    return F.linear(out, out_weight) * keep


def declare():
    return contract(
        "megafold_msa_pair_weighted_averaging", family="attention",
        forward=f"{__name__}:msa_pair_weighted_averaging_forward",
        dims=("B", "S", "N", "M", "W", "I", "C", "A"),
        args=(Active("x", "[B, S, N, M]", dtype="float32", note="the MSA representation"),
              Active("ln_weight", "[M]", dtype="float32"), Active("ln_bias", "[M]", dtype="float32"),
              Active("linear_weight", "[M, W]", note="MegaFold's [K, N] layout, W = 2 * I"),
              Active("pair", "[B, N, N, C]", dtype="float32", note="the pair representation"),
              Active("pair_ln_weight", "[C]", dtype="float32"),
              Active("pair_ln_bias", "[C]", dtype="float32"),
              Active("pair_linear_weight", "[C, A]", note="MegaFold's [K, N] layout, A = heads"),
              Active("out_weight", "[M, I]", note="I = heads * head width"),
              Inactive("mask", "[B, N]", dtype="bool", note="True = attend to this token"),
              Inactive("keep", "[B, 1, N, M]", note="the scaled row-dropout mask MegaFold drew"),
              Inactive("eps", default=LAYER_NORM_EPS)),
        output=Active("out", "[B, S, N, M]"),
        parameter_args=("ln_weight", "ln_bias", "linear_weight", "pair_ln_weight", "pair_ln_bias",
                        "pair_linear_weight", "out_weight"),
        forward_semantics=(
            "MegaFold's MSA pair-weighted averaging. bias = LayerNorm(pair; pair_ln_weight, "
            "pair_ln_bias) in float32, cast to the case dtype, @ pair_linear_weight, heads moved to "
            "axis 1; columns with mask False are set to the case dtype's most negative finite value. "
            "weights = softmax(bias) over the last axis in float32, cast to the case dtype. hidden = "
            "LayerNorm(x; ln_weight, ln_bias) in float32, cast to the case dtype, @ linear_weight, "
            "reshaped to values and gates, each [B, A, S, N, head width]. out = einsum('b h i j, b h s "
            "j d -> b h s i d', weights, values) * sigmoid(gates), heads merged, @ out_weight^T, "
            "then * keep. Every step but the LayerNorms and the softmax runs in the case dtype."),
        backward_semantics=(
            "Return dx, dpair and the gradient of every weight and bias, each in its input's dtype, "
            "as autograd differentiates the forward above."),
        fixtures=(dict(B=1, S=6, N=8, M=16, W=24, I=12, C=8, A=2),
                  dict(B=1, S=3, N=5, M=8, W=32, I=16, C=8, A=4)))
