"""``OuterProductMean.forward``: AlphaFold 3's Algorithm 9, the MSA's update to the pair track.

The residual add is outside, and ``reduce_scatter_sum`` is a no-op at ``sp_size`` 1.
The step always passes an MSA mask, so that is the branch modelled. The result is
**float32** under a bf16 autocast: a bf16 result is divided by the float32 count
``MSAModule.forward`` summed, outside the boundary, before its layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.benchmark.topdown.megafold.contracts import LAYER_NORM_EPS, contract, layernorm_linear
from evograd.opdecl import Active, Inactive

COUNT_EPS = 1e-5


def outer_product_mean_forward(x, weight, bias, linear_weight, pair_weight, pair_bias, mask, msa_mask,
                               msa_count, eps=LAYER_NORM_EPS, count_eps=COUNT_EPS):
    a, b = layernorm_linear(x, weight, bias, linear_weight, None, eps).chunk(2, dim=-1)
    a, b = a * msa_mask[..., None, None], b * msa_mask[..., None, None]
    outer = torch.einsum("b s i d, b s j e -> b i j d e", a, b).flatten(-2)
    pair_mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2))[..., None].to(outer.dtype)
    out = F.linear(outer * pair_mask, pair_weight, pair_bias)
    return out / msa_count[..., None, None, None].clamp(min=count_eps)


def declare():
    return contract(
        "megafold_outer_product_mean", family="pairwise",
        forward=f"{__name__}:outer_product_mean_forward",
        dims=("B", "S", "N", "M", "W", "P", "Q"),
        args=(Active("x", "[B, S, N, M]", dtype="float32", note="the MSA representation"),
              Active("weight", "[M]", dtype="float32", note="LayerNorm gamma"),
              Active("bias", "[M]", dtype="float32", note="LayerNorm beta"),
              Active("linear_weight", "[M, W]", note="MegaFold's [K, N] layout, W = 2 * hidden"),
              Active("pair_weight", "[P, Q]", note="Q = hidden * hidden"),
              Active("pair_bias", "[P]"),
              Inactive("mask", "[B, N]", dtype="bool", note="True = a real token"),
              Inactive("msa_mask", "[B, S]", note="1 = a real MSA row, in the case dtype"),
              Inactive("msa_count", "[B]", dtype="float32", note="how many MSA rows are real"),
              Inactive("eps", default=LAYER_NORM_EPS), Inactive("count_eps", default=COUNT_EPS)),
        output=Active("out", "[B, N, N, P]", dtype="float32"),
        parameter_args=("weight", "bias", "linear_weight", "pair_weight", "pair_bias"),
        forward_semantics=(
            "MegaFold's outer product mean. h = LayerNorm(x; weight, bias) in float32, cast to the case "
            "dtype, @ linear_weight; a, b = chunk(h, 2) over the last axis, each multiplied by msa_mask "
            "per MSA row. outer = einsum('b s i d, b s j e -> b i j d e', a, b) in the case dtype, the "
            "two last axes flattened, then multiplied by the pair mask mask[i] & mask[j]. p = outer @ "
            "pair_weight^T + pair_bias in the case dtype. out = p / max(msa_count, count_eps), which "
            "is float32 whatever the case dtype, because msa_count is float32."),
        backward_semantics=(
            "Return dx, dweight, dbias, dlinear_weight, dpair_weight, dpair_bias, each in its input's "
            "dtype, as autograd differentiates the forward above."))
