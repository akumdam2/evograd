"""``PreLayerNorm(TriangleAttention)``: AlphaFold 3's triangle self-attention.

The residual add is outside. Each node is its own task key because the two agree
in the shaped arguments a site dispatches on, up to the keep mask's orientation
(``docs/AF3.md`` section 11). ``gather_async`` is a no-op at ``sp_size`` 1 and is
omitted. ``Attend``'s own dropout is off for every provider.

At the ending node ``y`` is a transposed view, so the bias GEMM on it goes through
:func:`.contracts.projection`. ``pack_one`` copies it to rows after that, even at a
batch of one where a reshape would stay a view, so attention's biased GEMMs take
the fused path at both nodes.
"""

from __future__ import annotations

import torch.nn.functional as F

from evograd.benchmark.topdown.megafold.contracts import (
    LAYER_NORM_EPS,
    attention,
    contract,
    projection,
)
from evograd.opdecl import Active, Inactive


def _forward(ending, x, ln_weight, ln_bias, bias_weight, q_weight, q_bias, kv_weight, gate_weight,
             gate_bias, out_weight, mask, keep, eps):
    y = F.layer_norm(x, (x.shape[-1],), ln_weight, ln_bias, eps)
    if ending:
        y = y.permute(0, 2, 1, 3)
    b, rows, n, c = y.shape
    heads = bias_weight.shape[0]
    bias = projection(y.to(bias_weight.dtype), bias_weight).permute(0, 3, 1, 2)
    bias = bias[:, None].expand(b, rows, heads, rows, n).flatten(0, 1)
    key_mask = mask[:, None].expand(b, rows, n).flatten(0, 1)
    out = attention(y.reshape(b * rows, n, c).contiguous(), key_mask, bias, heads, q_weight, q_bias,
                    kv_weight, gate_weight, gate_bias, out_weight).reshape(b, rows, n, c)
    if ending:
        out = out.permute(0, 2, 1, 3)
    return out * keep


def starting_forward(x, ln_weight, ln_bias, bias_weight, q_weight, q_bias, kv_weight, gate_weight,
                     gate_bias, out_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(False, x, ln_weight, ln_bias, bias_weight, q_weight, q_bias, kv_weight, gate_weight,
                    gate_bias, out_weight, mask, keep, eps)


def ending_forward(x, ln_weight, ln_bias, bias_weight, q_weight, q_bias, kv_weight, gate_weight,
                   gate_bias, out_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(True, x, ln_weight, ln_bias, bias_weight, q_weight, q_bias, kv_weight, gate_weight,
                    gate_bias, out_weight, mask, keep, eps)


def declare(node: str):
    keep = "[B, 1, N, C]" if node == "starting" else "[B, N, 1, C]"
    orientation = "row" if node == "starting" else "column"
    transpose = ("" if node == "starting" else
                 "Transpose y's two sequence axes first, and the result's back at the end. ")
    return contract(
        f"megafold_triangle_attention_{node}", family="attention",
        forward=f"{__name__}:{node}_forward",
        dims=("B", "N", "C", "A", "I", "W"),
        args=(Active("x", "[B, N, N, C]", dtype="float32", note="the pair representation, residual excluded"),
              Active("ln_weight", "[C]", dtype="float32"), Active("ln_bias", "[C]", dtype="float32"),
              Active("bias_weight", "[A, C]", note="A = heads"),
              Active("q_weight", "[I, C]", note="I = heads * head width"), Active("q_bias", "[I]"),
              Active("kv_weight", "[W, C]", note="W = 2 * I"),
              Active("gate_weight", "[I, C]"), Active("gate_bias", "[I]"),
              Active("out_weight", "[C, I]"),
              Inactive("mask", "[B, N]", dtype="bool", note="True = attend to this key"),
              Inactive("keep", keep, note=f"the scaled {orientation}-dropout mask MegaFold drew"),
              Inactive("eps", default=LAYER_NORM_EPS)),
        output=Active("out", "[B, N, N, C]"),
        parameter_args=("ln_weight", "ln_bias", "bias_weight", "q_weight", "q_bias", "kv_weight",
                        "gate_weight", "gate_bias", "out_weight"),
        forward_semantics=(
            f"MegaFold's triangle attention, {node} node. y = LayerNorm(x; ln_weight, ln_bias) in "
            f"float32, cast to the case dtype. {transpose}Each row of y attends over its own N keys: "
            "the bias per head is y @ bias_weight^T, shared by every row; q, k, v are y @ q_weight^T + "
            "q_bias and the two halves of y @ kv_weight^T, split into A heads; attention is MegaFold's "
            "Attend default path with that bias and key mask, with a float32 softmax cast back; the "
            "merged heads are multiplied by sigmoid(y @ gate_weight^T + gate_bias) and projected by "
            "out_weight. out = that * keep. Every step but the LayerNorm and the softmax runs in the "
            "case dtype."),
        backward_semantics=(
            "Return dx and the gradient of every weight and bias, each in its input's dtype, as "
            "autograd differentiates the forward above."))
