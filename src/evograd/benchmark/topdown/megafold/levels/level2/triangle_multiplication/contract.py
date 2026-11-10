"""``TriangleMultiplication.forward``: the pair update of AlphaFold 2's Algorithms 11 and 12.

The residual add is outside. Each mix is its own task key because outgoing and
incoming agree in every shaped argument, which is all a site dispatches on
(``docs/AF3.md`` section 11). ``scatter`` and ``gather`` are no-ops at ``sp_size``
1 and are omitted. ``forward`` spells the GLU in primitives, as the Level-1 ``glu``
task does, because a seed cannot lower ``F.glu``; ``runtime_forward`` calls it as
MegaFold does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.benchmark.topdown.megafold.contracts import LAYER_NORM_EPS, contract, layernorm_linear
from evograd.opdecl import Active, Inactive

OUTGOING = "... i k d, ... j k d -> ... i j d"
INCOMING = "... k i d, ... k j d -> ... i j d"
EQUATIONS = {"outgoing": OUTGOING, "incoming": INCOMING}


def _glu(x):
    first, gates = x.chunk(2, dim=-1)
    return first * gates.float().sigmoid().to(x.dtype)


def _runtime_glu(x):
    return F.glu(x, dim=-1)


def _forward(glu, equation, x, ln_weight, ln_bias, linear_weight, linear_bias, out_ln_weight,
             out_ln_bias, out_linear_weight, mask, keep, eps):
    combined = layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias, eps)
    split = combined.shape[-1] // 5 * 4
    left_right, gate = combined[..., :split], combined[..., split:].sigmoid()
    left, right = glu(left_right).chunk(2, dim=-1)
    pair = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
    mixed = torch.einsum(equation, left * pair, right * pair)
    out = layernorm_linear(mixed.float(), out_ln_weight, out_ln_bias, out_linear_weight, None, eps)
    return out * keep * gate


def outgoing_forward(x, ln_weight, ln_bias, linear_weight, linear_bias, out_ln_weight, out_ln_bias,
                     out_linear_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(_glu, OUTGOING, x, ln_weight, ln_bias, linear_weight, linear_bias,
                    out_ln_weight, out_ln_bias, out_linear_weight, mask, keep, eps)


def incoming_forward(x, ln_weight, ln_bias, linear_weight, linear_bias, out_ln_weight, out_ln_bias,
                     out_linear_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(_glu, INCOMING, x, ln_weight, ln_bias, linear_weight, linear_bias,
                    out_ln_weight, out_ln_bias, out_linear_weight, mask, keep, eps)


def outgoing_runtime_forward(x, ln_weight, ln_bias, linear_weight, linear_bias, out_ln_weight,
                             out_ln_bias, out_linear_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(_runtime_glu, OUTGOING, x, ln_weight, ln_bias, linear_weight,
                    linear_bias, out_ln_weight, out_ln_bias, out_linear_weight, mask, keep, eps)


def incoming_runtime_forward(x, ln_weight, ln_bias, linear_weight, linear_bias, out_ln_weight,
                             out_ln_bias, out_linear_weight, mask, keep, eps=LAYER_NORM_EPS):
    return _forward(_runtime_glu, INCOMING, x, ln_weight, ln_bias, linear_weight,
                    linear_bias, out_ln_weight, out_ln_bias, out_linear_weight, mask, keep, eps)


def declare(mix: str):
    return contract(
        f"megafold_triangle_multiplication_{mix}", family="pairwise",
        forward=f"{__name__}:{mix}_forward", runtime_forward=f"{__name__}:{mix}_runtime_forward",
        dims=("B", "N", "C", "G"),
        args=(Active("x", "[B, N, N, C]", dtype="float32", note="the pair representation, residual excluded"),
              Active("ln_weight", "[C]", dtype="float32"), Active("ln_bias", "[C]", dtype="float32"),
              Active("linear_weight", "[C, G]", note="MegaFold's [K, N] layout, G = 5 * C"),
              Active("linear_bias", "[G]"),
              Active("out_ln_weight", "[C]", dtype="float32"),
              Active("out_ln_bias", "[C]", dtype="float32"),
              Active("out_linear_weight", "[C, C]"),
              Inactive("mask", "[B, N]", dtype="bool", note="True = a real token"),
              Inactive("keep", "[B, 1, N, C]", note="the scaled row-dropout mask MegaFold drew"),
              Inactive("eps", default=LAYER_NORM_EPS)),
        output=Active("out", "[B, N, N, C]"),
        parameter_args=("ln_weight", "ln_bias", "linear_weight", "linear_bias", "out_ln_weight",
                        "out_ln_bias", "out_linear_weight"),
        forward_semantics=(
            f"MegaFold's triangle multiplication, {mix} mix. combined = LayerNorm(x; ln_weight, ln_bias) "
            "in float32, cast to the case dtype, @ linear_weight + linear_bias. Split the last axis at "
            "4C: gate = sigmoid(combined[..., 4C:]); left, right = chunk(glu(combined[..., :4C]), 2). "
            "Zero left and right where the pair mask mask[i] & mask[j] is False. mixed = einsum("
            f"'{EQUATIONS[mix]}', left, right) in the case dtype. out = LayerNorm(mixed; out_ln_weight, "
            "out_ln_bias) in float32, cast to the case dtype, @ out_linear_weight, then * keep * gate. "
            "Every step but the two LayerNorms runs in the case dtype."),
        backward_semantics=(
            "Return dx, dln_weight, dln_bias, dlinear_weight, dlinear_bias, dout_ln_weight, "
            "dout_ln_bias, dout_linear_weight, each in its input's dtype, as autograd differentiates "
            "the forward above."))
