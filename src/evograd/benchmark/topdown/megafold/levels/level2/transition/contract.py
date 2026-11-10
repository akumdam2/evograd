"""``FusedTransition.forward`` with its LayerNorm: the SwiGLU feed-forward of every region.

The residual add is outside. ``include_ln=False`` (the diffusion transformer's) is a
different argument list and stays stock. Leading axes are flattened, so one contract
serves the pair, MSA and single representations alike: every GEMM here is biasless,
whose path does not depend on rank.
"""

from __future__ import annotations

import torch.nn.functional as F

from evograd.benchmark.topdown.megafold.contracts import LAYER_NORM_EPS, contract, layernorm_linear
from evograd.opdecl import Active, Inactive


def transition_forward(x, weight, bias, linear_weight, out_weight, eps=LAYER_NORM_EPS):
    first, gates = layernorm_linear(x, weight, bias, linear_weight, None, eps).chunk(2, dim=-1)
    return F.linear(F.silu(gates) * first, out_weight)


def declare():
    return contract(
        "megafold_transition", family="mlp", forward=f"{__name__}:transition_forward",
        dims=("M", "K", "W", "I"),
        args=(Active("x", "[M, K]", dtype="float32", note="leading axes flattened"),
              Active("weight", "[K]", dtype="float32", note="LayerNorm gamma"),
              Active("bias", "[K]", dtype="float32", note="LayerNorm beta"),
              Active("linear_weight", "[K, W]", note="MegaFold's [K, N] layout, W = 2 * I"),
              Active("out_weight", "[K, I]"),
              Inactive("eps", default=LAYER_NORM_EPS)),
        output=Active("out", "[M, K]"),
        parameter_args=("weight", "bias", "linear_weight", "out_weight"),
        forward_semantics=(
            "MegaFold's transition. h = LayerNorm(x; weight, bias) in float32, cast to the case dtype, "
            "@ linear_weight. first, gates = chunk(h, 2) over the last axis. out = (silu(gates) * first) "
            "@ out_weight^T. Everything but the LayerNorm runs in the case dtype, the SiLU included."),
        backward_semantics=("Return dx and the gradient of every weight and bias, each in its input's "
                            "dtype, as autograd differentiates the forward above."),
        fixtures=(dict(M=64, K=16, W=64, I=32), dict(M=129, K=32, W=64, I=32)))
