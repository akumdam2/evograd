"""Reviewed explicit-pair adapter for Liger's fused MoE autograd function."""

from evograd.ops._common import FunctionContext as _FunctionContext
from evograd.ops._common import make_pair_baseline


def _liger_factory():
    from liger_kernel.ops.fused_moe import LigerFusedMoEFunction

    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        ctx = _FunctionContext()
        output = LigerFusedMoEFunction.forward(
            ctx,
            x,
            gate_up_proj,
            down_proj,
            top_k_index,
            top_k_weights,
        )
        return output, (ctx,)

    def backward(dout, saved):
        (ctx,) = saved
        dx, dgate_up, ddown, _dindex, dtop_k_weights = (
            LigerFusedMoEFunction.backward(ctx, dout)
        )
        return dx, dgate_up, ddown, dtop_k_weights

    return forward, backward


measure_liger_baseline = make_pair_baseline(
    _liger_factory,
    ("x", "gate_up_proj", "down_proj", "top_k_index", "top_k_weights"),
)
