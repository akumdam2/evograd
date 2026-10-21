"""Liger pairs for the Qwen3.5 residual and MLP sites (legacy: not part of the benchmark protocol)."""

from __future__ import annotations


def liger_residual_rmsnorm():
    """``summed = branch + residual``, then the Gemma ``(1 + w)`` norm in float32."""
    from evograd.legacy.liger.library_pairs import liger_fused_add_rms_norm_pair

    return liger_fused_add_rms_norm_pair(offset=1.0, casting_mode="gemma")


def liger_swiglu_mlp():
    """The dense MLP as ``LigerSwiGLUMLP`` computes it: two ``F.linear``, Liger's
    fused ``silu(gate) * up`` (whose backward overwrites its own saved inputs --
    the pair's private copies, never a caller's tensor), one ``F.linear``."""
    import torch.nn.functional as F
    from liger_kernel.ops.swiglu import swiglu_backward, swiglu_forward

    def forward_with_saved(x, gate_weight, up_weight, down_weight):
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        a, b, hidden = swiglu_forward(gate, up)
        return F.linear(hidden, down_weight), (x, a, b, hidden)

    def backward_from_saved(dout, saved, gate_weight, up_weight, down_weight):
        x, a, b, hidden = saved
        rows = dout.reshape(-1, dout.shape[-1])
        dhidden = dout.matmul(down_weight)
        ddown = rows.t().matmul(hidden.reshape(-1, hidden.shape[-1]))
        dgate, dup = swiglu_backward(a, b, dhidden)
        dgate = dgate.reshape(dhidden.shape)
        dup = dup.reshape(dhidden.shape)
        flat_x = x.reshape(-1, x.shape[-1])
        dx = dgate.matmul(gate_weight) + dup.matmul(up_weight)
        dgate_weight = dgate.reshape(-1, dgate.shape[-1]).t().matmul(flat_x)
        dup_weight = dup.reshape(-1, dup.shape[-1]).t().matmul(flat_x)
        return dx, dgate_weight, dup_weight, ddown

    return forward_with_saved, backward_from_saved
