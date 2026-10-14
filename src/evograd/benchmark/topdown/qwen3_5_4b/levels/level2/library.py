"""Library reference lines for the Qwen3.5 Level-2 sites: FLA and Liger.

What a production stack would put at each site, as reviewed pairs over the
libraries' own kernels (:mod:`evograd.ops.library_pairs` for FLA and the fused
residual norm):

==================  ======  ==========================================================
site                name    library code
==================  ======  ==========================================================
gated_delta_rule    fla     ``chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)``
gdn_norm_out        fla     ``FusedRMSNormGated`` + the same ``F.linear`` out_proj
residual_rmsnorm    liger   ``LigerFusedAddRMSNorm(offset=1.0, casting_mode="gemma")``
swiglu_mlp          liger   ``F.linear`` gate/up, Liger's SiLU-mul kernels, ``F.linear``
                            down -- what ``LigerSwiGLUMLP`` runs
==================  ======  ==========================================================

``gdn_in_proj``, ``qkv_norm_partial_rope`` and ``gated_attention`` have none:
the libraries offer only pieces of them (FLA's short convolution, Liger's RoPE),
and a pair around a piece would mostly measure hand-written torch.

FLA is a sidecar outside the environment (:mod:`evograd.opdecl.sidecar`); it is
only ever importable in its own baseline's child process. A library baseline
goes through the same correctness gate as a candidate and is never given wider
tolerances.
"""

from __future__ import annotations


def fla_gated_delta_rule():
    """q, k, v ``[B, T, HV, D]``, g the float32 log decay, beta post-sigmoid --
    FLA's own layout and gate convention, so nothing is adapted."""
    from evograd.ops.library_pairs import fla_gated_delta_rule_pair

    return fla_gated_delta_rule_pair()


def fla_gdn_norm_out():
    """Gated RMSNorm per value head in FLA, then ``F.linear`` with w_out (the model's)."""
    import torch.nn.functional as F

    from evograd.ops.library_pairs import fla_rms_norm_gated_pair

    norm_forward, norm_backward = fla_rms_norm_gated_pair(activation="swish")

    def forward_with_saved(o, z, w_norm, w_out, eps):
        core, norm_saved = norm_forward(o, z, w_norm, eps)
        flat = core.reshape(core.shape[0], core.shape[1], -1)
        return F.linear(flat, w_out), (norm_saved, flat)

    def backward_from_saved(dy, saved, w_out):
        norm_saved, flat = saved
        dflat = dy.matmul(w_out)
        dw_out = dy.reshape(-1, dy.shape[-1]).t().matmul(flat.reshape(-1, flat.shape[-1]))
        (ctx,) = norm_saved
        do, dz, dw_norm = norm_backward(dflat.reshape(ctx.x_shape_og), norm_saved)
        return do, dz, dw_norm, dw_out

    return forward_with_saved, backward_from_saved


def liger_residual_rmsnorm():
    """``summed = branch + residual``, then the Gemma ``(1 + w)`` norm in float32."""
    from evograd.ops.library_pairs import liger_fused_add_rms_norm_pair

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


__all__ = ["fla_gated_delta_rule", "fla_gdn_norm_out", "liger_residual_rmsnorm",
           "liger_swiglu_mlp"]
