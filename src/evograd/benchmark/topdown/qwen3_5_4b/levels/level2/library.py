"""Library reference lines for the Qwen3.5 Level-2 sites: FLA.

What a production stack would put at each site, as reviewed pairs over the
libraries' own kernels (:mod:`evograd.ops.library_pairs`):

==================  ======  ==========================================================
site                name    library code
==================  ======  ==========================================================
gated_delta_rule    fla     ``chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)``
gdn_norm_out        fla     ``FusedRMSNormGated`` + the same ``F.linear`` out_proj
==================  ======  ==========================================================

The other sites have none: the library offers only pieces of them (FLA's
short convolution), and a pair around a piece would mostly measure
hand-written torch.

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


__all__ = ["fla_gated_delta_rule", "fla_gdn_norm_out"]
