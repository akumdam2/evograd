"""Reviewed library autograd pairs: Liger and Flash Linear Attention.

Each factory returns ``(forward_with_saved, backward_from_saved)`` over the
library's own ``torch.autograd.Function``, called directly with a
:class:`~evograd.ops._common.FunctionContext` -- so the pair runs exactly the
library's forward and backward, with the state it saves held in one object.
Layout adaptation (a task's ``[B, C, T]`` against a library's ``[B, T, C]``,
or a model-specific epilogue) belongs to the declaring task, not here.

FLA is a *sidecar* (:mod:`evograd.opdecl.sidecar`): its factories activate it
first, which succeeds only in a child process started for an FLA baseline.
Every import happens inside a factory, never at module import.
"""

from __future__ import annotations

import importlib

from evograd.ops._common import FunctionContext


def _fla(module: str):
    from evograd.opdecl.sidecar import activate

    activate("fla")
    return importlib.import_module(module)


# ── Liger ────────────────────────────────────────────────────────────────────


def liger_rms_norm_pair(*, offset: float, casting_mode: str):
    """``LigerRMSNormFunction``; ``in_place=False`` so dY is never overwritten
    (Liger's own Gemma patch makes the same choice)."""
    function = importlib.import_module("liger_kernel.ops.rms_norm").LigerRMSNormFunction

    def forward_with_saved(x, weight, eps):
        ctx = FunctionContext()
        y = function.forward(ctx, x, weight, eps, offset, casting_mode, False, None)
        return y.view(x.shape), (ctx,)

    def backward_from_saved(dy, saved):
        (ctx,) = saved
        dx, dweight, *_ = function.backward(ctx, dy)
        return dx.view(dy.shape), dweight

    return forward_with_saved, backward_from_saved


def liger_fused_add_rms_norm_pair(*, offset: float, casting_mode: str):
    """``LigerFusedAddRMSNormFunction``: ``(normed, summed)`` with
    ``summed = x + r``; its backward returns ``dX, dX, dW`` (dR is dX)."""
    function = importlib.import_module(
        "liger_kernel.ops.fused_add_rms_norm").LigerFusedAddRMSNormFunction

    def forward_with_saved(x, r, weight, eps):
        ctx = FunctionContext()
        y, summed = function.forward(ctx, x, r, weight, eps, offset, casting_mode, False)
        return (y.view(x.shape), summed.view(x.shape)), (ctx,)

    def backward_from_saved(output_grads, saved):
        (ctx,) = saved
        dy, dsummed = output_grads
        dx, dr, dweight, *_ = function.backward(ctx, dy, dsummed)
        return dx.view(dy.shape), dr.view(dy.shape), dweight

    return forward_with_saved, backward_from_saved


def _int32():
    import torch

    return torch.int32


# ── Flash Linear Attention (sidecar) ─────────────────────────────────────────


def fla_gated_delta_rule_pair(*, chunk_size: int = 64):
    """``ChunkGatedDeltaRuleFunction`` as ``chunk_gated_delta_rule(q, k, v, g,
    beta, scale=D^-0.5, use_qk_l2norm_in_kernel=True)`` calls it: q, k, v
    ``[B, T, H, D]``, g the log decay ``[B, T, H]`` (float32), beta post-sigmoid,
    no initial or final state. Returns ``dq, dk, dv, dg, dbeta``."""
    function = _fla("fla.ops.gated_delta_rule.chunk").ChunkGatedDeltaRuleFunction

    def forward_with_saved(q, k, v, g, beta):
        ctx = FunctionContext()
        o, _final = function.forward(
            ctx, q, k, v, g, beta, q.shape[-1] ** -0.5, None, False,
            False,       # state_v_first
            None, None,  # cu_seqlens, cu_seqlens_cpu
            True,        # use_qk_l2norm_in_kernel
            False, None, None,  # use_gate_in_kernel, A_log, dt_bias
            False, False,       # use_beta_sigmoid_in_kernel, allow_neg_eigval
            None, chunk_size,
        )
        return o, (ctx,)

    def backward_from_saved(do, saved):
        (ctx,) = saved
        dq, dk, dv, dg, dbeta, *_ = function.backward(ctx, do, None)
        return dq, dk, dv, dg, dbeta

    return forward_with_saved, backward_from_saved


def fla_rms_norm_gated_pair(*, activation: str = "swish"):
    """``LayerNormGatedFunction`` with ``is_rms_norm=True`` -- what
    ``FusedRMSNormGated`` runs: ``rmsnorm(x) * weight * act(z)`` over the last
    dim. Returns ``dx, dz, dweight``."""
    function = _fla("fla.modules.fused_norm_gate").LayerNormGatedFunction

    def forward_with_saved(x, z, weight, eps):
        ctx = FunctionContext()
        y = function.forward(ctx, x, z, weight, None, activation, None, eps, False, False, True)
        return y, (ctx,)

    def backward_from_saved(dy, saved):
        (ctx,) = saved
        dx, dz, dweight, *_ = function.backward(ctx, dy)
        return dx, dz, dweight

    return forward_with_saved, backward_from_saved


def fla_l2norm_pair():
    """``L2NormFunction``: ``x * rsqrt(sum(x^2, -1) + eps)``."""
    function = _fla("fla.modules.l2norm").L2NormFunction

    def forward_with_saved(x, eps):
        ctx = FunctionContext()
        return function.forward(ctx, x, eps, None), (ctx,)

    def backward_from_saved(dy, saved):
        (ctx,) = saved
        dx, *_ = function.backward(ctx, dy)
        return dx

    return forward_with_saved, backward_from_saved


def fla_causal_conv1d_pair():
    """FLA's Triton ``CausalConv1dFunction`` (``causal_conv1d(backend='triton')``):
    token-major ``x [B, T, C]``, weight ``[C, K]``, no bias, no state.
    ``activation`` is ``"silu"`` or ``None``. Returns ``dx, dweight``."""
    function = _fla("fla.modules.conv.triton.ops").CausalConv1dFunction

    def forward_with_saved(x, weight, activation):
        ctx = FunctionContext()
        y, _final = function.forward(ctx, x, weight, None, None, None, False, activation,
                                     None, None, None)
        return y, (ctx,)

    def backward_from_saved(dy, saved):
        (ctx,) = saved
        dx, dweight, *_ = function.backward(ctx, dy, None)
        return dx, dweight

    return forward_with_saved, backward_from_saved


__all__ = [
    "fla_causal_conv1d_pair",
    "fla_gated_delta_rule_pair",
    "fla_l2norm_pair",
    "fla_rms_norm_gated_pair",
    "liger_fused_add_rms_norm_pair",
    "liger_rms_norm_pair",
]
