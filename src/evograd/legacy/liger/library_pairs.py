"""Liger library pairs (legacy: not part of the benchmark protocol)."""

from __future__ import annotations

import importlib

from evograd.ops._common import FunctionContext


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
