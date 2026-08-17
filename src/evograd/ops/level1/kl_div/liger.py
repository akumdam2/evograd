"""Reviewed Liger batchmean KL-divergence autograd-pair adapter."""

import inspect

import torch
import torch.distributed.tensor  # noqa: F401


def _accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def make_liger_kl_div_autograd_pair_fns():
    from liger_kernel.ops.kl_div import kldiv_backward_triton, kldiv_forward_triton

    def forward_with_saved(y_pred, y_true):
        kwargs = {"log_target": False, "reduction": "batchmean"}
        if _accepts(kldiv_forward_triton, "eps"):
            kwargs["eps"] = torch.finfo(torch.float32).tiny
        loss = kldiv_forward_triton(
            y_pred.contiguous(), y_true.contiguous(), **kwargs
        )
        return loss.to(y_pred.dtype), (y_true,)

    def backward_from_saved(dloss, saved):
        (y_true,) = saved
        kwargs = {"log_target": False}
        if _accepts(kldiv_backward_triton, "in_place"):
            kwargs["in_place"] = False
        if _accepts(kldiv_backward_triton, "inplace"):
            kwargs["inplace"] = False
        output = torch.empty_like(y_true)
        output = kldiv_backward_triton(
            y_true, dloss.contiguous().clone(), output, **kwargs
        )
        return (output / y_true.shape[0]).to(y_true.dtype)

    return forward_with_saved, backward_from_saved
