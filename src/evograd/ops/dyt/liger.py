"""Reviewed Liger DyT autograd-pair adapter."""

import importlib

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def make_liger_dyt_autograd_pair_fns():
    module = importlib.import_module("liger_kernel.ops.dyt")
    forward = getattr(module, "liger_dyt_fwd")
    backward = getattr(module, "liger_dyt_bwd")

    def forward_with_saved(x, alpha, gamma, beta):
        x, alpha, gamma = x.contiguous(), alpha.contiguous(), gamma.contiguous()
        beta = beta.contiguous() if beta is not None else None
        beta_saved = beta if beta is not None else torch.empty(
            (0,), device=x.device, dtype=x.dtype
        )
        return forward(x, alpha, gamma, beta), (x, alpha, gamma, beta_saved)

    def backward_from_saved(dout, saved):
        x, alpha, gamma, beta_saved = saved
        beta = None if beta_saved.numel() == 0 else beta_saved
        dx, dalpha, dgamma, dbeta = backward(
            dout.contiguous(), x, alpha, gamma, beta
        )
        return dx, dalpha.view_as(alpha), dgamma, dbeta

    return forward_with_saved, backward_from_saved
