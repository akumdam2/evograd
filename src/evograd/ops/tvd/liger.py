"""Reviewed Liger total-variation-distance autograd-pair adapter."""

import importlib

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass

_MODULES = (
    "liger_kernel.ops.tvd",
    "liger_kernel.ops.tv_distance",
    "liger_kernel.ops.tvd_loss",
    "liger_kernel.ops.total_variation_distance",
    "liger_kernel.ops.total_variation_distance_loss",
)


def _load():
    last_error = None
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
            if hasattr(module, "tv_distance_forward_triton") and hasattr(
                module, "tvd_backward_triton"
            ):
                return module
        except Exception as exc:
            last_error = exc
    raise ImportError("Could not import Liger TVD raw ops") from last_error


def make_liger_tvd_autograd_pair_fns():
    module = _load()

    def forward_with_saved(p, q):
        output, grads = module.tv_distance_forward_triton(
            p.contiguous(), q.contiguous(), None, "batchmean", -100, False
        )
        return output, (grads,)

    def backward_from_saved(dout, saved):
        (grads,) = saved
        dp = module.tvd_backward_triton(dout.contiguous(), grads)
        return dp, -dp

    return forward_with_saved, backward_from_saved
