"""Reviewed Liger GeGLU autograd-pair adapter."""

import importlib

import torch
import torch.distributed.tensor  # noqa: F401


def make_liger_geglu_autograd_pair_fns():
    module = importlib.import_module("liger_kernel.ops.geglu")
    if not hasattr(module, "geglu_forward") or not hasattr(module, "geglu_backward"):
        raise ImportError("liger_kernel.ops.geglu lacks raw forward/backward")

    def forward_with_saved(a, b):
        a_saved, b_saved, output = module.geglu_forward(a.contiguous(), b.contiguous())
        return output, (a_saved, b_saved)

    def backward_from_saved(dout, saved):
        a_saved, b_saved = saved
        # Liger writes gradients into these buffers; clones keep repeated timing
        # calls pure with respect to the saved state.
        return module.geglu_backward(
            a_saved.clone().contiguous(),
            b_saved.clone().contiguous(),
            dout.contiguous(),
        )

    return forward_with_saved, backward_from_saved
