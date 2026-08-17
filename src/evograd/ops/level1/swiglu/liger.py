"""Reviewed Liger SwiGLU autograd-pair adapter."""

import torch
import torch.distributed.tensor  # noqa: F401


def make_liger_swiglu_autograd_pair_fns():
    from liger_kernel.ops.swiglu import swiglu_backward, swiglu_forward

    def forward_with_saved(a, b):
        a_saved, b_saved, output = swiglu_forward(a, b)
        return output, (a_saved, b_saved)

    def backward_from_saved(dout, saved):
        a_saved, b_saved = saved
        return swiglu_backward(a_saved, b_saved, dout)

    return forward_with_saved, backward_from_saved
