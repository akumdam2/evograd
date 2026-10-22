"""Reviewed Liger squared-ReLU autograd-pair adapter."""

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def make_liger_relu_squared_autograd_pair_fns():
    from liger_kernel.ops.relu_squared import (
        relu_squared_backward,
        relu_squared_forward,
    )

    def forward_with_saved(x):
        x = x.contiguous()
        return relu_squared_forward(x), (x,)

    def backward_from_saved(dout, saved):
        (x,) = saved
        return relu_squared_backward(x.contiguous(), dout.contiguous())

    return forward_with_saved, backward_from_saved
