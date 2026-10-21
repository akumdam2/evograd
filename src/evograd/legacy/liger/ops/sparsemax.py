"""Reviewed Liger sparsemax autograd-pair adapter."""

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def make_liger_sparsemax_autograd_pair_fns():
    from liger_kernel.ops.sparsemax import (
        _sparsemax_backward,
        _sparsemax_forward,
    )

    def forward_with_saved(x):
        output, flat_output = _sparsemax_forward(x.contiguous(), -1)
        return output, (flat_output,)

    def backward_from_saved(dout, saved):
        (flat_output,) = saved
        return _sparsemax_backward(dout.contiguous(), flat_output, -1)

    return forward_with_saved, backward_from_saved
