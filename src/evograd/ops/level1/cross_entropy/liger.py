"""Reviewed Liger hard-label cross-entropy autograd-pair adapter."""

import torch
import torch.distributed.tensor  # noqa: F401


def make_liger_cross_entropy_autograd_pair_fns():
    from liger_kernel.ops.cross_entropy import (
        cross_entropy_backward,
        cross_entropy_forward,
    )

    def forward_with_saved(logits, target):
        work = logits.detach().clone(memory_format=torch.contiguous_format).contiguous()
        work.requires_grad_(True)
        loss, _z, _accuracy, _tokens, logits_grad = cross_entropy_forward(
            work,
            target.contiguous(),
            None,
            -100,
            0.0,
            0.0,
            "mean",
            None,
            False,
            False,
            False,
        )
        return loss, (logits_grad.detach(),)

    def backward_from_saved(dloss, saved):
        (saved_grad,) = saved
        work = saved_grad.clone(memory_format=torch.contiguous_format).contiguous()
        return cross_entropy_backward(work, dloss.contiguous())

    return forward_with_saved, backward_from_saved
