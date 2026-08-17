"""Trusted PyTorch forward for mean-reduced hard-label cross entropy."""

import torch


def cross_entropy_forward_ref(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(
        logits.float(), target, reduction="mean", ignore_index=-100
    )
    return loss.to(logits.dtype)
