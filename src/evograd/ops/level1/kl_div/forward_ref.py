"""Trusted PyTorch forward for batchmean KL divergence."""

import torch


def kl_div_forward_ref(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.kl_div(
        y_pred.float(), y_true.float(), reduction="batchmean", log_target=False
    )
    return loss.to(y_pred.dtype)
