"""Trusted PyTorch forward for generalized Jensen-Shannon divergence."""

import torch


def jsd_forward_ref(log_q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    beta = 0.5
    log_p = target.float()
    log_qf = log_q.float()
    p = log_p.exp()
    q = log_qf.exp()
    mixture = beta * p + (1.0 - beta) * q
    log_mixture = mixture.log()
    kl_p_m = (p * (log_p - log_mixture)).sum()
    kl_q_m = (q * (log_qf - log_mixture)).sum()
    return (beta * kl_p_m + (1.0 - beta) * kl_q_m) / log_q.shape[0]
