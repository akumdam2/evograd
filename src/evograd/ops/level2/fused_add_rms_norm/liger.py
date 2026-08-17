"""Reviewed Liger fused-add-RMSNorm autograd-pair adapter."""

import importlib

import torch

try:
    importlib.import_module("torch.distributed.tensor")
except Exception:
    pass


def make_liger_fused_add_rms_norm_autograd_pair_fns():
    module = importlib.import_module("liger_kernel.ops.fused_add_rms_norm")
    casting_mode_name = "gemma"
    offset = 0.0
    num_stages = 2

    def forward_with_saved(x, residual, weight, eps):
        output, summed, rstd, *_ = module.fused_add_rms_norm_forward(
            x.contiguous(),
            residual.contiguous(),
            weight.contiguous(),
            eps,
            offset,
            casting_mode_name,
        )
        return output, (summed, weight, rstd)

    def backward_from_saved(dout, saved, eps):
        del eps
        summed, weight, rstd = saved
        block_size, num_warps = module.calculate_settings(summed.shape[-1])
        casting_mode = module._str_to_casting_mode[casting_mode_name]
        d_s_out = torch.zeros_like(summed)
        return module.fused_add_rms_norm_backward(
            dout.contiguous(),
            d_s_out,
            summed.contiguous(),
            weight.contiguous(),
            rstd.contiguous(),
            offset,
            casting_mode,
            block_size,
            num_warps,
            num_stages,
            False,
        )

    return forward_with_saved, backward_from_saved
