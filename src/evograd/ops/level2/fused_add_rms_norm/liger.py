"""Reviewed Liger fused-add-RMSNorm autograd-pair adapter."""

import importlib

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
        # Liger's forward already computes and returns the summed residual --
        # its own autograd Function exposes it as a second output. The adapter
        # used to discard it and hand back only `output`, which made this
        # provider a *different operator* from the one being benchmarked.
        return (output, summed), (summed, weight, rstd)

    def backward_from_saved(output_grads, saved, eps):
        del eps
        dout, dsummed = output_grads
        summed, weight, rstd = saved
        block_size, num_warps = module.calculate_settings(summed.shape[-1])
        casting_mode = module._str_to_casting_mode[casting_mode_name]
        # `dS_out` is Liger's own name for the gradient arriving at the summed
        # residual; its kernel adds it into dX after the normalization backward
        # (`dX_row += dS_out_row`) and returns `dX, dX, dW`, which is exactly
        # this task's contract. The adapter used to pass zeros here, silently
        # dropping every gradient that reaches x and residual without going
        # through the norm.
        return module.fused_add_rms_norm_backward(
            dout.contiguous(),
            dsummed.contiguous(),
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
