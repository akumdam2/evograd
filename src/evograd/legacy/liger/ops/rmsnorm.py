"""Reviewed Liger RMSNorm autograd-pair adapter.

RMSNorm is the operator Liger patches into more models than any other (39 of
its 40 ``apply_liger_kernel_to_*`` entry points enable it), and a Llama decoder
layer runs it twice. It nevertheless had no baseline here before v1, so every
reported RMSNorm speedup was measured against eager PyTorch alone.

Fairness notes, following the same review checklist as the other adapters:

* ``casting_mode="llama"`` and ``offset=0.0`` are the settings
  ``apply_liger_kernel_to_llama`` installs. The ``gemma`` mode computes
  ``(1 + weight)`` and ``offset`` shifts gamma, so either would silently
  disagree with the declared forward reference.
* ``in_place=False``. The in-place variant overwrites the incoming gradient,
  which the final-report protocol rejects as input mutation.
* No host-device synchronization in the backward, and nothing the forward could
  have precomputed is recomputed there.
"""


_OFFSET = 0.0
_CASTING_MODE = "llama"


def make_liger_rmsnorm_autograd_pair_fns():
    from liger_kernel.ops.rms_norm import rms_norm_backward, rms_norm_forward

    def forward_with_saved(x, weight, eps):
        # liger 0.8.0: (X, W, eps, offset, casting_mode, row_mode)
        #           -> (Y, X_2d, RSTD, BLOCK_SIZE, num_warps, row_mode)
        # casting_mode is not returned, so the backward is given the same
        # literal rather than a value read back out of the forward.
        y, x_2d, rstd, block_size, num_warps, row_mode = rms_norm_forward(
            x, weight, eps, _OFFSET, _CASTING_MODE, None
        )
        return y, (x_2d, weight, rstd, block_size, num_warps, row_mode)

    def backward_from_saved(dy, saved):
        x_2d, weight, rstd, block_size, num_warps, row_mode = saved
        dy_2d = dy.view(-1, dy.shape[-1]).contiguous()
        dx, dweight = rms_norm_backward(
            dy_2d,
            x_2d,
            weight,
            rstd,
            _OFFSET,
            _CASTING_MODE,
            block_size,
            num_warps,
            False,  # in_place: never overwrite dy, the protocol rejects mutation
            row_mode,
        )
        return dx.view_as(dy), dweight

    return forward_with_saved, backward_from_saved
