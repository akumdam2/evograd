"""MegaFold's fused SwiGLU Triton kernels as an evograd autograd pair.

MegaFold (``megafold/model/FusedTransition/fused_transition.py``) computes
``y = silu(right) * left`` on one tensor ``x = [left | right]`` and writes the
input gradient back into that same buffer. The declaration computes
``c = silu(a) * b`` on two tensors, so the adapter concatenates ``[b | a]``,
hands the kernel that buffer, and splits the gradient it writes into
``(da, db)``. Nothing numerical happens outside MegaFold's kernels; the glue
is one ``cat`` and one ``chunk`` per direction, and the timing reports it as
``declared_megafold_pair`` so a reader can see that glue was there.

MegaFold is a source checkout on ``PYTHONPATH``, not a package; without it the
factory raises and the baseline reports as unavailable, like Liger does.
"""

import torch


def make_megafold_swiglu_autograd_pair_fns():
    from megafold.model.FusedTransition.fused_transition import (
        swiglu_backward,
        swiglu_forward,
    )

    def forward_with_saved(a, b):
        x = torch.cat((b, a), dim=-1).contiguous()  # [left | right] = [b | a]
        x_saved, y = swiglu_forward(x)
        return y, (x_saved,)

    def backward_from_saved(dout, saved):
        (x_saved,) = saved
        dx = swiglu_backward(x_saved, dout.contiguous())  # in place into x_saved
        db, da = dx.chunk(2, dim=-1)
        return da.contiguous(), db.contiguous()

    return forward_with_saved, backward_from_saved
