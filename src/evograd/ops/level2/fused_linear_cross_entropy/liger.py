"""Reviewed Liger fused-linear-cross-entropy autograd-pair adapter.

This kernel splits the work unusually: the *forward* computes the gradients and
stashes them, and the backward only rescales them by the incoming
``grad_output``. That is not an implementation detail we can hide — it is the
reason the fusion saves memory, since the logits never exist as a tensor.

Two consequences for how this operator must be measured:

* A backward-only timing is meaningless here. In real training cross entropy is
  the last layer, so ``grad_output`` is exactly 1.0 and Liger skips the rescale
  entirely (``fused_linear_cross_entropy_backward`` early-outs on
  ``torch.equal(grad_output, 1.0)``). The full forward+backward step is the only
  honest measurement, which is what the suite report uses.
* ``fused_linear_cross_entropy_backward`` rescales ``grad_input`` and
  ``grad_weight`` **in place**. Every timing path here re-runs the forward per
  repetition, so each backward sees a fresh buffer; a caller that reused saved
  state across two backward calls would double-scale.

``bias`` is None throughout: the declaration has no bias, matching Llama's
``lm_head``, which is bias-free.
"""


def make_liger_flce_autograd_pair_fns():
    from liger_kernel.ops.fused_linear_cross_entropy import (
        fused_linear_cross_entropy_backward,
        fused_linear_cross_entropy_forward,
    )

    def forward_with_saved(x, weight, target):
        # Liger gates gradient computation on ``requires_grad``: it reads
        # ``_input.requires_grad`` into the kernel's ``HAS_GRADIENTS`` constant
        # and checks ``weight.requires_grad`` before allocating ``grad_weight``.
        # The autograd-pair protocol hands over plain tensors carrying no
        # autograd state, so without this the forward returns an all-zero
        # ``grad_input`` and a ``None`` ``grad_weight`` while still computing
        # the correct loss — a baseline that looks fast because it never
        # computed the gradients at all. Detached views share storage (no copy)
        # and leave the caller's tensors untouched, which the fair-bench
        # protocol's no-input-mutation rule requires.
        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        (
            loss,
            _z_loss,
            _token_accuracy,
            _predicted_tokens,
            grad_input,
            grad_weight,
            _grad_bias,
        ) = fused_linear_cross_entropy_forward(
            x,
            weight,
            target,
            ce_weight=None,
            bias=None,
            ignore_index=-100,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction="mean",
        )
        return loss, (grad_input, grad_weight)

    def backward_from_saved(dloss, saved):
        grad_input, grad_weight = saved
        dx, dweight, _dbias = fused_linear_cross_entropy_backward(
            dloss, grad_input, grad_weight, None
        )
        return dx, dweight

    return forward_with_saved, backward_from_saved
