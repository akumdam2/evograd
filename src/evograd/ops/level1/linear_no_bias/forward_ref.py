"""References for a biasless Linear layer: ``y = x @ weight.T``.

Every projection in a modern decoder is biasless -- Llama-3 and Qwen3 both set
``attention_bias=False`` and give their MLP and ``lm_head`` no bias either -- so
this, not the bias-carrying contract, is the GEMM those models actually run.

The distinction is not cosmetic. A bias-carrying task with a zero bias still
adds a broadcast add to the forward, a ``dbias`` row reduction to the backward,
and a third gradient to the contract. A candidate is then optimized against work
the model never does, and its speedup is measured against a baseline that pays
for it too.

``linear_no_bias_forward_ref`` is the oracle: the contraction in float32, cast
once. ``linear_no_bias_runtime_ref`` is ``F.linear(x, weight, None)`` -- what a
model runs and what the eager baseline is timed through.

``weight`` is ``[N, K]``, which is how ``nn.Linear`` stores it and how the
harvest observed it. That is deliberately *not* ``matmul``'s ``[K, N]``
interface: the two are different memory layouts for the same mathematics, and a
kernel written for one is not a kernel for the other.
"""

import torch
import torch.nn.functional as F


def _check(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2D tensors")
    if weight.shape[1] != x.shape[1]:
        raise ValueError(
            f"weight.shape[1] ({weight.shape[1]}) must match x.shape[1] ({x.shape[1]})"
        )


def linear_no_bias_forward_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` with the contraction accumulated in float32."""
    _check(x, weight)
    return torch.matmul(x.float(), weight.float().t()).to(x.dtype)


def linear_no_bias_runtime_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The exact call a model makes: ``F.linear`` with no bias at all.

    Not ``F.linear(x, weight, zeros)``. Passing ``None`` is what selects the
    kernel path without the bias epilogue, which is the point of the task.
    """
    _check(x, weight)
    return F.linear(x, weight, None)
