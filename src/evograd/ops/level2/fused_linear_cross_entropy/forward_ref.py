"""Trusted PyTorch reference for fused linear + cross entropy.

This is the language-model head and its loss taken together: project the final
hidden states to vocabulary logits, then reduce them to a mean cross-entropy.

Writing it as one operator is the whole point. Materializing the logits costs
``rows * vocab`` elements — 8192 x 128256 in bfloat16 is 2.1 GB, and the
backward needs a gradient of the same size — so the fusion that never forms them
is worth far more than the arithmetic it saves. Liger reports this kernel as its
largest memory win, and it is what ``apply_liger_kernel_to_llama`` enables by
default in place of the plain cross-entropy kernel.

The reference deliberately *does* materialize the logits: it is the definition
the candidate must match, not an implementation to imitate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_linear_cross_entropy_forward_ref(x, weight, target):
    """Mean cross entropy of ``x @ weight.T`` against hard labels ``target``.

    ``x`` is ``[rows, hidden]``, ``weight`` is ``[vocab, hidden]`` (the
    ``nn.Linear`` orientation, matching Liger and HuggingFace), ``target`` is
    ``[rows]`` of int64 class indices. Returns a scalar.

    The projection accumulates in float32 and the loss is computed in float32,
    which is what every production implementation does — a bfloat16 logsumexp
    over a 128k vocabulary loses too much to be a usable reference.
    """
    logits = torch.matmul(x.float(), weight.float().t())
    return F.cross_entropy(logits, target, reduction="mean")
