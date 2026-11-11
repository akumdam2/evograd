"""The LM head and the causal-LM loss, spelled as Transformers 5.16.1 computes them.

``Qwen3ForCausalLM.forward`` runs ``logits = self.lm_head(hidden_states)`` and
``ForCausalLMLoss`` then upcasts (``logits.float()``), flattens to
``[rows, vocab]`` and calls ``fixed_cross_entropy``, which is
``F.cross_entropy(source, target, ignore_index=-100, reduction="mean")`` when
no ``num_items_in_batch`` is passed. ``target`` here is the label tensor after
``ForCausalLMLoss``'s shift (padded with ``-100``, shifted by one, flattened),
so every sequence's last row is ignored.

``F.linear`` on the flattened ``[rows, hidden]`` rows is the module's GEMM on
``[B, T, hidden]``; the corpus' ``lm_head`` payloads replay the same way.
"""

import torch.nn.functional as F


def qwen3_lm_head_ce_forward(x, weight, target):
    return F.cross_entropy(F.linear(x, weight).float(), target, ignore_index=-100,
                           reduction="mean")


__all__ = ["qwen3_lm_head_ce_forward"]
