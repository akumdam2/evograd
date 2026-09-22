"""Reference for the causal masked softmax over a float32 score tensor.

``causal_softmax_forward_ref`` is both the oracle and the timed spelling:
``torch.softmax`` is already one fused kernel, so there is no faster eager
form to time against. Row ``m`` keeps keys ``n <= m``; strictly-future keys are
set to ``-inf`` before the softmax and therefore carry probability zero and
gradient zero. The softmax runs in float32 and the result is cast to bfloat16 -- the model
dtype the value product consumes -- so the cast is inside this contract, not
the next one. The contract is the bf16 decomposition's: float32 scores in,
bfloat16 probabilities out, whatever dtype the caller's model stores.
"""

import torch


def causal_softmax_forward_ref(s: torch.Tensor) -> torch.Tensor:
    if s.ndim != 4:
        raise ValueError("s must be [B, H, T, T]")
    tokens, kv_tokens = s.shape[-2], s.shape[-1]
    if tokens != kv_tokens:
        raise ValueError(f"s must be square in its last two dims, got {tuple(s.shape)}")
    causal = torch.ones(tokens, kv_tokens, dtype=torch.bool, device=s.device).tril()
    probabilities = torch.softmax(s.float().masked_fill(~causal, float("-inf")), dim=-1)
    return probabilities.to(torch.bfloat16)
