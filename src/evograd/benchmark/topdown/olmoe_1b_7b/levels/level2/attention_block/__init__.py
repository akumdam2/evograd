"""Site ``attention_block``: ``OlmoeAttention.forward`` whole -- projections,
full-width q/k norms, RoPE, causal SDPA and ``o_proj``.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling (bitwise the module's own).
"""

from .task import op

__all__ = ["op"]
