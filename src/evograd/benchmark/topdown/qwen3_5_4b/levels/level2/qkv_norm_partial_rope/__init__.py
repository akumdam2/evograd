"""Site ``qkv_norm_partial_rope``: gated attention's projections, (1 + w) q/k norms and partial RoPE.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived block.
"""

from .task import op

__all__ = ["op"]
