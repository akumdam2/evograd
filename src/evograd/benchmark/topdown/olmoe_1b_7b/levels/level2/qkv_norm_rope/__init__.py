"""Site ``qkv_norm_rope``: the attention prefix -- projections, full-width q/k norms, RoPE.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived layer.
"""

from .task import op

__all__ = ["op"]
