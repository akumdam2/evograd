"""Site ``moe_experts``: the routed SwiGLU experts, after the router has chosen.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived layer.
"""

from .task import op

__all__ = ["op"]
