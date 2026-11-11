"""Site ``moe_experts``: the routed SwiGLU experts, after the router has chosen.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling (Transformers' own experts kernel).
"""

from .task import op

__all__ = ["op"]
