"""Site ``residual_rmsnorm``: a residual add and the RMSNorm that follows it.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling (bitwise the model's own).
"""

from .task import op

__all__ = ["op"]
