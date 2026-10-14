"""Site ``swiglu_mlp``: the dense SwiGLU MLP.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived block.
"""

from .task import op

__all__ = ["op"]
