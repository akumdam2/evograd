"""Site ``swiglu_mlp``: the dense SwiGLU MLP.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling; the real invocations are the verification capture's
(:mod:`evograd.verification.integrations.qwen3_5_4b.capture`).
"""

from .task import op

__all__ = ["op"]
