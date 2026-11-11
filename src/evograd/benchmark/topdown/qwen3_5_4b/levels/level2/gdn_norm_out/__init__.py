"""Site ``gdn_norm_out``: gated RMSNorm per value head plus the output projection.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling; the real invocations are the verification capture's
(:mod:`evograd.verification.integrations.qwen3_5_4b.capture`).
"""

from .task import op

__all__ = ["op"]
