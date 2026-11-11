"""Site ``gated_delta_rule``: the chunked gated delta rule with in-kernel q/k L2 normalization.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling; the real invocations are the verification capture's
(:mod:`evograd.verification.integrations.qwen3_5_4b.capture`).
"""

from .task import op

__all__ = ["op"]
