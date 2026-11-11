"""Site ``gdn_in_proj``: the Gated DeltaNet's input stage (four projections, conv, gates, q/k repeat).

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling; the real invocations are the verification capture's
(:mod:`evograd.verification.integrations.qwen3_5_4b.capture`).
"""

from .task import op

__all__ = ["op"]
