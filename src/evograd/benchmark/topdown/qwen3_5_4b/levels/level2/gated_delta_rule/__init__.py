"""Site ``gated_delta_rule``: the chunked gated delta rule with in-kernel q/k L2 normalization.

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived block.
"""

from .task import op

__all__ = ["op"]
