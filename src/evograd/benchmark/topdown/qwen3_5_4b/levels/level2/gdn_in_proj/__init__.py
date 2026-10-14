"""Site ``gdn_in_proj``: the Gated DeltaNet's input stage (four projections, conv, gates, q/k repeat).

``task`` is the contract, ``reference`` the float32 oracle and the production
spelling, ``capture`` one real invocation from a captured or config-derived block.
"""

from .task import op

__all__ = ["op"]
