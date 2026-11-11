"""One Llama-3.2-1B Level-2 boundary: its contract and reference.

``task`` is the contract and ``reference`` defines it. Judging an
implementation of it belongs to :mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
