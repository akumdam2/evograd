"""Llama-3.2-1B's self-attention module as one Level-2 boundary.

``task`` is the contract and ``reference`` defines it. Judging an
implementation of it belongs to :mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
