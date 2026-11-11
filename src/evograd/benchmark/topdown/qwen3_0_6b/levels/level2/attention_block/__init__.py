"""Qwen3's whole self-attention module, ``x -> out``, as one boundary.

task is the contract and reference defines it (the module's own arithmetic).
Whether an implementation satisfies it is decided in :mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
