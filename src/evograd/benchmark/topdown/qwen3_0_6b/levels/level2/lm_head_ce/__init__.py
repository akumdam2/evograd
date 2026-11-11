"""The LM head and the causal-LM cross entropy, as one boundary.

task is the contract and reference defines it (the two lines Transformers
runs). Whether an implementation satisfies it is decided in
:mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
