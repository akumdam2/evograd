"""The residual add and the RMSNorm that follows it, as one boundary.

task is the contract, reference defines it, and capture derives this
site's case from the canonical layer artifact. What a candidate must *satisfy*
is here; whether one does is decided in
:mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
