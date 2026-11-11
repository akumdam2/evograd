"""The gate/up projections, SwiGLU and the down projection, as one boundary.

task is the contract and reference defines it. What a candidate must
*satisfy* is here; whether one does is decided in :mod:`evograd.verification`.
"""

from .task import op

__all__ = ["op"]
