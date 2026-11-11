"""References for a residual add and the (1 + w) RMSNorm that follows it.

Shared with every hybrid model (:mod:`....common.hybrid.residual_norm`); two
outputs: the normalized value (for the MLP or the next token mixer) and the sum
itself (the residual the next add takes).
"""

from __future__ import annotations

from .....common.hybrid.residual_norm import (
    forward_production as residual_rmsnorm_forward_production,
    forward_ref as residual_rmsnorm_forward_ref,
)

__all__ = ["residual_rmsnorm_forward_production", "residual_rmsnorm_forward_ref"]
