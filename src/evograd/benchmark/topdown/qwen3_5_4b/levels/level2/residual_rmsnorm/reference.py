"""References for the post-attention residual add and its (1 + w) RMSNorm.

Shared with every hybrid model (:mod:`....common.hybrid.residual_norm`); two
outputs: the normalized value (for the MLP) and the sum itself (the residual the
block adds back at its end).
"""

from __future__ import annotations

from .....common.hybrid.residual_norm import (
    forward_production as residual_rmsnorm_forward_production,
    forward_ref as residual_rmsnorm_forward_ref,
)

__all__ = ["residual_rmsnorm_forward_production", "residual_rmsnorm_forward_ref"]
