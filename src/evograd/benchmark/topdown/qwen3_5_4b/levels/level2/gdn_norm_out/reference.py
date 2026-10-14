"""References for the Gated DeltaNet's output stage: gated RMSNorm, then out_proj.

Shared with every hybrid model (:mod:`....common.hybrid.gated_norm`):
``Qwen3_5RMSNormGated``. The norm runs per
value head (rows of ``D``); production follows the module cast for cast and
projects in the model dtype, the oracle stays in float32.
"""

from __future__ import annotations

from .....common.hybrid.gated_norm import (
    forward_production as gdn_norm_out_forward_production,
    forward_ref as gdn_norm_out_forward_ref,
)

__all__ = ["gdn_norm_out_forward_production", "gdn_norm_out_forward_ref"]
