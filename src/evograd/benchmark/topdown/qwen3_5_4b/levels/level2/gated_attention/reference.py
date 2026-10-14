"""References for gated attention's core: causal GQA SDPA, the sigmoid gate, o_proj.

Shared with every hybrid model (:mod:`....common.hybrid.gated_attention`):
``y = o_proj(SDPA(q, k, v) * sigmoid(gate))``, scale ``D^-0.5``; production is
the mask-free causal ``sdpa`` branch Transformers takes (``enable_gqa``), the
oracle materializes float32 scores.
"""

from __future__ import annotations

from .....common.hybrid.gated_attention import (
    forward_production as gated_attention_forward_production,
    forward_ref as gated_attention_forward_ref,
)

__all__ = ["gated_attention_forward_production", "gated_attention_forward_ref"]
