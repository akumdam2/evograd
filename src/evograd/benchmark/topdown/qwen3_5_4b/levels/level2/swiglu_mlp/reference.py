"""References for Qwen3.5's dense SwiGLU MLP.

The mathematics is every gated MLP's (:mod:`....common.level2_references`): the
oracle accumulates ``silu(gate) * up`` in float32 and casts once; production is
what ``Qwen3_5MLP.forward`` executes, entirely in the model dtype.
"""

from __future__ import annotations

from .....common.level2_references import (
    gated_mlp_forward_model_dtype as swiglu_mlp_forward_production,
    gated_mlp_forward_ref as swiglu_mlp_forward_ref,
)

__all__ = ["swiglu_mlp_forward_production", "swiglu_mlp_forward_ref"]
