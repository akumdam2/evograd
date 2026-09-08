"""Forward references for Llama-3-8B's attention-plus-projection boundary.

The mathematics is shared with Qwen3-0.6B and lives once in
``benchmark.topdown.common.level2_references``. It is re-exported here under
the names this model's declaration points at, so the symbol a candidate is
generated against belongs to this task rather than to another model's.
"""

from evograd.benchmark.topdown.common.level2_references import (
    attention_projection_forward_production as llama3_attention_forward_production,
    attention_projection_forward_ref as llama3_attention_forward_ref,
)

__all__ = ["llama3_attention_forward_production", "llama3_attention_forward_ref"]
