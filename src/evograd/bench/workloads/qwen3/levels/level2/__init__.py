"""The four fused operators one Qwen3 decoder layer decomposes into.

The declarations live in :mod:`evograd.ops.level2` (and, for the residual
fusion, in ``ops.level2.fused_add_rms_norm``), which own the contracts and the
reference implementations. What this package owns is everything Qwen-specific
about them: the shapes the canonical workload actually calls them at, how often,
the tolerance those shapes justify, and the negative controls that show the
tolerance still rejects a wrong kernel.

``DECLARATIONS`` maps each module here to the ``OPS`` name it calibrates, so the
correspondence can be looked up rather than inferred from filenames.
"""

#: Module in this package -> the operator declaration it calibrates.
DECLARATIONS = {
    "qkv_norm_rope": "qwen3_qkv_norm_rope",
    "attention": "qwen3_attention",
    "swiglu_mlp": "qwen3_swiglu_mlp",
    "residual_rmsnorm": "fused_add_rms_norm",
}

#: The level-2 declarations this workload exercises, by their ``OPS`` names.
OPERATORS = tuple(DECLARATIONS.values())

__all__ = ["DECLARATIONS", "OPERATORS"]
