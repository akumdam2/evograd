"""The four fused operators one Llama-3 decoder layer decomposes into.

The declarations live in :mod:`evograd.ops.level2` (and, for the residual
fusion, in ``ops.level2.fused_add_rms_norm``), which own the contracts and the
reference implementations. What this package owns is everything Llama-specific
about them: the shapes the canonical workload actually calls them at, how often,
the tolerance those shapes justify, and the negative controls that show the
tolerance still rejects a wrong kernel.

``DECLARATIONS`` maps each module here to the ``OPS`` name it calibrates, so the
correspondence can be looked up rather than inferred from filenames.

**One of the four is Llama's own.** ``qkv_rope`` calibrates
``llama3_qkv_rope``, a declaration this workload introduced, because
``LlamaAttention`` has no per-head query/key RMSNorm and Qwen3's
``qwen3_qkv_norm_rope`` therefore describes a different computation. The other
three boundaries are dimensionally parameterized and shared -- their operator
names still carry a ``qwen3_`` prefix from the workload that first harvested
them, which is a rename this repository owes and not a claim about which model
runs them.

**Nothing here can run yet.** Every module in this package derives its
invocation from ``results/llama3-level4/layer16.pt``, which is produced by the
level-3 capture on a GPU, and from ``harvest/snapshot.json``, which is produced
by the harvest. Neither has been executed, so each entry point refuses by name
rather than substituting synthetic tensors.
"""

#: Module in this package -> the operator declaration it calibrates.
DECLARATIONS = {
    "qkv_rope": "llama3_qkv_rope",
    "attention": "qwen3_attention",
    "swiglu_mlp": "qwen3_swiglu_mlp",
    "residual_rmsnorm": "fused_add_rms_norm",
}

#: The level-2 declarations this workload exercises, by their ``OPS`` names.
OPERATORS = tuple(DECLARATIONS.values())

__all__ = ["DECLARATIONS", "OPERATORS"]
