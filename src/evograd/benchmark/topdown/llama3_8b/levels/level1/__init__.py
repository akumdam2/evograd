"""The primitives the fused Llama operators are built from.

Nothing here implements an operator. The declarations live in
:mod:`evograd.ops.level1`, which owns the reusable contracts; this package owns
the *Llama-specific* part -- which primitive feeds which fused operator, at what
shapes, and with what tolerance the canonical workload justifies.

The composition table is the discoverable form of that relationship:

    >>> from evograd.benchmark.topdown.llama3_8b.levels.level1 import COMPOSES_INTO
    >>> COMPOSES_INTO["rope"]
    ('llama3_qkv_rope',)

Note the one edge Qwen3 has and this does not: ``rmsnorm`` reaches only
``fused_add_rms_norm`` here, because Llama-3's projection boundary contains no
per-head query/key norm.
"""

#: The level-1 declarations this workload exercises, by their ``OPS`` names.
#: Kept here rather than imported so ``python -m ...levels.level1.mapping``
#: does not load the module twice.
OPERATORS = ("linear_no_bias", "rmsnorm", "rope", "swiglu",
             "causal_gqa_attention", "cross_entropy")

__all__ = ["COMPOSES_INTO", "OPERATORS"]


def __getattr__(name: str):
    if name == "COMPOSES_INTO":
        from .mapping import COMPOSES_INTO

        return COMPOSES_INTO
    raise AttributeError(name)
