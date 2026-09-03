"""The primitives the fused Qwen operators are built from.

Nothing here implements an operator. The declarations live in
:mod:`evograd.ops.level1`, which owns the reusable contracts; this package owns
the *Qwen-specific* part -- which primitive feeds which fused operator, at what
shapes, and with what tolerance the canonical workload justifies.

The composition table is the discoverable form of that relationship:

    >>> from evograd.bench.workloads.qwen3.levels.level1 import COMPOSES_INTO
    >>> COMPOSES_INTO["rope"]
    ('qwen3_qkv_norm_rope',)
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
