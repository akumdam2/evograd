"""The dtype each MegaFold result is computed through, per contract.

A stored dtype does not say how a value was computed. ``dx`` of the bf16
LayerNorm+Linear case is stored float32, but it is a bf16 GEMM's backward carried
through a float32 LayerNorm's, so bf16 rounding is in it. The mixed-precision
tolerance fix reads this table; no tolerance reads it yet, and no shared code does.

``THROUGH[task][result]`` is, for the bf16 case, the dtypes the result's value is
computed in, in execution order. A float32 case is float32 throughout, so it has
no entry. Stored dtypes are not repeated here: they come from the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

BF16, FP32 = "bfloat16", "float32"
#: A forward: LayerNorm in float32, GEMM and what follows in bf16.
NORM_THEN_GEMM = (FP32, BF16)
#: The gradient of a parameter of the leading LayerNorm, or of ``x``: the bf16 path
#: back from the output, then a float32 LayerNorm's backward.
GEMM_THEN_NORM = (BF16, FP32)
#: The gradient of a GEMM parameter: a bf16 backward all the way.
GEMM_ONLY = (BF16,)

THROUGH: dict[str, dict[str, tuple[str, ...]]] = {
    "megafold_layernorm_linear": {
        "out": NORM_THEN_GEMM, "dx": GEMM_THEN_NORM, "dweight": GEMM_THEN_NORM,
        "dbias": GEMM_THEN_NORM, "dlinear_weight": GEMM_ONLY},
    "megafold_evoattention": {
        # Every step but the softmax is bf16; the softmax is float32, cast back.
        "o": (BF16, FP32, BF16), "dq": (BF16, FP32, BF16), "dk": (BF16, FP32, BF16),
        "dv": GEMM_ONLY, "d_pair_bias": (BF16, FP32, BF16)},
}

_TRIANGLE_MULTIPLICATION = {
    "out": (FP32, BF16, FP32, BF16), "dx": (BF16, FP32, BF16, FP32),
    "dln_weight": (BF16, FP32, BF16, FP32), "dln_bias": (BF16, FP32, BF16, FP32),
    "dlinear_weight": (BF16, FP32, BF16), "dlinear_bias": (BF16, FP32, BF16),
    "dout_ln_weight": (BF16, FP32), "dout_ln_bias": (BF16, FP32), "dout_linear_weight": GEMM_ONLY}
THROUGH["megafold_triangle_multiplication_outgoing"] = _TRIANGLE_MULTIPLICATION
THROUGH["megafold_triangle_multiplication_incoming"] = _TRIANGLE_MULTIPLICATION

#: ``Attention``'s projections, from the bf16 output back. Whatever reaches them
#: through the attention weights passes the float32 softmax; the gate does not.
_ATTENTION = {"dq_weight": (BF16, FP32, BF16), "dq_bias": (BF16, FP32, BF16),
              "dkv_weight": (BF16, FP32, BF16), "dgate_weight": GEMM_ONLY, "dgate_bias": GEMM_ONLY,
              "dout_weight": GEMM_ONLY}
_TRIANGLE_ATTENTION = {
    "out": (FP32, BF16, FP32, BF16), "dx": (BF16, FP32, BF16, FP32),
    "dln_weight": (BF16, FP32, BF16, FP32), "dln_bias": (BF16, FP32, BF16, FP32),
    "dbias_weight": (BF16, FP32, BF16), **_ATTENTION}
THROUGH["megafold_triangle_attention_starting"] = _TRIANGLE_ATTENTION
THROUGH["megafold_triangle_attention_ending"] = _TRIANGLE_ATTENTION
THROUGH["megafold_transition"] = {
    "out": NORM_THEN_GEMM, "dx": GEMM_THEN_NORM, "dweight": GEMM_THEN_NORM, "dbias": GEMM_THEN_NORM,
    "dlinear_weight": GEMM_ONLY, "dout_weight": GEMM_ONLY}
#: Stored float32 and built on a bf16 product: the division by the float32 count
#: comes last in the forward and first in every backward.
THROUGH["megafold_outer_product_mean"] = {
    "out": (FP32, BF16, FP32), "dx": (FP32, BF16, FP32), "dweight": (FP32, BF16, FP32),
    "dbias": (FP32, BF16, FP32), "dlinear_weight": (FP32, BF16), "dpair_weight": (FP32, BF16),
    "dpair_bias": (FP32, BF16)}
THROUGH["megafold_msa_pair_weighted_averaging"] = {
    "out": (FP32, BF16, FP32, BF16), "dx": (BF16, FP32), "dln_weight": (BF16, FP32),
    "dln_bias": (BF16, FP32), "dlinear_weight": GEMM_ONLY, "dpair": (BF16, FP32, BF16, FP32),
    "dpair_ln_weight": (BF16, FP32, BF16, FP32), "dpair_ln_bias": (BF16, FP32, BF16, FP32),
    "dpair_linear_weight": (BF16, FP32, BF16), "dout_weight": GEMM_ONLY}


@dataclass(frozen=True)
class Result:
    """One result of one case: how it is stored and what it was computed through."""

    stored: str
    through: tuple[str, ...]

    @property
    def narrowest(self) -> str:
        return BF16 if BF16 in self.through else FP32


def result_precision(op, dtype: str) -> dict[str, Result]:
    """Every result of ``op`` at case dtype ``dtype``, output first."""

    def stored(arg) -> str:
        return arg.dtype or dtype

    results = {out.name: stored(out) for out in op.outputs}
    results.update({arg.grad_name: stored(arg) for arg in op.active_args()})
    if dtype == FP32:
        return {name: Result(store, (FP32,)) for name, store in results.items()}
    return {name: Result(store, THROUGH[op.name][name]) for name, store in results.items()}
