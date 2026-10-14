"""Operator declaration: row-wise L2 normalization (FLA's definition).

What a gated delta rule applies to its queries and keys before the recurrence.
Generic over rows and width.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_CORRECTNESS = tuple(
    Workload(dims=dict(rows=rows, cols=cols), dtype=dtype)
    for rows, cols, dtype in (
        (8, 64, "float32"),
        (17, 128, "float32"),
        (64, 128, "bfloat16"),
        (33, 96, "bfloat16"),
    )
)


def make_l2norm_inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "eps": 1e-6,
        "dy": torch.randn((rows, cols), device=device, dtype=dtype),
    }



def _fla_factory():
    """FLA's ``l2norm`` (``L2NormFunction``), the normalization its gated delta
    rule applies in kernel: ``x * rsqrt(sum(x^2) + eps)``, as here."""
    from evograd.ops.library_pairs import fla_l2norm_pair

    return fla_l2norm_pair()


# --- calibrated tolerances (scripts/qwen3_5_4b/calibration.py apply) ---
from evograd.opdecl.tolerance import ReductionScaledAtol  # noqa: E402

#: Calibrated for Qwen3.5-4B (run ``20261013_l1calib``; GH200, grid also on CPU):
#: ``op.forward`` against ``op.runtime_forward`` only -- never a library or
#: a candidate -- on correctness, correctness[cpu], observed;
#: four repeats each (production was deterministic). The bfloat16 base
#: atol is the smallest 1-2-5 step at or above both 1.5x the forward
#: outputs' grid need and one bfloat16 ulp of their magnitude; rtol 2e-2.
#: Multipliers are per dtype (in the hook): 1.5x that dtype's grid need,
#: rounded up (none below 1). Each gain is the smallest that carries the
#: bfloat16 multiplier over the real-size cases with the same 1.5x.
#: ``need`` is the atol required at the base rtol, in units of the base atol.
#:
#:   result           fp32 grid bf16 grid real need fp32x bf16x  gain margin@real
#:   dx                      0    0.189     0.556     1     1     0      1.8
#:   y                       0        0         0     1     1     0      nan
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.002, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"cols": 128, "rows": 64},
    reduction_dims={},
    result_dims={"dx": ('rows', 'cols'), "y": ('rows', 'cols')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="l2norm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.l2norm.forward_ref:l2norm_forward_ref",
    runtime_forward="evograd.ops.level1.l2norm.forward_ref:l2norm_forward_production",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"), Inactive("eps", default=1e-6)),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics=(
        "Row-wise L2 normalization over the last axis of x [rows, cols]: "
        "r = rsqrt(sum(x^2, axis=-1, keepdim=True) + eps); y = x * r. The eps is "
        "added to the sum of squares, inside the root, not to the norm. "
        "Accumulate the sum in float32 and return y in x's dtype. There is no "
        "weight. Do not call F.normalize, torch.linalg or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return dx: with r as above and y = x * r, "
        "dx = r * (dy - y * sum(dy * y, axis=-1, keepdim=True)), in x's dtype. "
        "Accumulate the row reduction in float32."
    ),
    correctness=_CORRECTNESS,
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={"fla": make_pair_baseline(_fla_factory, ("x", "eps"), sidecar="fla")},
    make_inputs=make_l2norm_inputs,
)
