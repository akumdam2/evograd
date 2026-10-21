"""Operator declaration: zero-centred RMSNorm, ``y = norm(x) * (1 + w)``.

A separate primitive from ``rmsnorm`` on purpose. The weight is an offset from
one rather than a gain (a fresh model stores zeros), and the cast back to the
input dtype happens after the scale rather than before it. A kernel written for
one is wrong for the other, by the whole weight in the first case and by a
rounding step in the second.
"""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import dtype_for

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_CORRECTNESS = tuple(
    Workload(dims=dict(rows=rows, hidden=hidden), dtype=dtype)
    for rows, hidden, dtype in (
        (8, 64, "float32"),
        (17, 128, "float32"),
        (32, 256, "bfloat16"),
        (64, 512, "bfloat16"),
    )
)


def make_gemma_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + hidden)
    # Zero-centred, as the stored parameter is: a weight drawn around 1 would be
    # a gain of 2 here.
    weight = (0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(dtype)
    return {
        "x": torch.randn((rows, hidden), device=device, dtype=dtype),
        "weight": weight,
        "eps": 1e-6,
        "dy": torch.randn((rows, hidden), device=device, dtype=dtype),
    }


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
#:   dweight                 0        0         0     1     1     0      nan
#:   dx                      0        0         0     1     1     0      nan
#:   y                       0        0         0     1     1     0      nan
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"hidden": 512, "rows": 64},
    reduction_dims={"dweight": ('rows',)},
    result_dims={"dweight": ('hidden',), "dx": ('rows', 'hidden'), "y": ('rows', 'hidden')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="gemma_rmsnorm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.gemma_rmsnorm.forward_ref:gemma_rmsnorm_forward_ref",
    dims=("rows", "hidden"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Zero-centred RMSNorm over the last axis of x [rows, hidden], entirely "
        "in float32: xhat = x * rsqrt(mean(x^2, -1, keepdim=True) + eps); "
        "y = xhat * (1 + weight), then cast y to x's dtype. weight is an offset "
        "from one: a weight of zero is the identity scale. The cast happens "
        "after the scale, not before it. Do not call F.rms_norm or autograd in "
        "the generated math."
    ),
    backward_semantics=(
        "Return dx, dweight IN THIS ORDER. With r = rsqrt(mean(x^2) + eps), "
        "xhat = x * r and g = dy * (1 + weight): dx = r * (g - xhat * "
        "mean(g * xhat, -1, keepdim=True)) in x's dtype; dweight = sum over rows "
        "of dy * xhat in weight's dtype. Accumulate every reduction in float32."
    ),
    correctness=_CORRECTNESS,
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    make_inputs=make_gemma_rmsnorm_inputs,
)
