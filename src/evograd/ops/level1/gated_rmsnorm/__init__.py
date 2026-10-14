"""Operator declaration: RMSNorm gated by ``silu(z)``.

``y = (norm(x) * w) * silu(z)`` with the casts of the reference implementation
-- see :mod:`.forward_ref`. The weight is a plain gain (initialized to ones),
unlike ``gemma_rmsnorm``'s offset.
"""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_CORRECTNESS = tuple(
    Workload(dims=dict(rows=rows, hidden=hidden), dtype=dtype)
    for rows, hidden, dtype in (
        (8, 64, "float32"),
        (17, 128, "float32"),
        (64, 128, "bfloat16"),
        (33, 96, "bfloat16"),
    )
)


def make_gated_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + hidden * 7 + 1)
    weight = (1.0 + 0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(dtype)
    return {
        "x": torch.randn((rows, hidden), device=device, dtype=dtype),
        "z": torch.randn((rows, hidden), device=device, dtype=dtype),
        "weight": weight,
        "eps": 1e-6,
        "dy": torch.randn((rows, hidden), device=device, dtype=dtype),
    }



def _fla_factory():
    """FLA's ``FusedRMSNormGated`` (``LayerNormGatedFunction``, RMS, swish gate)."""
    from evograd.ops.library_pairs import fla_rms_norm_gated_pair

    return fla_rms_norm_gated_pair(activation="swish")


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
#:   dz                      0        0         0     1     1     0      nan
#:   y                       0        0         0     1     1     0      nan
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"hidden": 128, "rows": 64},
    reduction_dims={"dweight": ('rows',)},
    result_dims={"dweight": ('hidden',), "dx": ('rows', 'hidden'), "dz": ('rows', 'hidden'), "y": ('rows', 'hidden')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="gated_rmsnorm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.gated_rmsnorm.forward_ref:gated_rmsnorm_forward_ref",
    dims=("rows", "hidden"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("z", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Gated RMSNorm over the last axis, with these casts in this order: "
        "h = float32(x); h = h * rsqrt(mean(h^2, -1, keepdim=True) + eps); "
        "h = weight * h.to(x.dtype) (the product is in x's dtype); "
        "y = (h * silu(float32(z))).to(x.dtype). The weight is a gain "
        "(initialized to one). Do not call F.rms_norm or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return dx, dz, dweight IN THIS ORDER. With r = rsqrt(mean(x^2) + eps), "
        "xhat = x * r, s = silu(z) and n = weight * xhat: dz = dy * n * "
        "silu'(z) where silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z))); "
        "dweight = sum over rows of dy * s * xhat; with g = dy * s * weight, "
        "dx = r * (g - xhat * mean(g * xhat, -1, keepdim=True)). Each gradient in "
        "its input's dtype, every reduction accumulated in float32."
    ),
    correctness=_CORRECTNESS,
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={
        "fla": make_pair_baseline(_fla_factory, ("x", "z", "weight", "eps"), sidecar="fla")},
    make_inputs=make_gated_rmsnorm_inputs,
)
