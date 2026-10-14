"""Operator declaration: Gated DeltaNet gating, two outputs.

``g = -exp(A_log) * softplus(a + dt_bias)`` (float32) and ``beta = sigmoid(b)``.
Two outputs because both are computed from one projection and consumed by the
same delta-rule kernel; a fused kernel would produce them together.
"""

import math

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import dtype_for

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_CORRECTNESS = tuple(
    Workload(dims=dict(rows=rows, heads=heads), dtype=dtype)
    for rows, heads, dtype in (
        (16, 4, "float32"),
        (33, 8, "float32"),
        (64, 32, "bfloat16"),
        (130, 8, "bfloat16"),
    )
)


def make_gdn_gating_inputs(torch, op, workload, device="cuda"):
    rows, heads = workload.dims["rows"], workload.dims["heads"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + heads)
    # A_log as the model initializes it, log U(0.01, 16); dt_bias near its
    # initial value of one, perturbed so its gradient is not a pure row sum
    # of a constant.
    A_log = torch.empty((heads,), device=device, dtype=torch.float32).uniform_(0.01, 16).log_()
    dt_bias = 1.0 + 0.1 * torch.randn((heads,), device=device, dtype=torch.float32)
    return {
        "a": torch.randn((rows, heads), device=device, dtype=dtype),
        "b": torch.randn((rows, heads), device=device, dtype=dtype),
        "A_log": A_log.to(dtype),
        "dt_bias": dt_bias.to(dtype),
        "dg": torch.randn((rows, heads), device=device, dtype=torch.float32),
        "dbeta": torch.randn((rows, heads), device=device, dtype=dtype),
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
#:   beta                    0        0         0     1     1     0      nan
#:   dA_log                  0        0         0     1     1     0      nan
#:   da                      0        0         0     1     1     0      nan
#:   db                      0        0         0     1     1     0      nan
#:   ddt_bias                0        0         0     1     1     0      nan
#:   g                       0        0         0     1     1     0      nan
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"heads": 32, "rows": 64},
    reduction_dims={"dA_log": ('rows',), "ddt_bias": ('rows',)},
    result_dims={"beta": ('rows', 'heads'), "dA_log": ('heads',), "da": ('rows', 'heads'), "db": ('rows', 'heads'), "ddt_bias": ('heads',), "g": ('rows', 'heads')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="gdn_gating",
    level=1,
    family="elementwise",
    forward="evograd.ops.level1.gdn_gating.forward_ref:gdn_gating_forward_ref",
    dims=("rows", "heads"),
    args=(
        Active("a", "[rows, heads]"),
        Active("b", "[rows, heads]"),
        Active("A_log", "[heads]"),
        Active("dt_bias", "[heads]"),
    ),
    output=(
        Active("g", "[rows, heads]", dtype="float32"),
        Active("beta", "[rows, heads]"),
    ),
    parameter_args=("A_log", "dt_bias"),
    forward_semantics=(
        "Return (g, beta) IN THAT ORDER. g = -exp(float32(A_log)) * "
        "softplus(float32(a) + dt_bias), computed and returned in float32 "
        "(softplus(u) = log(1 + exp(u)), linear above a threshold of 20). "
        "beta = sigmoid(b) in b's dtype. A_log and dt_bias are per-head and "
        "broadcast over rows. Do not call autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dg, dbeta) and returns da, db, dA_log, ddt_bias "
        "IN THIS ORDER. With u = float32(a) + dt_bias and A = exp(A_log): "
        "dsoft = dg * (-A) * sigmoid(u); da = dsoft in a's dtype; ddt_bias = sum "
        "over rows of dsoft; dA_log = sum over rows of dg * g (since "
        "dg/dA_log = g); db = dbeta * beta * (1 - beta). Accumulate the row "
        "sums in float32 and return each gradient in its input's dtype."
    ),
    correctness=_CORRECTNESS,
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    make_inputs=make_gdn_gating_inputs,
)
