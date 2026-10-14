"""Operator declaration: partial rotary embedding (``R`` of ``D`` channels).

The generic ``rope`` primitive rotates the whole head; this one rotates only
the leading ``R`` channels and passes the rest through, which is what models
with ``partial_rotary_factor < 1`` run. A kernel that rotated all ``D``
channels would be wrong on three quarters of every head at a factor of 0.25.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import dtype_for, is_head_major_view

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "H", "T", "D", "R")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, H=h, T=t, D=d, R=r), dtype=dtype)
    for b, h, t, d, r, dtype in (
        (1, 2, 8, 16, 4, "float32"),
        (2, 4, 16, 32, 8, "float32"),
        # R == D: the partial operator must reduce to full RoPE.
        (1, 2, 12, 16, 16, "float32"),
        (2, 4, 32, 64, 16, "bfloat16"),
        (1, 3, 24, 32, 8, "bfloat16"),
    )
)

#: A rotation base for the generated tables of a case with no model behind it.
#: The operator does not depend on it -- cos/sin are inputs -- but a random pair
#: would not be a rotation.
_THETA = 10000.0


def rotation_base(workload) -> float:
    """The theta a case's tables are generated at: its model's ``rope_theta``
    when the provenance names a registered model (1e7 for
    Qwen3.5), else :data:`_THETA`. An observed case then carries the
    frequencies the model rotates with, not a generic base."""
    provenance = getattr(workload, "provenance", None)
    if provenance is None:
        return _THETA
    from evograd.opdecl.models import MODELS

    record = MODELS.get(getattr(provenance, "model", None))
    theta = getattr(record, "rope_theta", None)
    return float(theta) if theta else _THETA


def make_partial_rope_inputs(torch, op, workload, device="cuda"):
    """Head-major input, a non-contiguous view where the model's is one, and
    real rotary tables of width ``R`` at the model's own theta."""
    dims = workload.dims
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(dims["B"] * 100003 + dims["H"] * 1009 + dims["T"] * 31 + dims["R"])
    shape = (dims["B"], dims["H"], dims["T"], dims["D"])
    if is_head_major_view(workload):
        x = torch.randn((dims["B"], dims["T"], dims["H"], dims["D"]),
                        device=device, dtype=dtype).transpose(1, 2)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
    rotary = dims["R"]
    inv_freq = 1.0 / (rotation_base(workload) ** (torch.arange(0, rotary, 2, device=device,
                                                               dtype=torch.float32) / rotary))
    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None]
    return {
        "x": x,
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "dy": torch.randn(shape, device=device, dtype=dtype),
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
#:   dx                      0    0.308     0.358     1     1     0     2.79
#:   y                       0    0.263     0.355     1     1     0     2.82
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "D": 64, "H": 4, "R": 16, "T": 32},
    reduction_dims={},
    result_dims={"dx": ('B', 'H', 'T', 'D'), "y": ('B', 'H', 'T', 'D')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="partial_rope",
    level=1,
    family="positional",
    forward="evograd.ops.level1.partial_rope.forward_ref:partial_rope_forward_ref",
    runtime_forward=(
        "evograd.ops.level1.partial_rope.forward_ref:partial_rope_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, H, T, D]"),
        Inactive("cos", "[1, T, R]", note="rotary cosine table, one row of positions"),
        Inactive("sin", "[1, T, R]", note="rotary sine table, same layout as cos"),
    ),
    output=Active("y", "[B, H, T, D]"),
    parameter_args=(),
    forward_semantics=(
        "Rotate the leading R channels of every head and pass the rest through. "
        "With cos and sin unsqueezed at dim 1 (broadcast over heads), "
        "xr = x[..., :R], rotate_half(t) = cat((-t[..., R/2:], t[..., :R/2]), -1): "
        "y[..., :R] = xr * cos + rotate_half(xr) * sin and y[..., R:] = x[..., R:]. "
        "Return a contiguous [B, H, T, D] tensor in x's dtype. Do not call "
        "autograd in the generated math."
    ),
    backward_semantics=(
        "Return dx. On the rotary slice, dx[..., :R] = dy_r * cos - "
        "rotate_half(dy_r * sin) with dy_r = dy[..., :R] (rotate_half is its own "
        "negative inverse); dx[..., R:] = dy[..., R:]. cos and sin are inactive "
        "and receive no gradient."
    ),
    extra_constraints=(
        "x may be a non-contiguous head-major view (a decoder reaches it by "
        "transposing [B, T, H, D]); the output is contiguous either way. R is "
        "even and at most D."
    ),
    correctness=_CORRECTNESS,
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    memory_inputs=("x",),
    make_inputs=make_partial_rope_inputs,
)
