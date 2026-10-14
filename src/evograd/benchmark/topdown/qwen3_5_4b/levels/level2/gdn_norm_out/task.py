"""Operator declaration: gated RMSNorm plus the output projection (site ``gdn_norm_out``).

The Gated DeltaNet's tail: the delta rule's output ``o`` is normalized per value
head, scaled, gated by ``silu(z)`` and projected back to the residual width
(4096 -> 2560 in Qwen3.5-4B).
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import make_pair_baseline

from .. import library
from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "HV", "D", "VD", "H")

_BENCHMARK = observed("gdn_norm_out")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, HV=hv, D=d, VD=hv * d, H=h), dtype=dtype,
             provenance=shrunk("gdn_norm_out"))
    for b, t, hv, d, h, dtype in (
        (1, 12, 2, 8, 32, "float32"),
        (2, 20, 4, 16, 64, "float32"),
        (2, 24, 4, 16, 64, "bfloat16"),
    )
)


#: ``o`` is drawn at a per-token scale, log-uniform over 10**[-5, 0]. The captured
#: layer-14 call hands this site ``o`` at rms 8.5e-5 (max 2.5e-3;
#: ``results/experiments/qwen3_5_4b_cases/20261011_calib/derived_stats.json``):
#: mean(o^2) ~7e-9 against eps 1e-6, so the model's rows sit deep in the
#: eps-dominated regime, where rsqrt(mean + eps) is ~1/sqrt(eps) and an
#: implementation that mishandles eps is wrong. Unit-scale rows -- what a plain randn
#: draws -- never reach it; the spread covers that regime, the transition (rms
#: ~1e-3) and the normalized one in every case.
O_LOG10_SCALE = (-5.0, 0.0)
EPS = 1e-6


def make_gdn_norm_out_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 13))
    shape = (dims["B"], dims["T"], dims["HV"], dims["D"])
    scale = 10.0 ** torch.empty((dims["B"], dims["T"], 1, 1), device=device).uniform_(
        *O_LOG10_SCALE)
    o = (torch.randn(shape, device=device) * scale).to(dtype)
    z = torch.randn(shape, device=device, dtype=dtype)
    w_norm = (1.0 + 0.1 * torch.randn((dims["D"],), device=device)).to(dtype)
    w_out = (torch.randn((dims["H"], dims["VD"]), device=device) * dims["VD"] ** -0.5).to(dtype)
    # The norm's backward multiplies by r = rsqrt(mean(o^2) + eps): up to 1/sqrt(eps)
    # = 1e3 on an eps-dominated token. Scaling that token's dy by 1/r keeps ``do``
    # O(1) on every token, so one absolute atol is tight in every regime instead of
    # being set by the largest rows (a uniform dy put do's atol at 52x the base,
    # ~100% of the unit-regime rows' signal). Gradients are linear in dy: this
    # changes magnitudes only.
    dy_scale = (scale.reshape(dims["B"], dims["T"], 1).pow(2) + EPS).sqrt()
    dy = (torch.randn((dims["B"], dims["T"], dims["H"]), device=device) * dy_scale).to(dtype)
    return {"o": o, "z": z, "w_norm": w_norm, "w_out": w_out, "eps": EPS, "dy": dy}


# --- calibrated tolerances (scripts/qwen3_5_4b/calibration.py apply) ---
from evograd.opdecl.tolerance import ReductionScaledAtol  # noqa: E402

#: Calibrated for Qwen3.5-4B (run ``20261011_calib``; GH200, grid also on CPU):
#: ``op.forward`` against ``op.runtime_forward`` only -- never a library or
#: a candidate -- on captured, correctness, correctness[cpu], observed;
#: four repeats each (production was deterministic). The bfloat16 base
#: atol is the smallest 1-2-5 step at or above both 1.5x the forward
#: outputs' grid need and one bfloat16 ulp of their magnitude; rtol 2e-2.
#: Multipliers are per dtype (in the hook): 1.5x that dtype's grid need,
#: rounded up (none below 1). Each gain is the smallest that carries the
#: bfloat16 multiplier over the real-size cases with the same 1.5x.
#: ``need`` is the atol required at the base rtol, in units of the base atol.
#:
#:   result           fp32 grid bf16 grid real need fp32x bf16x  gain margin@real
#:   do                      0    0.103     0.108     1     1     0     9.27
#:   dw_norm                 0    0.192      8.83     1     1     1     2.77
#:   dw_out                  0     0.49      5.24     1     1     1     1.74
#:   dz                      0        0  6.79e-05     1     1     0 1.47e+04
#:   y                       0     0.25      0.41     1     1     0     2.44
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "D": 16, "H": 64, "HV": 4, "T": 24, "VD": 64},
    reduction_dims={"dw_norm": ('B', 'HV', 'T'), "dw_out": ('B', 'T')},
    result_dims={"do": ('B', 'T', 'HV', 'D'), "dw_norm": ('D',), "dw_out": ('H', 'VD'), "dz": ('B', 'T', 'HV', 'D'), "y": ('B', 'T', 'H')},
    gain=0.0,
    result_gain={"dw_norm": 1, "dw_out": 1},
    dtype_multipliers={},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_gdn_norm_out",
    level=2,
    family="linear_attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gdn_norm_out.reference:"
            "gdn_norm_out_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gdn_norm_out.reference:"
                    "gdn_norm_out_forward_production",
    dims=_DIMS,
    args=(
        Active("o", "[B, T, HV, D]"),
        Active("z", "[B, T, HV, D]"),
        Active("w_norm", "[D]"),
        Active("w_out", "[H, VD]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[B, T, H]"),
    parameter_args=("w_norm", "w_out"),
    forward_semantics=(
        "Per value head (rows of D): h = float32(o); h = h * rsqrt(mean(h^2, -1) "
        "+ eps); h = w_norm * h.to(o.dtype); h = (h * silu(float32(z))).to(o.dtype). "
        "Flatten the heads to [B, T, VD] (VD = HV * D) and project: "
        "y = h @ w_out^T, [B, T, H] in o's dtype. Do not call F.linear, @, "
        "torch.matmul, F.rms_norm or autograd in the generated math."
    ),
    backward_semantics=(
        "Return do, dz, dw_norm, dw_out IN THIS ORDER. dh = dy @ w_out, "
        "dw_out = dy^T @ h summed over B and T; then the gated-norm backward per "
        "head: with r = rsqrt(mean(o^2) + eps), xhat = o * r, s = silu(z): "
        "dz = dh * w_norm * xhat * silu'(z), dw_norm = sum over rows of "
        "dh * s * xhat, and with gn = dh * s * w_norm, do = r * (gn - xhat * "
        "mean(gn * xhat, -1)). Accumulate in float32; each gradient in its input's dtype."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={"fla": make_pair_baseline(
        library.fla_gdn_norm_out, ("o", "z", "w_norm", "w_out", "eps"), ("w_out",), sidecar="fla")},
    make_inputs=make_gdn_norm_out_inputs,
)
