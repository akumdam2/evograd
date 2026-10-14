"""Operator declaration: residual add + zero-centred RMSNorm, two outputs
(site ``residual_rmsnorm``).

Between the token mixer and the MLP of both block kinds: the post-attention
``residual + branch`` and ``post_attention_layernorm``. Returns the normalized
value and the sum itself, so the backward combines two upstream gradients -- the
characteristic error is to ignore ``dsummed``. Block semantics: the layer's input
norm and its closing residual add are not part of this site (the cross-layer
fusion is a later step's).
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import make_pair_baseline

from .. import library
from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "H")

_BENCHMARK = observed("residual_rmsnorm")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, H=h), dtype=dtype, provenance=shrunk("residual_rmsnorm"))
    for b, t, h, dtype in (
        (1, 8, 64, "float32"),
        (2, 17, 128, "float32"),
        (2, 32, 256, "bfloat16"),
        (1, 64, 640, "bfloat16"),
    )
)


def make_residual_rmsnorm_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 16))
    shape = (dims["B"], dims["T"], dims["H"])
    return {
        "branch": torch.randn(shape, device=device, dtype=dtype),
        "residual": torch.randn(shape, device=device, dtype=dtype),
        # Zero-centred, as the model stores it.
        "weight": (0.1 * torch.randn((dims["H"],), device=device)).to(dtype),
        "eps": 1e-6,
        # Two independent upstream gradients: a backward that ignored dsummed,
        # or assumed the two equal, must not pass.
        "dnormed": torch.randn(shape, device=device, dtype=dtype),
        "dsummed": torch.randn(shape, device=device, dtype=dtype),
    }


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
#:   dbranch                 0    0.352     0.355     1     1     0     2.82
#:   dresidual               0    0.352     0.355     1     1     0     2.82
#:   dweight                 0     2.17       9.2     1   3.3     1     2.24
#:   normed                  0        0         0     1     1     0      nan
#:   summed                  0        0         0     1     1     0      nan
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 1, "H": 640, "T": 64},
    reduction_dims={"dweight": ('B', 'T')},
    result_dims={"dbranch": ('B', 'T', 'H'), "dresidual": ('B', 'T', 'H'), "dweight": ('H',), "normed": ('B', 'T', 'H'), "summed": ('B', 'T', 'H')},
    gain=0.0,
    result_gain={"dweight": 1},
    dtype_multipliers={"bfloat16": {"dweight": 3.3}},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_residual_rmsnorm",
    level=2,
    family="norm",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.residual_rmsnorm.reference:"
            "residual_rmsnorm_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.residual_rmsnorm."
                    "reference:residual_rmsnorm_forward_production",
    dims=_DIMS,
    args=(
        Active("branch", "[B, T, H]"),
        Active("residual", "[B, T, H]"),
        Active("weight", "[H]"),
        Inactive("eps", default=1e-6),
    ),
    output=(
        Active("normed", "[B, T, H]"),
        Active("summed", "[B, T, H]"),
    ),
    parameter_args=("weight",),
    forward_semantics=(
        "s = residual + branch; with the reduction in float32, "
        "normed = s * rsqrt(mean(s^2, -1) + eps) * (1 + weight), computed in "
        "float32 and cast to the input dtype once at the end (zero-centred "
        "weight: zero is the identity scale). Return (normed, summed) IN THAT "
        "ORDER, both [B, T, H] in the input dtype, summed = s itself. Do not "
        "call F.rms_norm or autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dnormed, dsummed) and returns dbranch, dresidual, "
        "dweight IN THIS ORDER. With r = rsqrt(mean(s^2) + eps), xhat = s * r and "
        "g = dnormed * (1 + weight): ds = dsummed + r * (g - xhat * mean(g * "
        "xhat, -1)); dbranch = dresidual = ds; dweight = sum over B and T of "
        "dnormed * xhat. Accumulate the reductions in float32."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={"liger": make_pair_baseline(
        library.liger_residual_rmsnorm, ("branch", "residual", "weight", "eps"))},
    make_inputs=make_residual_rmsnorm_inputs,
)
