"""Operator declaration: Qwen3.5's Gated DeltaNet input stage (site ``gdn_in_proj``).

From the normalized residual stream to everything the delta rule consumes: four
input projections (q|k|v, z, b, a), the causal depthwise convolution with SiLU
over q|k|v, the gates, and the repeat of q and k up to the value-head count.
``z`` -- the output gate -- leaves here too and is consumed by
``qwen3_5_gdn_norm_out``. Four separate projections, no per-key-head
deinterleave.
"""

from evograd.opdecl import Active, Workload, declare_op

from ..task_tools import OBSERVED_SUITE, PROVISIONAL, head_major, observed, seed_of, shrunk

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "H", "C", "VD", "HV", "K", "D")


def model_strides(dims):
    """The layouts the block hands this site's backward.

    The delta rule moves q, k, v, g and beta head-major with
    ``transpose(1, 2).contiguous()``, so the gradients flowing back into them are
    ``[B, T, HV, ...]`` views of ``[B, HV, T, ...]`` buffers -- fixed by those
    view operations, not by any backward kernel.
    """
    b, t, hv, d = dims["B"], dims["T"], dims["HV"], dims["D"]
    heads = head_major(b, t, hv, d)
    gates = (hv * t, 1, t)
    return (("dq", heads), ("dk", heads), ("dv", heads), ("dbeta", gates), ("dg", gates))


_BENCHMARK = observed("gdn_in_proj", strides=model_strides)


def _dims(batch, tokens, hidden, hk, hv, d, taps=4):
    return {"B": batch, "T": tokens, "H": hidden, "C": 2 * hk * d + hv * d, "VD": hv * d,
            "HV": hv, "K": taps, "D": d}


_CORRECTNESS = tuple(
    Workload(dims=_dims(*shape), dtype=dtype, provenance=shrunk("gdn_in_proj"))
    for shape, dtype in (
        ((1, 12, 32, 2, 4, 8), "float32"),
        ((2, 20, 64, 2, 4, 16), "float32"),
        # HV == HK: no repeat; a kernel that always repeats fails here.
        ((1, 16, 32, 2, 2, 8), "float32"),
        ((2, 24, 64, 2, 4, 16), "bfloat16"),
    )
) + (
    # The model's gradient layouts, so a backward that assumed contiguous
    # upstream gradients is caught on the grid.
    Workload(dims=_dims(2, 20, 64, 2, 4, 16), dtype="float32",
             provenance=shrunk("gdn_in_proj", model_strides(_dims(2, 20, 64, 2, 4, 16)))),
)


def make_gdn_in_proj_inputs(torch, op, workload, device="cuda"):
    """Unit-scale activations and gradients; projections at the initializer's
    ``fan_in^-0.5``; ``A_log`` and ``dt_bias`` as the model initializes them.
    Not yet rescaled to the model's real magnitudes (a calibration step)."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 11))
    batch, tokens, hidden, hv, d = dims["B"], dims["T"], dims["H"], dims["HV"], dims["D"]

    def randn(*shape, scale=1.0, out=dtype):
        return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(out)

    A_log = torch.empty((hv,), device=device, dtype=torch.float32).uniform_(0.01, 16).log_()
    return {
        "x": randn(batch, tokens, hidden),
        "w_qkv": randn(dims["C"], hidden, scale=hidden ** -0.5),
        "w_z": randn(dims["VD"], hidden, scale=hidden ** -0.5),
        "w_b": randn(hv, hidden, scale=hidden ** -0.5),
        "w_a": randn(hv, hidden, scale=hidden ** -0.5),
        "conv_w": randn(dims["C"], dims["K"], scale=dims["K"] ** -0.5),
        "A_log": A_log.to(dtype),
        "dt_bias": (1.0 + 0.1 * torch.randn((hv,), device=device)).to(dtype),
        "dq": randn(batch, tokens, hv, d),
        "dk": randn(batch, tokens, hv, d),
        "dv": randn(batch, tokens, hv, d),
        "dz": randn(batch, tokens, hv, d),
        "dbeta": randn(batch, tokens, hv),
        "dg": randn(batch, tokens, hv, out=torch.float32),
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
#:   beta                    0        0         0     1     1     0      nan
#:   dA_log                  0        0  5.28e-07     1     1     0 1.89e+06
#:   dconv_w           0.00776     1.86      19.4     1   2.8   1.5     1.76
#:   ddt_bias                0        0  2.52e-07     1     1     0 3.96e+06
#:   dw_a                    0      5.5       117     1   8.3     3     1.84
#:   dw_b                    0    0.482      6.67     1     1   1.5     2.02
#:   dw_qkv             0.0663     2.83      45.9     1   4.3     2     1.58
#:   dw_z                    0        0    0.0088     1     1     0      114
#:   dx                 0.0107    0.825      1.62     1   1.3     3     1.74
#:   g                       0        0         0     1     1     0      nan
#:   k                0.000623   0.0925     0.329     1     1     0     3.04
#:   q                0.000852    0.108     0.345     1     1     0     2.89
#:   v                4.29e-05    0.179     0.346     1     1     0     2.89
#:   z                       0        0  0.000259     1     1     0 3.86e+03
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "C": 128, "D": 16, "H": 64, "HV": 4, "K": 4, "T": 24, "VD": 64},
    reduction_dims={"dA_log": ('B', 'T'), "dconv_w": ('B', 'T'), "ddt_bias": ('B', 'T'), "dw_a": ('B', 'T'), "dw_b": ('B', 'T'), "dw_qkv": ('B', 'T'), "dw_z": ('B', 'T')},
    result_dims={"beta": ('B', 'T', 'HV'), "dA_log": ('HV',), "dconv_w": ('C', 'K'), "ddt_bias": ('HV',), "dw_a": ('HV', 'H'), "dw_b": ('HV', 'H'), "dw_qkv": ('C', 'H'), "dw_z": ('VD', 'H'), "dx": ('B', 'T', 'H'), "g": ('B', 'T', 'HV'), "k": ('B', 'T', 'HV', 'D'), "q": ('B', 'T', 'HV', 'D'), "v": ('B', 'T', 'HV', 'D'), "z": ('B', 'T', 'HV', 'D')},
    gain=0.0,
    result_gain={"dconv_w": 1.5, "dw_a": 3, "dw_b": 1.5, "dw_qkv": 2, "dx": 3},
    dtype_multipliers={"bfloat16": {"dconv_w": 2.8, "dw_a": 8.3, "dw_qkv": 4.3, "dx": 1.3}},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_gdn_in_proj",
    level=2,
    family="linear_attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gdn_in_proj.reference:"
            "gdn_in_proj_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gdn_in_proj.reference:"
                    "gdn_in_proj_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("w_qkv", "[C, H]"),
        Active("w_z", "[VD, H]"),
        Active("w_b", "[HV, H]"),
        Active("w_a", "[HV, H]"),
        Active("conv_w", "[C, K]"),
        Active("A_log", "[HV]"),
        Active("dt_bias", "[HV]"),
    ),
    output=(
        Active("q", "[B, T, HV, D]"),
        Active("k", "[B, T, HV, D]"),
        Active("v", "[B, T, HV, D]"),
        Active("z", "[B, T, HV, D]"),
        Active("beta", "[B, T, HV]"),
        Active("g", "[B, T, HV]", dtype="float32"),
    ),
    parameter_args=("w_qkv", "w_z", "w_b", "w_a", "conv_w", "A_log", "dt_bias"),
    forward_semantics=(
        "Let HK = (C - VD) / (2*D) key heads and R = HV / HK (VD = HV*D). "
        "qkv = x @ w_qkv^T, [B, T, C], channels laid out q (HK*D) | k (HK*D) | v (VD); "
        "transpose to [B, C, T] and apply the depthwise causal conv "
        "y[c, t] = sum_j conv_w[c, j] * qkv[c, t - (K-1) + j] (zeros before t=0) "
        "and SiLU, transpose back and split into q [B,T,HK,D], k [B,T,HK,D], "
        "v [B,T,HV,D]. z = (x @ w_z^T) viewed [B, T, HV, D]; b = x @ w_b^T and "
        "a = x @ w_a^T, both [B, T, HV]; beta = sigmoid(b); g = -exp(float32(A_log)) "
        "* softplus(float32(a) + dt_bias) in float32. If R > 1 repeat q and k R "
        "times along the head axis (repeat_interleave). Return (q, k, v, z, beta, g) "
        "IN THAT ORDER; g is float32, the rest x's dtype. Do not call F.linear, "
        "F.conv1d, @, torch.matmul or autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dq, dk, dv, dz, dbeta, dg) and returns dx, dw_qkv, "
        "dw_z, dw_b, dw_a, dconv_w, dA_log, ddt_bias IN THIS ORDER. Sum dq and dk "
        "over each group of R repeated heads; with u = float32(a) + dt_bias, "
        "da = dg * (-exp(A_log)) * sigmoid(u), ddt_bias = sum over B, T of that, "
        "dA_log = sum over B, T of dg * g, db = dbeta * beta * (1 - beta). Back "
        "through SiLU and the causal conv (dconv_w[c, j] = sum over B, t of "
        "ds[c, t] * qkv[c, t - (K-1) + j]; the input gradient is the transposed "
        "convolution) to dqkv [B, T, C]; then dx = dqkv @ w_qkv + dz @ w_z + "
        "db @ w_b + da @ w_a, and each weight gradient d^T @ x summed over B and T. "
        "Accumulate in float32; each gradient in its input's dtype."
    ),
    extra_constraints=(
        "No deinterleave: w_qkv's output channels are q | k | v in the convolution's "
        "own order. Outputs keep the model's layouts: q, k, z, beta, g contiguous; v "
        "is a view of the transposed [B, C, T] convolution output. Dk == Dv == D."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    memory_inputs=("x", "w_qkv", "w_z", "w_b", "w_a", "conv_w"),
    make_inputs=make_gdn_in_proj_inputs,
)
