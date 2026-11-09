"""Operator declaration: Qwen3.5's Gated DeltaNet input stage (site ``gdn_in_proj``).

From the normalized residual stream to everything the delta rule consumes: four
input projections (q|k|v, z, b, a), the causal depthwise convolution with SiLU
over q|k|v, the gates, and the repeat of q and k up to the value-head count.
``z`` -- the output gate -- leaves here too and is consumed by
``qwen3_5_gdn_norm_out``. Four separate projections, no per-key-head
deinterleave.
"""

from evograd.opdecl import Active, declare_op

from ..task_tools import OBSERVED_SUITE, head_major, observed, seed_of


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
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    memory_inputs=("x", "w_qkv", "w_z", "w_b", "w_a", "conv_w"),
    make_inputs=make_gdn_in_proj_inputs,
)
