"""Operator declaration: causal GQA attention, sigmoid output gate, o_proj
(site ``gated_attention``).

The second half of ``Qwen3_5Attention``: what remains after
``qwen3_5_qkv_norm_partial_rope`` has produced q, k, v and the gate. 16 query
heads over 4 KV heads at D = 256 in Qwen3.5-4B.
"""

from evograd.opdecl import Active, Workload, declare_op

from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk, token_major

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "HQ", "HK", "D", "QO", "H")


def model_strides(dims):
    """``v`` is ``v_proj(x).view(B, T, HK, D).transpose(1, 2)``: a head-major view."""
    return (("v", token_major(dims["B"], dims["T"], dims["HK"], dims["D"])),)


_BENCHMARK = observed("gated_attention", strides=model_strides)

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, HQ=hq, HK=hk, D=d, QO=hq * d, H=h), dtype=dtype,
             provenance=shrunk("gated_attention", model_strides(dict(B=b, T=t, HK=hk, D=d))))
    for b, t, hq, hk, d, h, dtype in (
        (1, 12, 4, 2, 16, 32, "float32"),
        (2, 20, 8, 2, 16, 64, "float32"),
        (2, 24, 4, 1, 32, 64, "bfloat16"),
    )
)


def make_gated_attention_inputs(torch, op, workload, device="cuda"):
    """q and k contiguous (RoPE's concatenation returns them so); v drawn
    contiguous and re-laid out by the cases that state the model's layout."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 15))
    b, t, hq, hk, d = dims["B"], dims["T"], dims["HQ"], dims["HK"], dims["D"]
    return {
        "q": torch.randn((b, hq, t, d), device=device, dtype=dtype),
        "k": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "v": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "gate": torch.randn((b, t, dims["QO"]), device=device, dtype=dtype),
        "o_weight": (torch.randn((dims["H"], dims["QO"]), device=device)
                     * dims["QO"] ** -0.5).to(dtype),
        "dy": torch.randn((b, t, dims["H"]), device=device, dtype=dtype),
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
#:   dgate            7.74e-05    0.084     0.137     1     1     0     7.29
#:   dk                0.00234    0.461     0.604     1     1     0     1.66
#:   do_weight          0.0161     3.46       5.4     1   5.2  0.25     2.83
#:   dq                 0.0133    0.145     0.392     1     1     0     2.55
#:   dv                0.00165    0.762     0.913     1   1.2   0.5     1.58
#:   y                 0.00271    0.294     0.446     1     1     0     2.24
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.01, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "D": 32, "H": 64, "HK": 1, "HQ": 4, "QO": 128, "T": 24},
    reduction_dims={"do_weight": ('B', 'T')},
    result_dims={"dgate": ('B', 'T', 'QO'), "dk": ('B', 'HK', 'T', 'D'), "do_weight": ('H', 'QO'), "dq": ('B', 'HQ', 'T', 'D'), "dv": ('B', 'HK', 'T', 'D'), "y": ('B', 'T', 'H')},
    gain=0.0,
    result_gain={"do_weight": 0.25, "dv": 0.5},
    dtype_multipliers={"bfloat16": {"do_weight": 5.2, "dv": 1.2}},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_gated_attention",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_attention.reference:"
            "gated_attention_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_attention."
                    "reference:gated_attention_forward_production",
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("gate", "[B, T, QO]"),
        Active("o_weight", "[H, QO]"),
    ),
    output=Active("y", "[B, T, H]"),
    parameter_args=("o_weight",),
    forward_semantics=(
        "Causal grouped-query attention with scale D^-0.5: each KV head serves "
        "HQ/HK consecutive query heads; scores = q @ k^T * D^-0.5 with strictly "
        "future positions masked, softmax in float32, a = weights @ v, "
        "[B, HQ, T, D]. Transpose to [B, T, HQ, D], flatten to [B, T, QO], "
        "multiply by sigmoid(gate) elementwise and project: y = (a * "
        "sigmoid(gate)) @ o_weight^T, [B, T, H] in q's dtype. No mask tensor, no "
        "dropout. Do not call F.scaled_dot_product_attention, F.linear, @, "
        "torch.matmul or autograd in the generated math."
    ),
    backward_semantics=(
        "Return dq, dk, dv, dgate, do_weight IN THIS ORDER. dag = dy @ o_weight, "
        "do_weight = dy^T @ (a * sigmoid(gate)) summed over B and T; "
        "dgate = dag * a * s * (1 - s) with s = sigmoid(gate); da = dag * s, "
        "reshaped back to [B, HQ, T, D]; then the causal attention backward, "
        "summing dk and dv over each group of query heads. Accumulate in "
        "float32; each gradient in its input's dtype."
    ),
    extra_constraints=(
        "HQ divisible by HK; head dim D may be 256. v may be a non-contiguous "
        "transposed view."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    make_inputs=make_gated_attention_inputs,
)
