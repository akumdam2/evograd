"""Operator declaration: Qwen3.5's dense SwiGLU MLP (site ``swiglu_mlp``).

Every one of the 32 layers runs it, 2560 -> 9216 -> 2560 in Qwen3.5-4B, after
``post_attention_layernorm``. The contract is Qwen3-0.6B's ``qwen3_swiglu_mlp``
at Qwen3.5's widths, under this model's own key, cases and provenance.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import make_pair_baseline

from .. import library
from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "H", "I")

_BENCHMARK = observed("swiglu_mlp")

_CORRECTNESS = tuple(
    Workload(dims={"B": b, "T": t, "H": h, "I": i}, dtype=dtype, provenance=shrunk("swiglu_mlp"))
    for b, t, h, i, dtype in (
        (1, 8, 16, 48, "float32"),
        (2, 16, 32, 112, "float32"),
        (2, 16, 32, 112, "bfloat16"),
    )
)


def make_swiglu_mlp_inputs(torch, op, workload, device="cuda"):
    """Unit activations; projections at ``fan_in^-0.5`` so SiLU is not saturated."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 17))

    def weight(rows, cols):
        return (torch.randn((rows, cols), device=device, dtype=torch.float32)
                * cols ** -0.5).to(dtype)

    return {
        "x": torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype),
        "gate_weight": weight(dims["I"], dims["H"]),
        "up_weight": weight(dims["I"], dims["H"]),
        "down_weight": weight(dims["H"], dims["I"]),
        "dout": torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype),
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
#:   ddown_weight            0     1.64      20.6     1   2.5   1.5     2.04
#:   dgate_weight            0    0.921      10.7     1   1.4     1     1.51
#:   dup_weight              0        1        11     1   1.5     1     1.57
#:   dx                      0    0.535      1.05     1     1   1.5     1.66
#:   out                     0    0.252     0.861     1     1     1     1.74
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "H": 32, "I": 112, "T": 16},
    reduction_dims={"ddown_weight": ('B', 'T'), "dgate_weight": ('B', 'T'), "dup_weight": ('B', 'T')},
    result_dims={"ddown_weight": ('H', 'I'), "dgate_weight": ('I', 'H'), "dup_weight": ('I', 'H'), "dx": ('B', 'T', 'H'), "out": ('B', 'T', 'H')},
    gain=0.0,
    result_gain={"ddown_weight": 1.5, "dgate_weight": 1, "dup_weight": 1, "dx": 1.5, "out": 1},
    dtype_multipliers={"bfloat16": {"ddown_weight": 2.5, "dgate_weight": 1.4, "dup_weight": 1.5}},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_swiglu_mlp",
    level=2,
    family="mlp",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.swiglu_mlp.reference:"
            "swiglu_mlp_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.swiglu_mlp.reference:"
                    "swiglu_mlp_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("gate_weight", "[I, H]"),
        Active("up_weight", "[I, H]"),
        Active("down_weight", "[H, I]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("gate_weight", "up_weight", "down_weight"),
    forward_semantics=(
        "gate = x @ gate_weight.T; up = x @ up_weight.T; hidden = silu(gate) * up "
        "computed with float32 accumulation and cast back to x's dtype; out = "
        "hidden @ down_weight.T. x is [B, T, H], gate_weight and up_weight are "
        "[I, H], down_weight is [H, I], out is [B, T, H]. silu(v) = v * sigmoid(v). "
        "Accumulate every GEMM in float32. Do not call F.linear, torch.matmul, @, "
        "F.silu, or autograd in the generated math."
    ),
    backward_semantics=(
        "Return gradients for x, gate_weight, up_weight, and down_weight IN THIS "
        "ORDER. With hidden = silu(gate) * up and dh = dout @ down_weight: "
        "ddown_weight = dout^T @ hidden ([H, I]); dup = dh * silu(gate); dgate = dh * "
        "up * dsilu(gate) where dsilu(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v))); "
        "dgate_weight = dgate^T @ x and dup_weight = dup^T @ x (both [I, H], summed "
        "over B and T); dx = dgate @ gate_weight + dup @ up_weight. Accumulate every "
        "reduction and matmul in float32 before casting each gradient to its input's "
        "dtype."
    ),
    extra_constraints=(
        "gate_weight and up_weight are separate tensors, as Qwen3.5 stores them. No "
        "biases. B and T are separate axes because that is the shape the module "
        "receives; a kernel may flatten them internally."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={"liger": make_pair_baseline(
        library.liger_swiglu_mlp, ("x", "gate_weight", "up_weight", "down_weight"),
        ("gate_weight", "up_weight", "down_weight"))},
    make_inputs=make_swiglu_mlp_inputs,
)
