"""Operator declaration: Qwen3.5's dense SwiGLU MLP (site ``swiglu_mlp``).

Every one of the 32 layers runs it, 2560 -> 9216 -> 2560 in Qwen3.5-4B, after
``post_attention_layernorm``. The contract is Qwen3-0.6B's ``qwen3_swiglu_mlp``
at Qwen3.5's widths, under this model's own key, cases and provenance.
"""

from evograd.opdecl import Active, declare_op

from ..task_tools import OBSERVED_SUITE, observed, seed_of


_DIMS = ("B", "T", "H", "I")

_BENCHMARK = observed("swiglu_mlp")


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
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    make_inputs=make_swiglu_mlp_inputs,
)
