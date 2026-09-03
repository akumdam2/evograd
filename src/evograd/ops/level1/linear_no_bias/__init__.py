"""Operator declaration: biasless Linear, the projection a decoder actually runs.

Llama-3 and Qwen3 both set ``attention_bias=False`` and give their MLP
projections and ``lm_head`` no bias. The bias-carrying ``linear`` task therefore
did not describe them: a zero-valued bias still costs a broadcast add, a
``dbias`` row reduction, and a third entry in the gradient contract. Those are
real work a candidate would be optimized against and a baseline would pay for,
so the model-derived grids live here and ``linear`` keeps only its explicitly
biased ablation.

``weight`` is ``[N, K]`` -- ``nn.Linear``'s stored layout, and the one the
harvest recorded. ``matmul``'s ``[K, N]`` is a different interface over the same
mathematics; mapping the observed projections onto it would benchmark a
transpose the model does not perform.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import (
    fixed_shape_suites,
    model_workloads,
    observed_workloads,
)

#: The four Llama-3-8B projections, all of which are biasless in the published
#: configuration. Derived from it, so the widths cannot quietly stop being
#: Llama's.
_GEMM_COMPONENTS = ("attn_qkv", "mlp_up", "mlp_down", "lm_head")
_GEMM_TOKENS = (2048, 8192)
_DERIVED = tuple(
    workload
    for component in _GEMM_COMPONENTS
    for workload in model_workloads(
        LLAMA_3_8B,
        component,
        tuple({"tokens": tokens} for tokens in _GEMM_TOKENS),
        ("bfloat16",),
    )
)

#: The six deduplicated GEMMs one Qwen3-0.6B step runs -- q_proj, k/v_proj,
#: o_proj, gate/up_proj, down_proj and lm_head -- at the shapes and dtype the
#: harvest observed. All six are biasless, which is why they are here.
_QWEN3_OBSERVED = observed_workloads("qwen3_0_6b", "linear_no_bias")


def make_linear_no_bias_inputs(torch, op, workload, device="cuda"):
    """``x`` and a ``[N, K]`` weight in the layout ``nn.Linear`` stores.

    Contiguity is asserted rather than assumed: the observed weights are
    contiguous ``[out_features, in_features]`` tensors, and a benchmark that
    silently fed a transposed view would be measuring a different access
    pattern.
    """
    m, k, n = (workload.dims[name] for name in ("M", "K", "N"))
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed((m * 100003 + n) * 100003 + k)
    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = (torch.randn((n, k), device=device, dtype=dtype) * (k**-0.5)).to(dtype)
    dy = torch.randn((m, n), device=device, dtype=dtype)
    assert weight.is_contiguous() and list(weight.shape) == [n, k]
    return {"x": x, "weight": weight, "dy": dy}


op = declare_op(
    name="linear_no_bias",
    level=1,
    family="gemm",
    forward=(
        "evograd.ops.level1.linear_no_bias.forward_ref:linear_no_bias_forward_ref"
    ),
    # The eager baseline is timed through the call a model makes. The oracle
    # widens the operands to float32, which is more accurate and correspondingly
    # slower; timing against it would inflate every speedup.
    runtime_forward=(
        "evograd.ops.level1.linear_no_bias.forward_ref:linear_no_bias_runtime_ref"
    ),
    dims=("M", "K", "N"),
    args=(
        Active("x", "[M, K]"),
        Active("weight", "[N, K]"),
    ),
    output=Active("y", "[M, N]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Forward computes a biasless Linear layer y = x @ weight.T, where x is "
        "[M, K], weight is [N, K] and y is [M, N], all contiguous CUDA tensors. "
        "There is no bias term and no eps. Accumulate the matmul in float32 and "
        "cast y back to the input dtype. Do not call F.linear, torch.matmul, "
        "the @ operator, or autograd in the generated math; use a Triton tiled "
        "matmul (tl.dot)."
    ),
    backward_semantics=(
        "Backward must return (dx, dweight) -- two gradients, not three. "
        "dx = dy @ weight, shape [M, K], x's dtype. dweight = dy.T @ x, shape "
        "[N, K], weight's dtype. There is no bias and therefore no dbias: a "
        "row reduction over dy is work this operator does not do. Accumulate "
        "every matmul in float32 before casting."
    ),
    extra_constraints=(
        "Tensor layout notes:\n"
        "- x: [M, K], weight: [N, K], dy: [M, N], contiguous CUDA\n"
        "- dx: [M, K] (x.dtype), dweight: [N, K] (weight.dtype)\n"
        "- weight is [out_features, in_features], the layout nn.Linear stores "
        "and the layout the observed models use. This is not matmul's [K, N] "
        "interface.\n"
        "- These shapes are compute-bound; prefer tensor-core tiled matmul with "
        "fp32 accumulation and boundary masking for non-tile-aligned M/N/K."
    ),
    correctness=(
        Workload(dims=dict(M=64, K=64, N=64), dtype="float32"),
        Workload(dims=dict(M=129, K=127, N=257), dtype="float32"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="float16"),
        Workload(dims=dict(M=512, K=256, N=512), dtype="float16"),
        # bfloat16 is the training dtype of both models this task is derived
        # from, and the only one the observed grid uses.
        Workload(dims=dict(M=128, K=128, N=256), dtype="bfloat16"),
        Workload(dims=dict(M=257, K=129, N=127), dtype="bfloat16"),
    ),
    coverage=_DERIVED + _QWEN3_OBSERVED,
    benchmark=_DERIVED,
    benchmark_suites={
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
        **fixed_shape_suites(_DERIVED),
    },
    # Measured, not inherited. `bench.workloads.qwen3.levels.level1.mapping calibrate --op
    # linear_no_bias` compares the float32-accumulated oracle against
    # `runtime_forward` -- the call a model makes, and therefore the smallest
    # disagreement any correct implementation can have with the oracle -- on
    # every correctness workload and on all six observed Qwen configurations.
    #
    #   float32   0.0 exactly: at float32 the oracle's upcast is a no-op and the
    #             two spellings are the same computation. Kept at the
    #             repository's ordinary float32 pair rather than at zero.
    #   float16   worst 9.2e-04  -> 1.4e-03 (1.5x margin)
    #   bfloat16  worst 7.7e-03  -> 1.2e-02 (1.5x margin); the observed
    #             configurations are the binding ones, all within 7.6-7.7e-03,
    #             including the 151936-wide lm_head.
    #
    # These are far tighter than the (1.5e-1, 3e-2) the bias-carrying `linear`
    # task declares. That pair was never calibrated; this one is, and nothing
    # else consumes it.
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (1.4e-3, 1.4e-3),
        "bfloat16": (1.2e-2, 1.2e-2),
    },
    make_inputs=make_linear_no_bias_inputs,
)
