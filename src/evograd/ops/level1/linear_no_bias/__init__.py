"""Operator declaration: biasless Linear, the projection a decoder actually runs.

Llama-3 and Qwen3 both set ``attention_bias=False`` and give their MLP
projections and ``lm_head`` no bias. The bias-carrying ``linear`` task therefore
did not describe them: a zero-valued bias still costs a broadcast add, a
``dbias`` row reduction, and a third entry in the gradient contract. Those are
real work a candidate would be optimized against and a baseline would pay for,
so the model-derived grids live here and ``linear`` keeps only its explicitly
biased ablation.

``weight`` is ``[N, K]`` -- ``nn.Linear``'s stored layout, and the one the
harvest recorded. A ``[K, N]`` weight is a different interface over the same
mathematics; mapping the observed projections onto it would benchmark a
transpose the model does not perform.
"""

from evograd.opdecl import Active, declare_op


#: The six deduplicated GEMMs one Qwen3-0.6B step runs -- q_proj, k/v_proj,
#: o_proj, gate/up_proj, down_proj and lm_head -- at the shapes and dtype the
#: harvest observed. All six are biasless, which is why they are here.


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
    make_inputs=make_linear_no_bias_inputs,
)
