"""Operator declaration: bias-carrying Linear.

The operator-suite grid remains a biased ablation of decoder projection widths.
The vision workloads additionally bind genuine biased projections observed in
ConvNeXt V2 and Qwen2-VL. Biasless decoder projections use ``linear_no_bias``;
observed model suites keep their provenance separate from the historical grid.

"""

from evograd.opdecl import Active, Workload, declare_op


def make_linear_inputs(torch, op, workload, device="cuda"):
    m, k, n = (workload.dims[name] for name in ("M", "K", "N"))
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed((m * 100003 + n) * 100003 + k)
    # Preserve the legacy task_spec draw order and initialization scale.
    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = (torch.randn((n, k), device=device, dtype=dtype) * (k ** -0.5)).to(dtype)
    dy = torch.randn((m, n), device=device, dtype=dtype)
    bias = torch.zeros((n,), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "dy": dy}

op = declare_op(
    name="linear",
    forward="evograd.ops.level1.linear.forward_ref:linear_forward_ref",
    level=1,
    family="gemm",
    dims=('M', 'K', 'N'),
    args=(
        Active("x", "[M, K]"),
        Active("weight", "[N, K]"),
        Active("bias", "[N]"),
    ),
    output=Active("y", "[M, N]"),
    parameter_args=("weight", "bias"),
    forward_semantics='Forward computes a Linear layer y = x @ weight.T + bias, where x is [M, K], weight is [N, K], bias is [N], and y is [M, N], all contiguous CUDA tensors. There is no eps. Accumulate the matmul in float32 and cast y back to the input dtype. Do not call F.linear, torch.matmul, the @ operator, or autograd in the generated math; use a Triton tiled matmul (tl.dot).',
    backward_semantics="Backward must return (dx, dweight, dbias). dx = dy @ weight, shape [M, K], x's dtype. dweight = dy.T @ x, shape [N, K], weight's dtype. dbias = sum(dy, dim=0), shape [N], dy's dtype. Accumulate all reductions/matmuls in float32. dbias does not depend on the bias value, so the bias tensor need not be saved.",
    extra_constraints='Tensor layout notes:\n- x: [M, K], weight: [N, K], bias: [N], dy: [M, N], contiguous CUDA, float32 or float16\n- dx: [M, K] (x.dtype), dweight: [N, K] (weight.dtype), dbias: [N] (dy.dtype)\n- These shapes are compute-bound; prefer tensor-core tiled matmul with fp32 accumulation and boundary masking for non-tile-aligned M/N/K.',
    # Ported from benchmark/triton_linear_backward_bench/task_spec.py
    # (task_spec cases are (M, N, K); declaration dims are M/K/N).
    correctness=(
        Workload(dims=dict(M=64, K=64, N=64), dtype="float32"),
        Workload(dims=dict(M=129, K=127, N=257), dtype="float32"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="float16"),
        Workload(dims=dict(M=512, K=256, N=512), dtype="float16"),
    ),
    # bfloat16 added for the derived grid: Llama-3 trains in bf16, and a GEMM
    # benchmark that only measures fp32/fp16 is not measuring the training path.
    tolerances={
        "float32": (8e-2, 2e-2),
        "float16": (1e-1, 2e-2),
        "bfloat16": (1.5e-1, 3e-2),
    },
    make_inputs=make_linear_inputs,
)
