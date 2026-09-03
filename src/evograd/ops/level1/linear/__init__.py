"""Operator declaration: bias-carrying Linear.

**This is not the projection a modern decoder runs.** Llama-3 and Qwen3 both set
``attention_bias=False`` and give their MLP projections and ``lm_head`` no bias;
those live in ``linear_no_bias``. What remains here is deliberately an
*ablation*: the same widths measured through a contract that adds a bias.

The grid is kept because ``dbias`` is a real row reduction worth exercising at
realistic widths, and because deleting a historical timed set silently is worse
than labelling it. Its provenance is marked ``scaled`` with a note saying what
deviates, so the deviation is machine-visible rather than a comment: a
zero-valued bias tensor does *not* model a biasless projection, and this
declaration no longer claims it does.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import fixed_shape_suites, model_workloads

_GEMM_COMPONENTS = ("attn_qkv", "mlp_up", "mlp_down", "lm_head")
_GEMM_TOKENS = (2048, 8192)
_BIAS_ABLATION_NOTE = (
    "Llama-3-8B's projection widths measured through a bias-carrying Linear "
    "contract. Llama-3 has no projection biases, so the bias, its broadcast add "
    "and the dbias reduction are an ablation on top of the real configuration; "
    "the faithful grid is in linear_no_bias."
)
_DERIVED = tuple(
    workload
    for component in _GEMM_COMPONENTS
    for workload in model_workloads(
        LLAMA_3_8B,
        component,
        tuple({"tokens": tokens} for tokens in _GEMM_TOKENS),
        ("bfloat16",),
        scaled=True,
        note=_BIAS_ABLATION_NOTE,
    )
)


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
    coverage=_DERIVED,
    benchmark=_DERIVED,
    benchmark_suites={
        **fixed_shape_suites(_DERIVED),
        # Pre-v1 square grid, retained as an ablation control.
        "legacy": tuple(
            Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
            for (m, n, k) in (
                (512, 512, 512), (1024, 1024, 1024), (2048, 1024, 1024),
                (1024, 2048, 1024), (2048, 2048, 1024), (4096, 1024, 1024),
            )
            for dtype in ("float32", "float16")
        ),
    },
    # bfloat16 added for the derived grid: Llama-3 trains in bf16, and a GEMM
    # benchmark that only measures fp32/fp16 is not measuring the training path.
    tolerances={
        "float32": (8e-2, 2e-2),
        "float16": (1e-1, 2e-2),
        "bfloat16": (1.5e-1, 3e-2),
    },
    make_inputs=make_linear_inputs,
)
