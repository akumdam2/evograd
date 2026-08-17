"""Operator declaration: Liger fused MoE grouped-GEMM + SwiGLU."""

from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op

# Liger's own fused-MoE benchmark grid. The expert count and top-k match
# Mixtral-style routing, but hidden/intermediate are scaled down from
# Mixtral-8x7B's 4096/14336 so the grid fits one GPU, so this is quoted from
# a published sweep rather than derived from a model config.
_PROVENANCE = Provenance(
    model="mixtral_style",
    component="moe_grouped_gemm_swiglu",
    source="paper",
    scaled=True,
    note=(
        "Liger fused_moe benchmark grid; routing matches Mixtral-8x7B (8-16 "
        "experts, top-2) with hidden/intermediate scaled down to fit one GPU"
    ),
)
from evograd.ops.level2.fused_moe_swiglu.liger import measure_liger_baseline


_DIMS = ("T", "H", "E", "G", "I", "K")
_BENCHMARK_CASES = (
    # T, H, E, 2I, I, top-k
    (128, 512, 8, 2816, 1408, 2),
    (512, 512, 8, 2816, 1408, 2),
    (512, 1024, 8, 5632, 2816, 2),
    (2048, 1024, 8, 5632, 2816, 2),
    (1024, 2048, 8, 11264, 5632, 2),
    (2048, 2048, 16, 11264, 5632, 2),
)


def _workload(values):
    return Workload(
        dims=dict(zip(_DIMS, values)), dtype="bfloat16", provenance=_PROVENANCE
    )


def _benchmark_workloads():
    return tuple(_workload(values) for values in _BENCHMARK_CASES)


def make_fused_moe_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    if dims["G"] != 2 * dims["I"]:
        raise ValueError("G must equal 2 * I for gate/up projection weights")
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["T"] * 100003 + dims["H"] * 1009 + dims["E"] * 101 + dims["I"]
    )
    x = torch.randn((dims["T"], dims["H"]), device=device, dtype=dtype)
    gate_up_proj = (
        torch.randn(
            (dims["E"], dims["G"], dims["H"]),
            device=device,
            dtype=dtype,
        )
        * (dims["H"] ** -0.5)
    ).to(dtype)
    down_proj = (
        torch.randn(
            (dims["E"], dims["H"], dims["I"]),
            device=device,
            dtype=dtype,
        )
        * (dims["I"] ** -0.5)
    ).to(dtype)
    token = torch.arange(dims["T"], device=device, dtype=torch.int32)[:, None]
    rank = torch.arange(dims["K"], device=device, dtype=torch.int32)[None, :]
    top_k_index = (token + rank).remainder(dims["E"]).contiguous()
    top_k_weights = torch.softmax(
        torch.randn(
            (dims["T"], dims["K"]),
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    ).to(dtype)
    dout = torch.randn((dims["T"], dims["H"]), device=device, dtype=dtype)
    return {
        "x": x,
        "gate_up_proj": gate_up_proj,
        "down_proj": down_proj,
        "top_k_index": top_k_index,
        "top_k_weights": top_k_weights,
        "dout": dout,
    }


op = declare_op(
    name="fused_moe_swiglu",
    level=2,
    family="moe",
    forward=(
        "evograd.ops.level2.fused_moe_swiglu.forward_ref:"
        "fused_moe_swiglu_forward_ref"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[T, H]"),
        Active("gate_up_proj", "[E, G, H]"),
        Active("down_proj", "[E, H, I]"),
        Inactive("top_k_index", "[T, K]", dtype="int32"),
        Active("top_k_weights", "[T, K]"),
    ),
    output=Active("out", "[T, H]"),
    forward_semantics=(
        "For each token and routed expert, compute a grouped gate/up GEMM, "
        "apply SwiGLU as silu(gate) * up, compute the expert down-projection, "
        "then sum the top-k expert outputs weighted by top_k_weights. "
        "top_k_index is fixed routing metadata and is non-differentiable. "
        "Generated math must use Triton grouped tl.dot kernels and may not "
        "call the Liger implementation or PyTorch autograd."
    ),
    backward_semantics=(
        "Return gradients for x, gate_up_proj, down_proj, and top_k_weights "
        "in that order. No gradient is returned for top_k_index. Accumulate "
        "GEMMs and expert reductions in float32 before casting each gradient "
        "to its corresponding active input dtype."
    ),
    extra_constraints=(
        "This is the Liger-specific GEMM+activation benchmark: it is routed "
        "MoE semantics, not a generic dense GEMM. G must equal 2*I, routing "
        "indices are int32 in [0,E), and all floating tensors are contiguous "
        "BF16 CUDA tensors."
    ),
    correctness=(
        _workload((16, 64, 4, 256, 128, 2)),
        _workload((32, 128, 8, 512, 256, 2)),
    ),
    coverage=_benchmark_workloads(),
    benchmark=_benchmark_workloads(),
    benchmark_suites={"liger_moe_bf16": _benchmark_workloads()},
    performance_baselines={"liger": measure_liger_baseline},
    memory_inputs=("x", "gate_up_proj", "down_proj", "top_k_weights"),
    tolerances={"bfloat16": (3e-1, 1e-1)},
    tolerance_multipliers={
        "dgate_up_proj": (2.0, 1.0),
        "ddown_proj": (2.0, 1.0),
    },
    make_inputs=make_fused_moe_inputs,
)
