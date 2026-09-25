"""Operator declaration: layernorm_linear."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import make_pair_baseline
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import fixed_shape_suites, model_workloads

# MegaFold's FusedLayernormLinear, normalizing the c_s single representation and
# projecting it to the pair-bias attention fan-out. The declaration is 2-D, so
# the batch / MSA / residue axes flatten into M; the sweep varies that flattened
# token count while K and N stay at the AlphaFold3 widths.
_FREE = (
    {"batch": 1, "n_seq": 1, "residues": 384},
    {"batch": 1, "n_seq": 64, "residues": 128},
    {"batch": 1, "n_seq": 64, "residues": 256},
    {"batch": 1, "n_seq": 128, "residues": 256},
)
_DERIVED = model_workloads(
    ALPHAFOLD3,
    "layernorm_linear",
    _FREE,
    # bfloat16 added in v1: MegaFold trains in bf16, and the pre-v1 grid
    # measured only fp32/fp16, i.e. never the dtype this kernel actually runs in.
    ("float32", "float16", "bfloat16"),
    scaled=True,
    note="MegaFold runs batch 1 with activation checkpointing; the MSA depth is swept instead",
)

# The pair track. ``_DERIVED`` above is the single-representation site
# (``c_s`` -> q/k/v fan-out); the level-4 MegaFold workload spends most of its
# LayernormLinear calls on the pair representation instead -- one row per
# residue pair, ``c_z = 128`` in -- feeding triangle multiplication (N = 5 c_z,
# twice per block) and the pair transition (N = 2 * 4 c_z, once per block).
# At the production crop of 384 residues that is M = 147456, K = 128.
_PAIR_DTYPES = ("float32", "bfloat16")
_PAIR = model_workloads(
    ALPHAFOLD3, "pair_transition_layernorm_linear",
    tuple({"residues": r} for r in AF3_RESIDUE_SWEEP), _PAIR_DTYPES,
    note="MegaFold runs this site in float32 outside its own Triton path",
) + model_workloads(
    ALPHAFOLD3, "triangle_mult_layernorm_linear",
    tuple({"residues": r} for r in AF3_RESIDUE_SWEEP), _PAIR_DTYPES,
    note="MegaFold runs this site in float32 outside its own Triton path",
)
#: Both pair-track shapes at the production crop, float32: one program has to
#: serve both, so this pair is the level-4 evolution objective.
_PAIR_384 = tuple(w for w in _PAIR if w.dims["M"] == 384 * 384 and w.dtype == "float32")


def make_layernorm_linear_inputs(torch, op, workload, device="cuda"):
    m, k, n = (workload.dims[name] for name in ("M", "K", "N"))
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed((m * 100003 + k) * 100003 + n)
    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = (1.0 + 0.5 * torch.randn((k,), device=device, dtype=dtype)).to(dtype)
    bias = (0.5 * torch.randn((k,), device=device, dtype=dtype)).to(dtype)
    linear_weight = (torch.randn((k, n), device=device, dtype=dtype) * (k ** -0.5)).to(dtype)
    dout = torch.randn((m, n), device=device, dtype=dtype)
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "linear_weight": linear_weight,
        "eps": 1e-5,
        "dout": dout,
    }

def _megafold_factory():
    from evograd.benchmark.operator_suite.tasks.level2.layernorm_linear.megafold_pair import (
        make_megafold_layernorm_linear_autograd_pair_fns,
    )

    return make_megafold_layernorm_linear_autograd_pair_fns()


op = declare_op(
    name="layernorm_linear",
    forward="evograd.benchmark.operator_suite.tasks.level2.layernorm_linear.forward_ref:layernorm_linear_forward_ref",
    level=2,
    family="gemm",
    dims=('M', 'K', 'N'),
    args=(
        Active("x", "[M, K]"),
        Active("weight", "[K]", note="LayerNorm gamma"),
        Active("bias", "[K]", note="LayerNorm beta; not needed by backward"),
        Active("linear_weight", "[K, N]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("out", "[M, N]"),
    parameter_args=("weight", "bias", "linear_weight"),
    forward_semantics='Forward computes a fused LayerNorm -> Linear: x_hat = (x - mean(x, axis=-1)) * rstd with rstd = rsqrt(var(x, axis=-1) + eps); y_hat = x_hat * weight + bias (weight=gamma, bias=beta, both length K); out = y_hat @ linear_weight. x is [M, K], linear_weight is [K, N], out is [M, N]. Compute the LayerNorm row statistics in float32 and accumulate the GEMM in float32. Do not call F.layer_norm, F.linear, torch.matmul, @, or autograd in the generated math.',
    backward_semantics="Backward must return (dx, dlinear_weight, dweight, dbias) IN THIS ORDER. With dy_hat = dout @ linear_weight.T: dlinear_weight = y_hat.T @ dout (shape [K, N], linear_weight's dtype); dweight = sum(dy_hat * x_hat, dim=0) (shape [K], weight/gamma's dtype); dbias = sum(dy_hat, dim=0) (shape [K], bias/beta's dtype); with wdy = dy_hat * weight, dx = (wdy - x_hat * mean(x_hat * wdy, axis=-1) - mean(wdy, axis=-1)) * rstd (shape [M, K], x's dtype). Use float32 accumulation for all reductions and matmuls.",
    extra_constraints='Tensor layout notes:\n- x: [M, K], weight (gamma): [K], bias (beta): [K], linear_weight (B): [K, N], dout: [M, N]; contiguous CUDA, float32 or float16\n- Output order is (dx, dlinear_weight, dweight, dbias) -- note dlinear_weight (the Linear matrix grad) comes before the LayerNorm gamma/beta grads.\n- bias (beta) is not needed to compute any gradient and need not be saved.\n- Prefer saving compact per-row statistics (mean, rstd) over the full [M, K] x_hat/y_hat activations when backward can recompute them cheaply.',
    grad_order=("dx", "dlinear_weight", "dweight", "dbias"),
    # Ported from benchmark/triton_layernorm_linear_backward_bench/task_spec.py
    # (task_spec cases are (M, K, N) — same order as the declaration dims).
    correctness=(
        Workload(dims=dict(M=64, K=64, N=128), dtype="float32"),
        Workload(dims=dict(M=129, K=127, N=257), dtype="float32"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="float16"),
        Workload(dims=dict(M=512, K=256, N=512), dtype="float16"),
    ),
    benchmark=_DERIVED,
    benchmark_suites={
        **fixed_shape_suites(_DERIVED),
        "megafold_pair": _PAIR,
        "megafold_pair384": _PAIR_384,
        **fixed_shape_suites(_PAIR),
        "legacy": tuple(
            Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
            for (m, k, n) in (
                (4096, 128, 512), (8192, 128, 1024), (16384, 128, 256),
                (4096, 256, 1024), (2048, 512, 2048), (384, 768, 1536),
            )
            for dtype in ("float32", "float16")
        ),
    },
    tolerances={
        "float32": (8e-2, 2e-2),
        "float16": (1e-1, 2e-2),
        "bfloat16": (1.5e-1, 3e-2),
    },
    tolerance_multipliers={"dweight": (2.0, 1.0), "dbias": (2.0, 1.0)},
    performance_baselines={
        "megafold": make_pair_baseline(
            _megafold_factory, ("x", "weight", "bias", "linear_weight", "eps")
        ),
    },
    make_inputs=make_layernorm_linear_inputs,
)
