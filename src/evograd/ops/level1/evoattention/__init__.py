"""Operator declaration: evoattention."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import fixed_shape_suites, model_workloads


def make_evoattention_inputs(torch, op, workload, device="cuda"):
    """Semantic input construction (ported from the bench task_spec):
    res_mask is a real keep/drop mask with key 0 always kept (an all-dropped
    query row would NaN the softmax), and pair_bias stays float32 regardless
    of the case dtype, as in MegaFold."""
    d = workload.dims
    b, s, h, n, dim = d["B"], d["S"], d["H"], d["N"], d["D"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[workload.dtype]

    torch.manual_seed((((b * 131 + s) * 131 + h) * 131 + n) * 131 + dim)

    # std=0.5 keeps fp16/bf16 magnitudes in a sane range (matches MegaFold tests).
    q = torch.randn((b, s, n, h, dim), device=device, dtype=dtype) * 0.5
    k = torch.randn((b, s, n, h, dim), device=device, dtype=dtype) * 0.5
    v = torch.randn((b, s, n, h, dim), device=device, dtype=dtype) * 0.5
    do = torch.randn((b, s, n, h, dim), device=device, dtype=dtype) * 0.5

    keep = torch.rand((b, s, 1, 1, n), device=device) > 0.5
    keep[..., 0] = True
    res_mask = torch.where(
        keep,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), -1e9, device=device, dtype=torch.float32),
    )

    pair_bias = torch.randn((b, 1, h, n, n), device=device, dtype=torch.float32) * 0.5
    return {"q": q, "k": k, "v": v, "res_mask": res_mask, "pair_bias": pair_bias, "do": do}


_DTYPES = ("float16", "bfloat16")

# Each AlphaFold3 attention flavour carries its own head count and head dim, so
# the grid is three derived families rather than one sweep. The pre-v1 grid had
# the right (H, D) pairs but set S=64 for triangle attention at N in {128, 256};
# in real triangle attention the MSA axis *is* the third residue index, so those
# cases understated the work by 2x and 4x. Deriving S from the residue count
# makes that mistake unrepresentable.
_PAIR_BIAS = model_workloads(
    ALPHAFOLD3,
    "pair_bias_attention",
    tuple({"batch": 1, "n_seq": 1, "residues": n} for n in AF3_RESIDUE_SWEEP),
    _DTYPES,
    scaled=True,
    note="MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU",
)
_TRIANGLE = model_workloads(
    ALPHAFOLD3,
    "triangle_attention",
    tuple({"batch": 1, "residues": n} for n in AF3_RESIDUE_SWEEP[:2]),
    _DTYPES,
    scaled=True,
    note=(
        "MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU. "
        "Residues stop at 256 because S == N makes this the most expensive family"
    ),
)
_MSA = model_workloads(
    ALPHAFOLD3,
    "msa_attention",
    tuple({"batch": 1, "n_seq": 64, "residues": n} for n in AF3_RESIDUE_SWEEP[:2]),
    _DTYPES,
    scaled=True,
    note="MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU",
)
_BENCHMARK = _PAIR_BIAS + _TRIANGLE + _MSA

_STRESS = tuple(
    Workload(dims=dict(B=1, S=1, H=8, N=n, D=128), dtype=dtype)
    for n in (256, 384)
    for dtype in _DTYPES
)

op = declare_op(
    name="evoattention",
    forward="evograd.ops.level1.evoattention.forward_ref:evoattention_forward_ref",
    level=1,
    family="attention",
    dims=('B', 'S', 'H', 'N', 'D'),
    args=(
        Active("q", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Active("k", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Active("v", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Inactive("res_mask", "[B, S, 1, 1, N]", dtype="float32",
          note="additive per-key mask: 0 = keep, large-negative = drop"),
        Active("pair_bias", "[B, 1, H, N, N]", dtype="float32", grad="d_pair_bias",
               note="trainable; broadcast over the N_seq (S) axis"),
    ),
    output=Active("o", "[B, S, N, H, D]"),
    forward_semantics='Forward must produce the same output as: softmax(scale * Q @ K^T + pair_bias + res_mask) @ V, where scale = dim**-0.5 and Q/K/V are first transposed from [B, N_seq, N_res, Head, Dim] to [B, N_seq, Head, N_res, Dim]. The softmax is taken over the key residue axis in float32. Do not call PyTorch sdpa, flash_attn, or high-level attention APIs in the generated math.',
    backward_semantics='Backward must return (dq, dk, dv, d_pair_bias). dq/dk/dv have the same shape [B, N_seq, N_res, Head, Dim] and dtype as q/k/v. d_pair_bias has shape [B, 1, Head, N_res, N_res] and the same dtype as pair_bias. The AtenIR graph may produce a d_res_mask output — discard it, do not include it in the return tuple.',
    extra_constraints='Tensor layout notes:\n- q, k, v, do: [B, N_seq, N_res, Head, Dim], float16 or bfloat16\n- res_mask: [B, N_seq, 1, 1, N_res], float32, additive (0 = keep, large-negative = drop key)\n- pair_bias: [B, 1, Head, N_res, N_res], float32, broadcast over N_seq\n- output / do: [B, N_seq, N_res, Head, Dim]\n- Attention contracts over the N_res (key) axis, not Head or Dim.\n- pair_bias is shared across N_seq; d_pair_bias must sum/reduce over the N_seq axis.\n- Use float32 accumulation for all reductions (softmax, dV, dK, dQ).',

    # Ported from benchmark/triton_evoattention_backward_bench/task_spec.py.
    # Dims: B=batch, S=N_seq (MSA axis), H=heads, N=N_res, D=head dim.
    correctness=(
        Workload(dims=dict(B=1, S=1, H=4, N=23, D=8), dtype="float16", atol=2e-2, rtol=2e-2),
        Workload(dims=dict(B=1, S=2, H=4, N=64, D=16), dtype="float16", atol=2e-2, rtol=2e-2),
        Workload(dims=dict(B=1, S=1, H=8, N=80, D=32), dtype="bfloat16", atol=4e-2, rtol=4e-2),
        Workload(dims=dict(B=2, S=2, H=4, N=128, D=64), dtype="bfloat16", atol=4e-2, rtol=4e-2),
        # Dim=128: large accumulators / register-pressure path.
        Workload(dims=dict(B=1, S=1, H=4, N=48, D=128), dtype="float16", atol=3e-2, rtol=3e-2),
    ),
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK + _STRESS,
    benchmark_suites={
        "pair_bias": _PAIR_BIAS,
        "triangle": _TRIANGLE,
        "msa": _MSA,
        **fixed_shape_suites(_BENCHMARK),
        # Not an AlphaFold3 configuration: no AF3 module uses head dim 128 (the
        # largest is 64). These probe register pressure, which is worth running
        # but is not evidence about protein-model performance, so they are an
        # ablation suite and untimed coverage rather than part of the objective.
        "register_pressure": _STRESS,
        # Pre-v1 grid. Retained for comparison with earlier runs; note its
        # triangle-attention cases set S != N, which understates the real work.
        "legacy": tuple(
            Workload(dims=dict(B=b, S=s, H=h, N=n, D=d), dtype=dtype)
            for (b, s, h, n, d) in (
                (1, 1, 16, 128, 64), (1, 1, 16, 256, 64), (1, 1, 16, 384, 64),
                (1, 64, 4, 128, 32), (1, 64, 4, 256, 32), (1, 128, 4, 128, 32),
                (1, 1, 8, 256, 128), (1, 1, 8, 384, 128),
            )
            for dtype in ("float16", "bfloat16")
        ),
    },
    tolerances={"float16": (2e-2, 2e-2), "bfloat16": (4e-2, 4e-2)},
    tolerance_multipliers={"d_pair_bias": (2.0, 1.0)},
    make_inputs=make_evoattention_inputs,
)
