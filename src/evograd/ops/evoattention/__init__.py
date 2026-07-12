"""Operator declaration: evoattention."""

from evograd.opdecl import declare_op, Duplicated, Const, Workload


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


op = declare_op(
    name="evoattention",
    forward="evograd.ops.evoattention.forward_ref:evoattention_forward_ref",
    dims=('B', 'S', 'H', 'N', 'D'),
    args=(
        Duplicated("q", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Duplicated("k", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Duplicated("v", "[B, S, N, H, D]", dtype="float16|bfloat16"),
        Const("res_mask", "[B, S, 1, 1, N]", dtype="float32",
          note="additive per-key mask: 0 = keep, large-negative = drop"),
        Duplicated("pair_bias", "[B, 1, H, N, N]", dtype="float32", grad="d_pair_bias",
               note="trainable; broadcast over the N_seq (S) axis"),
    ),
    output=Duplicated("o", "[B, S, N, H, D]"),
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
    benchmark=tuple(
        Workload(dims=dict(B=b, S=s, H=h, N=n, D=d), dtype=dtype)
        for (b, s, h, n, d) in (
            # Dim=64 attention-pair-bias
            (1, 1, 16, 128, 64),
            (1, 1, 16, 256, 64),
            (1, 1, 16, 384, 64),
            # Dim=32 triangle attention
            (1, 64, 4, 128, 32),
            (1, 64, 4, 256, 32),
            (1, 128, 4, 128, 32),
            # Dim=128 register-spill stress shapes
            (1, 1, 8, 256, 128),
            (1, 1, 8, 384, 128),
        )
        for dtype in ("float16", "bfloat16")
    ),
    tolerances={"float16": (2e-2, 2e-2), "bfloat16": (4e-2, 4e-2)},
    tolerance_multipliers={"d_pair_bias": (2.0, 1.0)},
    make_inputs=make_evoattention_inputs,
)
