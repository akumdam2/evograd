"""Operator declaration: layernorm."""

from evograd.opdecl import declare_op, Duplicated, Const, Workload

op = declare_op(
    name="layernorm",
    forward="evograd.ops.layernorm_forward_ref:layernorm_forward_ref",
    dims=('rows', 'hidden'),
    args=(
        Duplicated("x", "[rows, hidden]"),
        Duplicated("weight", "[hidden]"),
        Duplicated("bias", "[hidden]"),
        Const("eps", default=1e-5),
    ),
    output=Duplicated("y", "[rows, hidden]"),
    forward_semantics='Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. Forward must produce the same `y` as row-wise LayerNorm over the last dimension.',
    backward_semantics='Backward must consume only `dy`, `saved_tensors`, and `eps`. Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.',
    # Ported from benchmark/triton_layernorm_backward_bench/task_spec.py
    # (benchmark suite: "mixed" — the old repo's default; other suites were
    # env-selected and can come back as declaration variants if needed).
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="float16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="float16"),
        Workload(dims=dict(rows=32, hidden=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="bfloat16"),
    ),
    benchmark=tuple(
        Workload(dims=dict(rows=r, hidden=h), dtype=dtype)
        for (r, h) in (
            # dynamic
            (1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
            # non-tile-aligned
            (17, 127), (17, 513), (17, 1000),
        )
        for dtype in ("float32", "float16", "bfloat16")
    ),
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
)
