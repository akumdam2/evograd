"""Operator declaration: layernorm."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import make_pair_baseline


def make_layernorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    bias = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "eps": 1e-5, "dy": dy}


_MIXED = ((1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
          (17, 127), (17, 513), (17, 1000))
_SMALL = ((1, 768), (8, 1024), (17, 127), (17, 513), (17, 1000))
_LARGE = ((1024, 1024), (4096, 1024), (16384, 1024), (65536, 1024), (131072, 1024))
_LEGACY_SMALL = ((1, 256), (4, 512), (8, 768), (17, 127))
_LEGACY_LARGE = ((32, 1536), (8, 4096), (1, 8192), (64, 8192))


def _tb(i):
    return (2**i // 1024, 1024)


_TB_SWEEP = tuple(_tb(i) for i in range(12, 28))
_SHAPE_SUITES = {
    "mixed": _MIXED,
    "all": _MIXED,
    "small": _SMALL,
    "large": _LARGE,
    "legacy_small": _LEGACY_SMALL,
    "legacy_large": _LEGACY_LARGE,
    "tb_small": tuple(_tb(i) for i in range(12, 17)),
    "tb_medium": tuple(_tb(i) for i in range(17, 23)),
    "tb_large": tuple(_tb(i) for i in range(23, 28)),
    "tb_mixed": tuple(_tb(i) for i in (12, 14, 16, 18, 20, 22, 24, 26, 27)),
    "tb_sweep": _TB_SWEEP,
    **{f"tb_i{i}": (_tb(i),) for i in range(12, 28)},
}


def _workloads(shapes):
    return tuple(
        Workload(dims=dict(rows=rows, hidden=hidden), dtype=dtype)
        for rows, hidden in shapes
        for dtype in ("float32", "float16", "bfloat16")
    )


def _liger_factory():
    from liger_kernel.ops.layer_norm import layer_norm_backward, layer_norm_forward

    def forward(x, weight, bias, eps):
        y, x_2d, mean, rstd, _block_size, _num_warps = layer_norm_forward(
            x, weight, bias, eps
        )
        return y, (x_2d, weight, bias, mean, rstd)

    def backward(dy, saved):
        x_2d, weight, bias, mean, rstd = saved
        dy_2d = dy.view(-1, dy.shape[-1]).contiguous()
        return layer_norm_backward(dy_2d, x_2d, weight, bias, mean, rstd)

    return forward, backward


measure_liger_baseline = make_pair_baseline(
    _liger_factory, ("x", "weight", "bias", "eps")
)

op = declare_op(
    name="layernorm",
    forward="evograd.ops.layernorm.forward_ref:layernorm_forward_ref",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Active("bias", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
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
    benchmark=_workloads(_MIXED),
    benchmark_suites={name: _workloads(shapes) for name, shapes in _SHAPE_SUITES.items()},
    performance_baselines={"liger": measure_liger_baseline},
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    make_inputs=make_layernorm_inputs,
)
