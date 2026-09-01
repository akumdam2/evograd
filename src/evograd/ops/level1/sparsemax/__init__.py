"""Operator declaration: sparsemax."""

from evograd.opdecl import Active, Provenance, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP_FP32,
)
from evograd.ops._common import (
    fixed_shape_suites,
    model_workloads,
    dtype_for,
    log_distance_weight,
    make_pair_baseline,
    regime_suites,
    standard_correctness,
    workloads_2d,
)

_TOLERANCES = {"float32": (2e-5, 2e-5)}
_SHAPES = (
    (4096, 128), (4096, 256), (4096, 512), (4096, 1024),
    (4096, 2048), (4096, 4096), (4096, 8192), (4096, 12288),
    (4096, 16384), (4096, 30522), (4096, 32768), (2048, 49152),
    (2048, 65536), (1024, 98304), (1024, 128256),
)
_SPLIT = LLAMA_VOCAB_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(_SHAPES, ("float32",), tolerances=_TOLERANCES)

# Sparsemax is the one operator here that no shipped model actually contains, so
# its shapes cannot be derived the way the others' are. The v1 grid claimed
# `llama_3_8b/logits` — sparsemax substituted for softmax over a 128256-wide
# vocabulary — and that claim is too strong. Sparsemax's published use is
# attention and moderate-width classification, and the evidence that nobody runs
# it at vocabulary width is that Liger's kernel cannot: it needs one row per
# Triton block and refuses anything past 65536 columns. A benchmark grid that
# only real implementations cannot execute is measuring the wrong thing.
#
# The timed grid is therefore honest about being chosen rather than derived, and
# spans the width range implementations do support, up to the 65536 boundary.
# The 98304 and 128256 rows stay in `coverage` (untimed) and in the `legacy`
# suite, so the point at which implementations stop is still on record.
_PROVENANCE = Provenance(
    model="sparsemax_paper",
    component="row_projection",
    source="handpicked",
    note=(
        "no shipped architecture contains sparsemax, so its width is chosen "
        "rather than derived: 32768 is the scale of a mid-sized vocabulary "
        "(Llama-2's is 32000) and sits inside the range Triton implementations "
        "support, unlike Llama-3's 128256. Rows keep the token sweep the other "
        "vocabulary-width operators use, so the shape regimes still split"
    ),
)
#: Rows sweep tokens exactly as before — the regime split is defined on rows, and
#: keeping it is what lets the small/large specialists stay meaningful. Only the
#: width changed.
_TIMED_SHAPES = tuple((tokens, 32768) for tokens in LLAMA_VOCAB_TOKEN_SWEEP_FP32)
_BENCHMARK = tuple(
    Workload(
        dims={"rows": rows, "cols": cols},
        dtype="float32",
        atol=_TOLERANCES["float32"][0],
        rtol=_TOLERANCES["float32"][1],
        provenance=_PROVENANCE,
    )
    for rows, cols in _TIMED_SHAPES
)
# Untimed, but kept on record: the widths a Llama-3 sized vocabulary would need,
# which is exactly where Triton implementations stop. A candidate must still run
# them, so "sparsemax at vocabulary width" remains a coverage question rather
# than disappearing from the benchmark.
_COVERAGE = _BENCHMARK + tuple(
    Workload(
        dims={"rows": rows, "cols": cols},
        dtype="float32",
        atol=_TOLERANCES["float32"][0],
        rtol=_TOLERANCES["float32"][1],
        provenance=_PROVENANCE,
    )
    for rows, cols in ((1024, 98304), (1024, 128256))
)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.sparsemax.liger import make_liger_sparsemax_autograd_pair_fns

    return make_liger_sparsemax_autograd_pair_fns()


op = declare_op(
    name="sparsemax",
    level=1,
    family="reduction",
    forward="evograd.ops.level1.sparsemax.forward_ref:sparsemax_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]", dtype="float32"),),
    output=Active("out", "[rows, cols]", dtype="float32"),
    parameter_args=(),
    forward_semantics="Project each row onto the probability simplex with sparsemax.",
    backward_semantics=(
        "On support S={i:out_i>0}, return dx_i=dout_i-mean_S(dout); "
        "return zero outside S."
    ),
    correctness=workloads_2d(
        ((8, 64), (17, 128), (32, 256), (64, 512)),
        ("float32",),
        tolerances=_TOLERANCES,
    ),
    coverage=_COVERAGE,
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("x",))},
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
