"""Operator declaration: fused linear + cross entropy (FLCE).

Liger's largest memory win, and the loss kernel
``apply_liger_kernel_to_llama`` installs by default in place of the plain
cross-entropy one. Evograd had ``linear`` and ``cross_entropy`` separately but
never the fusion, so the operator that actually runs in a Llama training step
was not in the benchmark.

Level 2: it composes the lm_head projection with the loss, and the whole point
is optimizing across that boundary. A candidate that materializes the
``[rows, vocab]`` logits gets the right answer and loses on memory, which is
exactly the trade-off the saved-memory metric is there to expose.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP,
)
from evograd.ops._common import (
    fixed_shape_suites,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    regime_suites,
)


def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def make_flce_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    rows, hidden, vocab = dims["rows"], dims["hidden"], dims["vocab"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + hidden * 1009 + vocab)

    x = torch.randn((rows, hidden), device=device, dtype=dtype) * 0.5
    # lm_head weights at their real scale. A unit-variance [vocab, hidden]
    # projection would produce logits with a standard deviation of sqrt(hidden),
    # i.e. ~64 here, which saturates the softmax and makes the loss surface
    # nothing like a trained model's.
    weight = (
        torch.randn((vocab, hidden), device=device, dtype=torch.float32)
        * hidden**-0.5
    ).to(dtype)
    target = torch.randint(0, vocab, (rows,), device=device, dtype=torch.int64)
    # Exactly 1.0, not a random draw: cross entropy is the last layer of a real
    # training step, so its upstream gradient is 1.0 by construction. Liger's
    # backward early-outs on that value, and a benchmark that fed it something
    # else would be measuring a rescale no training run performs.
    dloss = torch.ones((), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "target": target, "dloss": dloss}


def _liger_factory():
    from evograd.ops.level2.fused_linear_cross_entropy.liger import (
        make_liger_flce_autograd_pair_fns,
    )

    return make_liger_flce_autograd_pair_fns()


_TOLERANCES = {
    "float32": (2e-5, 1e-3),
    "float16": (5e-4, 1e-2),
    "bfloat16": (2e-3, 2e-2),
}

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "flce",
    tuple({"tokens": tokens} for tokens in LLAMA_VOCAB_TOKEN_SWEEP),
    ("float16", "bfloat16"),
    tolerances=_TOLERANCES,
)

_CORRECTNESS = tuple(
    Workload(
        dims=dict(rows=rows, hidden=hidden, vocab=vocab),
        dtype=dtype,
        atol=_TOLERANCES[dtype][0],
        rtol=_TOLERANCES[dtype][1],
    )
    for rows, hidden, vocab in ((8, 64, 512), (17, 128, 1000))
    for dtype in ("float32", "float16", "bfloat16")
)

op = declare_op(
    name="fused_linear_cross_entropy",
    forward=(
        "evograd.ops.level2.fused_linear_cross_entropy.forward_ref:"
        "fused_linear_cross_entropy_forward_ref"
    ),
    level=2,
    family="loss",
    dims=("rows", "hidden", "vocab"),
    args=(
        Active("x", "[rows, hidden]", note="final hidden states"),
        Active("weight", "[vocab, hidden]", note="lm_head weight, nn.Linear orientation"),
        Inactive("target", "[rows]", dtype="int64", note="hard class labels"),
    ),
    output=Active("loss", "[]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Compute logits = x @ weight.T and return their mean hard-label cross "
        "entropy against target, with ignore_index=-100, no label smoothing and "
        "no z-loss. Accumulate the projection and the logsumexp in float32. "
        "The point of this operator is to never materialize the [rows, vocab] "
        "logits tensor: process the rows in chunks so peak memory stays "
        "proportional to a chunk rather than to rows*vocab. Do not call "
        "F.linear, F.cross_entropy, torch.matmul, or autograd in the generated "
        "math."
    ),
    backward_semantics=(
        "Return (dx, dweight). With p = softmax(logits) and n the number of "
        "non-ignored rows: dlogits = dloss * (p - onehot(target)) / n, then "
        "dx = dlogits @ weight and dweight = dlogits.T @ x. target is inactive "
        "and receives no gradient. dx carries x's dtype and dweight carries "
        "weight's. Accumulate in float32. The same chunking the forward uses "
        "applies here: dlogits must not be materialized in full either."
    ),
    extra_constraints=(
        "Tensor layout notes:\n"
        "- x: [rows, hidden], weight: [vocab, hidden], target: [rows] int64\n"
        "- loss is a float32-accumulated scalar cast to x's dtype\n"
        "- dloss is 1.0 in every declared workload, because this operator is the "
        "last layer of a training step\n"
        "- vocab is 128256 (Llama-3), so a materialized logits tensor is 2.1 GB "
        "at 8192 rows in bfloat16 and the same again for its gradient. Chunking "
        "is not an optimization here, it is the contract."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_VOCAB_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
    },
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("x", "weight", "target"))
    },
    tolerances=_TOLERANCES,
    # int64 labels are not model state and must not count against the memory
    # ratio, the same treatment cross_entropy gives its targets.
    memory_inputs=("x", "weight"),
    regime_feature=_regime_feature,
    regime_split=LLAMA_VOCAB_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_VOCAB_REGIME_SPLIT),
    make_inputs=make_flce_inputs,
)
