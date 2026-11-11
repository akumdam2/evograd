"""Operator declaration: the LM head plus the causal-LM cross entropy (site ``lm_head_ce``).

The boundary is the end of ``Qwen3ForCausalLM.forward`` on a training step:

    x [rows, hidden]            the final norm's output, rows = B*T
      -> lm_head (the tied embedding, no bias), bf16 logits [rows, vocab]
      -> float32, mean cross entropy over targets != -100
      -> loss [] float32

Once per step. ``target`` is the labels as ``ForCausalLMLoss`` hands them to
``fixed_cross_entropy``: padded with ``-100``, shifted by one and flattened, so
each sequence's last row is ignored. A fused kernel need not materialize the
``[rows, vocab]`` logits (about 1.2 GB in BF16 and 2.5 GB in float32 here).

The case is config-derived (``hf_config``: :meth:`ModelConfig.flce_dims` of
Qwen3-0.6B) at the harvested run's token count.
"""

from evograd.benchmark.topdown import load_snapshot as _load_snapshot
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.opdecl.models import QWEN3_0_6B

_WORKLOAD = "qwen3_0_6b"
_SNAPSHOT = _load_snapshot(_WORKLOAD)
_FREE = {"tokens": _SNAPSHOT["batch_size"] * _SNAPSHOT["seq_len"]}
_CASE_DIMS = QWEN3_0_6B.flce_dims(**_FREE)
#: The harvested lm_head GEMM of the same run, which this boundary contains.
_LM_HEAD = next(c["dims"] for c in _SNAPSHOT["level1"]["linear_no_bias"]["configurations"]
                if c["provenance"]["component"] == "lm_head")
assert (_LM_HEAD["M"], _LM_HEAD["K"], _LM_HEAD["N"]) == (
    _CASE_DIMS["rows"], _CASE_DIMS["hidden"], _CASE_DIMS["vocab"]), _LM_HEAD

_BENCHMARK = (
    Workload(
        dims=_CASE_DIMS,
        dtype="bfloat16",
        provenance=Provenance(model=_WORKLOAD, component="flce", free=_FREE,
                              source="hf_config"),
    ),
)


def make_qwen3_lm_head_ce_inputs(torch, op, workload, device="cuda"):
    """Hidden states and a head at an initialised model's scale; the last row ignored."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    rows, hidden, vocab = dims["rows"], dims["hidden"], dims["vocab"]
    torch.manual_seed(rows * 100003 + hidden * 1009 + vocab)
    target = torch.randint(0, vocab, (rows,), device=device, dtype=torch.int64)
    target[-1] = -100
    return {
        "x": torch.randn((rows, hidden), device=device, dtype=dtype),
        "weight": (torch.randn((vocab, hidden), device=device, dtype=torch.float32)
                   * hidden**-0.5).to(dtype),
        "target": target,
        # What the step's backward seeds: d loss = 1.
        "dloss": torch.ones((), device=device, dtype=torch.float32),
    }


op = declare_op(
    name="qwen3_lm_head_ce",
    level=2,
    family="loss",
    forward=(
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.lm_head_ce.reference:"
        "qwen3_lm_head_ce_forward"
    ),
    dims=("rows", "hidden", "vocab"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[vocab, hidden]"),
        Inactive("target", "[rows]", dtype="int64"),
    ),
    output=Active("loss", "[]", dtype="float32"),
    parameter_args=("weight",),
    forward_semantics=(
        "logits = x @ weight^T in x's dtype (float32 accumulation, rounded to x's "
        "dtype), [rows, vocab]; upcast to float32; loss = mean over the rows whose "
        "target != -100 of logsumexp(logits_row) - logits_row[target], a float32 "
        "scalar (no label smoothing, no z-loss, no class weights). Rows with target "
        "-100 contribute nothing. The logits need not be materialized."
    ),
    backward_semantics=(
        "Return dx and dweight IN THIS ORDER; target gets none. With n the number of "
        "rows whose target != -100: dlogits = dloss * (softmax(logits) - "
        "onehot(target)) / n in float32 on those rows and 0 on ignored rows, rounded "
        "to x's dtype; dx = dlogits @ weight ([rows, hidden]); dweight = dlogits^T @ x "
        "([vocab, hidden]). Accumulate in float32; each gradient in its input's dtype."
    ),
    extra_constraints=(
        "weight is lm_head.weight, the tied input embedding. target is already shifted "
        "(labels padded with -100 and shifted by one, as ForCausalLMLoss does), so each "
        "sequence's last row is -100. The kernel runs inside a CUDA-graph-captured "
        "training step: no host synchronization (no .item(), no host-side count of "
        "valid targets, no data-dependent shapes)."
    ),
    grad_order=("dx", "dweight"),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("x", "weight"),
    make_inputs=make_qwen3_lm_head_ce_inputs,
)
