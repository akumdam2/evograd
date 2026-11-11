"""Operator declaration: Qwen3's gated (SwiGLU) MLP block.

The first task in this repository whose shape was **observed** rather than
chosen. Every dimension comes from one canonical Qwen3-0.6B training step:
batch 2, sequence 2048, BF16, SDPA, ``use_cache=False``. That step was harvested
into a manifest, and the ``Qwen3MLP`` invocation it records is what this
declaration describes; the real-run capture records every one of its calls, and
the model reference replays each of them.

The provenance is therefore checkable twice over, in two independent ways:

* ``Provenance(model="qwen3_0_6b", component="swiglu_mlp", ...)`` re-derives the
  benchmark dims from the published Qwen3-0.6B configuration, exactly as every
  other ``hf_config`` workload here does.
* :data:`HARVEST` carries the observed record itself -- configuration id,
  invocation frequency, and the module path of every layer that produced it --
  read at import time from the tracked workload snapshot. The benchmark dims are
  *derived from* that record rather than typed alongside it, so a snapshot that
  stopped agreeing with this file would fail at import.

Neither path touches ``results/``. The snapshot is a small tracked JSON file;
the manifest and the capture artifacts it was extracted from are local results
that most machines will not have.
"""

from evograd.benchmark.topdown import load_snapshot as _load_snapshot
from evograd.benchmark.topdown import load_snapshot_task as _snapshot_task
from evograd.opdecl import Active, Provenance, Workload, declare_op

#: The harvested workload these dims came from; also the ``Provenance``
#: model key, so the declaration and the snapshot cannot disagree about
#: which run they describe.
_WORKLOAD = "qwen3_0_6b"

_SNAPSHOT = _load_snapshot(_WORKLOAD)

#: The harvested record this declaration is derived from. Structured, not prose:
#: the configuration id it collapsed into, how many times it ran in one step, and
#: which module produced each of those runs.
HARVEST = _snapshot_task(_WORKLOAD, "qwen3_swiglu_mlp")

#: 28 -- once per decoder layer. Every Qwen3-0.6B layer's MLP deduplicated into a
#: single configuration, so one kernel serves all of them.
FREQUENCY = HARVEST["frequency"]

#: The chain this declaration stands on, each link independently verifiable.
PROVENANCE_CHAIN = (
    f"canonical workload {_SNAPSHOT['workload_id']}",
    f"harvest manifest {_SNAPSHOT['manifest_hash']}",
    f"{_SNAPSHOT['representative_layer']['module_path']} "
    f"(event ordinal {_SNAPSHOT['representative_layer']['event_ordinal']})",
    "real-run capture, every call replayed by the model reference",
    f"Qwen3MLP invocation, harvested configuration {HARVEST['config_id']}",
    "qwen3_swiglu_mlp",
)

_BATCH, _SEQ, _HIDDEN = HARVEST["input_shapes"][0]["shape"]
_INTERMEDIATE = HARVEST["attrs"]["intermediate_size"]
assert _HIDDEN == HARVEST["attrs"]["hidden_size"], HARVEST
assert HARVEST["attrs"]["hidden_act"] == "silu", HARVEST
assert (
    HARVEST["output_shapes"][0]["shape"] == HARVEST["input_shapes"][0]["shape"]
), HARVEST

_DIMS = ("B", "T", "H", "I")

#: The observed configuration, re-derivable from the published Qwen3-0.6B config.
_OBSERVED = Provenance(
    model="qwen3_0_6b",
    component="swiglu_mlp",
    free={"batch": _BATCH, "seq": _SEQ},
    source="hf_config",
)


_BENCHMARK = (
    Workload(
        dims={"B": _BATCH, "T": _SEQ, "H": _HIDDEN, "I": _INTERMEDIATE},
        dtype="bfloat16",
        provenance=_OBSERVED,
    ),
)


def make_qwen3_swiglu_mlp_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["H"] * 101 + dims["I"]
    )
    # Scale the projections the way an initialised model does, so the reference
    # activations stay in the range the observed capture lives in rather than
    # saturating SiLU.
    x = torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype)
    gate_weight = (
        torch.randn((dims["I"], dims["H"]), device=device, dtype=torch.float32)
        * dims["H"] ** -0.5
    ).to(dtype)
    up_weight = (
        torch.randn((dims["I"], dims["H"]), device=device, dtype=torch.float32)
        * dims["H"] ** -0.5
    ).to(dtype)
    down_weight = (
        torch.randn((dims["H"], dims["I"]), device=device, dtype=torch.float32)
        * dims["I"] ** -0.5
    ).to(dtype)
    dout = torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype)
    return {
        "x": x,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
        "dout": dout,
    }


op = declare_op(
    name="qwen3_swiglu_mlp",
    level=2,
    family="mlp",
    forward=(
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference:qwen3_swiglu_mlp_forward_ref"
    ),
    # The eager baseline is timed through the spelling Qwen3MLP actually
    # executes, not through the float32-accumulated oracle. Timing the oracle
    # would compare a candidate against a slower-than-real PyTorch and inflate
    # every speedup. The model reference (`reference_op`) differentiates this
    # spelling itself.
    runtime_forward=(
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference:qwen3_swiglu_mlp_forward_hf"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("gate_weight", "[I, H]"),
        Active("up_weight", "[I, H]"),
        Active("down_weight", "[H, I]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("gate_weight", "up_weight", "down_weight"),
    forward_semantics=(
        "Qwen3's gated MLP block. gate = x @ gate_weight.T; up = x @ up_weight.T; "
        "hidden = silu(gate) * up computed with float32 accumulation and cast "
        "back to x's dtype; out = hidden @ down_weight.T. x is [B, T, H], "
        "gate_weight and up_weight are [I, H], down_weight is [H, I], out is "
        "[B, T, H]. silu(v) = v * sigmoid(v). Accumulate every GEMM in float32. "
        "Do not call F.linear, torch.matmul, @, F.silu, or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return gradients for x, gate_weight, up_weight, and down_weight IN "
        "THIS ORDER. With g = silu(gate) * up (the float32 intermediate) and "
        "dh = dout @ down_weight: ddown_weight = dout^T @ hidden (shape [H, I]); "
        "dup = dh * silu(gate); dgate = dh * up * dsilu(gate) where "
        "dsilu(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v))); "
        "dgate_weight = dgate^T @ x and dup_weight = dup^T @ x (both [I, H], "
        "summed over B and T); dx = dgate @ gate_weight + dup @ up_weight "
        "(shape [B, T, H]). Accumulate every reduction and matmul in float32 "
        "before casting each gradient to its input's dtype."
    ),
    extra_constraints=(
        "Observed from a real Qwen3-0.6B training step, not chosen: the timed "
        "shape is the [2, 2048, 1024] BF16 activation that all 28 decoder "
        "layers pass to their MLP. B and T are kept as separate axes because "
        "that is the shape the module actually received; a kernel may flatten "
        "them internally. gate_weight and up_weight are separate tensors, as "
        "Qwen3 stores them -- not one fused [2I, H] matrix. No biases. All "
        "floating tensors are contiguous CUDA tensors."
    ),
    grad_order=("dx", "dgate_weight", "dup_weight", "ddown_weight"),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("x", "gate_weight", "up_weight", "down_weight"),
    make_inputs=make_qwen3_swiglu_mlp_inputs,
)
