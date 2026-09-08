"""Operator declaration: Qwen3's gated (SwiGLU) MLP block.

The first task in this repository whose shape was **observed** rather than
chosen. Every dimension comes from one canonical Qwen3-0.6B training step:
batch 2, sequence 2048, BF16, SDPA, ``use_cache=False``. That step was harvested
into a manifest, one decoder layer was captured and verified to replay
standalone, and the ``Qwen3MLP`` invocation inside that replay is what this
declaration describes.

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
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import model_workloads as _model_workloads
from evograd.opdecl.tolerance import ReductionScaledAtol

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
    "verified standalone Qwen3DecoderLayer replay",
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

#: The correctness cases are the same block at a size a CPU test can run. They
#: are not the model's shape and say so.
_SHRUNK = Provenance(
    model="qwen3_0_6b",
    component="swiglu_mlp",
    free={},
    source="handpicked",
    scaled=True,
    note=(
        "hidden and intermediate reduced from Qwen3-0.6B's 1024/3072 so the "
        "correctness cases run on CPU in a normal test; the gate/up/down "
        "structure and the 3x hidden-to-intermediate ratio are preserved"
    ),
)

#: The second harvested architecture at this boundary. Llama-3-8B runs the gated MLP block
#: at wider dims -- same computation, different widths -- so the declaration
#: carries its observed configuration as its own suite rather than letting a
#: Llama-derived task be timed at Qwen3's shape.
#:
#: Derived from the frozen model configuration, not from a snapshot: the shape
#: is a property of the architecture, and ``tests/test_provenance`` re-derives
#: it. Batch 2 x sequence 2048 is the canonical Level-4 step for both models.
#: Measured at this shape, and only this shape. `out` accumulates over `I`,
#: which is 14336 here against the anchor's 96, and `ReductionScaledAtol` does
#: not model that: its `result_dims` count output *elements*, and `out` carries
#: no `reduction_dims` entry. The two measurements available are
#:
#:     grid {B:2,T:16,H:32,I:96}      out required_t 4.630e-03
#:     canonical Llama-3-8B           out required_t 3.208e-02   (6.93x)
#:
#: and the hook supplies 2.10x of that. Adding a sqrt(I) walk term would supply
#: 12.2x, which is a hole rather than a gate -- the observed growth is roughly
#: I**0.39, sub-square-root, as fp32-accumulating tensor cores produce. Two
#: points do not justify a growth law, so the shortfall is carried as a
#: multiplier on `out` alone (see `tolerance_multipliers` below) rather than
#: as a wider base on this workload: a workload-level atol widens all five
#: results, and the negative controls measured what that costs -- fault
#: detection on the four gradients degraded from 2% to 5-10% for headroom
#: none of them needed.
_LLAMA_OBSERVED = _model_workloads(
    LLAMA_3_8B,
    "swiglu_mlp",
    ({"batch": 2, "seq": 2048},),
    ("bfloat16",),
)

_BENCHMARK = (
    Workload(
        dims={"B": _BATCH, "T": _SEQ, "H": _HIDDEN, "I": _INTERMEDIATE},
        dtype="bfloat16",
        provenance=_OBSERVED,
    ),
)

_CORRECTNESS = tuple(
    Workload(
        dims={"B": batch, "T": seq, "H": hidden, "I": 3 * hidden},
        dtype=dtype,
        provenance=_SHRUNK,
    )
    for batch, seq, hidden, dtype in (
        (1, 8, 16, "float32"),
        (2, 16, 32, "float32"),
        (2, 16, 32, "bfloat16"),
    )
)


#: The correctness case the declared multipliers were measured on. The hook is
#: the identity here and below it, so adding it cannot loosen this grid.
_ANCHOR = {"B": 2, "T": 16, "H": 32, "I": 96}

#: Only the three weight gradients contract over tokens; `out` and `dx` keep the
#: token axis and so carry no reduction term. Measured exponent of the required
#: atol against reduction length, holding widths fixed: 0.35-0.39 (see the
#: scaling study), against the 0.5 the random-walk term assumes -- the law is
#: conservative for this operator, which the reported margin reflects.
_REDUCTION_SCALED = ReductionScaledAtol(
    anchor_dims=_ANCHOR,
    reduction_dims={
        "dgate_weight": ("B", "T"),
        "dup_weight": ("B", "T"),
        "ddown_weight": ("B", "T"),
    },
    result_dims={
        "out": ("B", "T", "H"),
        "dx": ("B", "T", "H"),
        "dgate_weight": ("I", "H"),
        "dup_weight": ("I", "H"),
        "ddown_weight": ("H", "I"),
    },
    gain=2.0,
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
        "evograd.ops.level2.qwen3_swiglu_mlp.forward_ref:qwen3_swiglu_mlp_forward_ref"
    ),
    # The eager baseline is timed through the spelling Qwen3MLP actually
    # executes, not through the float32-accumulated oracle. Timing the oracle
    # would compare a candidate against a slower-than-real PyTorch and inflate
    # every speedup; `verify_runtime_forward` proves the two agree before any
    # timing is trusted.
    runtime_forward=(
        "evograd.ops.level2.qwen3_swiglu_mlp.forward_ref:qwen3_swiglu_mlp_forward_hf"
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
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={
        "qwen3_0_6b_observed": _BENCHMARK,
        # The same boundary in Llama-3-8B, at its own observed width.
        "llama_3_8b_observed": _LLAMA_OBSERVED,
    },
    memory_inputs=("x", "gate_weight", "up_weight", "down_weight"),
    # Measured, not chosen. `evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp calibrate`
    # compares the declared float32-accumulated forward against
    # `runtime_forward` -- the spelling the model runs, and therefore the
    # smallest disagreement any correct implementation can have with the oracle
    # -- on every correctness workload and on the canonical [2, 2048, 1024]
    # invocation, and reports the smallest base `t` for which
    # `allclose(atol=ma*t, rtol=t)` accepts each result.
    #
    # float32: exactly 0.0 everywhere. The upcast is a no-op when the inputs are
    # already float32, so the two spellings are bit-identical and the tolerance
    # is the repository's ordinary float32 value rather than a measured bound.
    #
    # bfloat16: the disagreement is one rounding of the SwiGLU intermediate,
    # propagated. Relative to each tensor's scale it is remarkably uniform --
    # 2.8e-3 to 7.6e-3, i.e. 0.7 to 2 BF16 epsilons -- across both the synthetic
    # correctness cases and the real Layer-14 tensors. What varies is how much
    # cancellation each reduction has, which is what the atol multipliers exist
    # to absorb.
    tolerances={
        "float32": (2e-5, 2e-5),
        "bfloat16": (1e-2, 1e-2),
    },
    # Each multiplier is the measured minimum at base 1e-2, times a 1.5 safety
    # margin, rounded up to one decimal. The three weight gradients need the
    # largest because they reduce over all B*T tokens and cancel; `dx` needs a
    # small one for the same reason at a shorter contraction. `out` needed none
    # (measured 1.00) on the Qwen3 grid and got one only when the wider
    # Llama-3-8B intermediate was measured -- see the note on its entry below.
    #
    #   result         measured min ma at t=1e-2   declared
    #   dx                         1.48              2.3
    #   dgate_weight               2.42              3.7
    #   dup_weight                 3.22              4.9
    #   ddown_weight               4.33              6.5
    tolerance_multipliers={
        # `out` accumulates over `I`, which the reduction hook does not model
        # (its `result_dims` count output elements and `out` has no
        # `reduction_dims` entry). Measured at Llama-3-8B's 14336-wide
        # intermediate: the harvested invocation needs atol 3.242e-02 against
        # the 2.098e-02 the hook supplies, so 2.4x carries it with the 1.5x
        # margin (0.0504 / 0.0324 = 1.55x). Scoped to this result because only
        # this result needed it -- the four gradients require 5e-08..5e-07 at
        # that shape and keep their measured, tighter gates.
        "out": (2.4, 2.4),
        "dx": (2.3, 1.0),
        "dgate_weight": (3.7, 1.0),
        "dup_weight": (4.9, 1.0),
        "ddown_weight": (6.5, 1.0),
    },
    # Those multipliers were measured on the correctness grid, whose longest
    # token reduction is 32 terms. The observed workload's is 4096, and a
    # constant cannot describe a quantity that grows with the sum. Measured on a
    # GH200 with `calibrate inventory`: the three weight gradients needed 8.0x
    # to 9.1x more absolute tolerance at the observed shape than the constants
    # allow, while `out` and `dx` -- which have no token reduction -- needed
    # none. The hook supplies exactly that growth and is the identity at and
    # below the anchor, so every correctness case keeps the tolerance it has.
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_qwen3_swiglu_mlp_inputs,
)
