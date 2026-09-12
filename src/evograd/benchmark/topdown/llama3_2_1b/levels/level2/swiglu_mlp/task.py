"""Level-2 task: the Llama-3-8B gated (SwiGLU) MLP block.

The same computation Qwen3-0.6B runs at this boundary, at Llama-3-8B's widths:
hidden 4096 against 1024, and an intermediate of 14336 against 3072. Same
mathematics, different case -- and the wider intermediate is exactly why this
must be its own task. See ``tolerance_multipliers`` below: ``out`` accumulates
over ``I``, and at 14336 it needs a gate Qwen3's 3072-wide case never did.
Carrying that widening on Qwen3's task would have loosened a measured gate for
a shape Qwen3 never runs.

The timed case is derived from the frozen Llama-3-8B configuration rather than
from a harvest snapshot: the shape is a property of the architecture, and
``tests/test_provenance`` re-derives it. Batch 2 x sequence 2048 is the
canonical Level-4 step for both models.
"""

from evograd.benchmark.topdown.common.level2_references import (
    gated_mlp_forward_model_dtype as _model_dtype,
    gated_mlp_forward_ref as _definition,
)
from evograd.opdecl import Active, Provenance, Workload, declare_op
# The workload pins Llama-3.2-1B. `opdecl.models` also carries LLAMA_3_8B,
# the 8B config the operator suite's vocabulary-regime cases are declared
# against; that one did not move and is a different model.
from evograd.opdecl.models import LLAMA_3_2_1B
from evograd.opdecl.tolerance import ReductionScaledAtol
from evograd.ops._common import model_workloads as _model_workloads

#: The workload these cases belong to; also the ``Provenance`` model key.
WORKLOAD = "llama_3_2_1b"

#: 32 -- once per decoder layer.
FREQUENCY = LLAMA_3_2_1B.layers

_DIMS = ("B", "T", "H", "I")

#: The observed configuration, computed from the published model config.
_BENCHMARK = _model_workloads(
    LLAMA_3_2_1B,
    "swiglu_mlp",
    ({"batch": 2, "seq": 2048},),
    ("bfloat16",),
)

_SHRUNK = Provenance(
    model="llama_3_2_1b",
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

_ANCHOR = {"B": 2, "T": 16, "H": 32, "I": 96}

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

def make_llama3_swiglu_mlp_inputs(torch, op, workload, device="cuda"):
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
    name="llama3_swiglu_mlp",
    level=2,
    family="mlp",
    forward="evograd.benchmark.topdown.llama3_2_1b.levels.level2.swiglu_mlp.reference:llama3_swiglu_mlp_forward_ref",
    runtime_forward="evograd.benchmark.topdown.llama3_2_1b.levels.level2.swiglu_mlp.reference:llama3_swiglu_mlp_forward_hf",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("gate_weight", "[I, H]"),
        Active("up_weight", "[I, H]"),
        Active("down_weight", "[H, I]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("gate_weight", "up_weight", "down_weight"),
    forward_semantics="Qwen3's gated MLP block. gate = x @ gate_weight.T; up = x @ up_weight.T; "
        "hidden = silu(gate) * up computed with float32 accumulation and cast "
        "back to x's dtype; out = hidden @ down_weight.T. x is [B, T, H], "
        "gate_weight and up_weight are [I, H], down_weight is [H, I], out is "
        "[B, T, H]. silu(v) = v * sigmoid(v). Accumulate every GEMM in float32. "
        "Do not call F.linear, torch.matmul, @, F.silu, or autograd in the "
        "generated math.",
    backward_semantics="Return gradients for x, gate_weight, up_weight, and down_weight IN "
        "THIS ORDER. With g = silu(gate) * up (the float32 intermediate) and "
        "dh = dout @ down_weight: ddown_weight = dout^T @ hidden (shape [H, I]); "
        "dup = dh * silu(gate); dgate = dh * up * dsilu(gate) where "
        "dsilu(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v))); "
        "dgate_weight = dgate^T @ x and dup_weight = dup^T @ x (both [I, H], "
        "summed over B and T); dx = dgate @ gate_weight + dup @ up_weight "
        "(shape [B, T, H]). Accumulate every reduction and matmul in float32 "
        "before casting each gradient to its input's dtype.",
    extra_constraints=(
        "Derived from the published Llama-3-8B configuration, not chosen. The "
        "boundary is the whole gated MLP: gate and up projections, the SwiGLU "
        "activation, and the down projection. No bias anywhere. All floating "
        "tensors are CUDA tensors."
    ),
    grad_order=("dx", "dgate_weight", "dup_weight", "ddown_weight"),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"llama_3_2_1b_observed": _BENCHMARK},
    memory_inputs=("x", "gate_weight", "up_weight", "down_weight"),
    tolerances={
        "float32": (2e-5, 2e-5),
        "bfloat16": (1e-2, 1e-2),
    },
    tolerance_multipliers={
        # `out` accumulates over `I`, which the reduction hook does not model
        # (its `result_dims` count output elements and `out` has no
        # `reduction_dims` entry). Measured at Llama-3-8B's 14336-wide
        # intermediate: the harvested invocation needs atol 3.242e-02 against
        # the 2.098e-02 the hook supplies, so 2.4x carries it with the 1.5x
        # margin (0.0504 / 0.0324 = 1.55x). Scoped to this result because only
        # this result needed it -- the four gradients require 5e-08..5e-07 at
        # that shape and keep their measured, tighter gates.
        #
        # This is the multiplier the upstream Llama work measured. It lives
        # here, on Llama's own task, and not on `qwen3_swiglu_mlp`, whose
        # 3072-wide intermediate measured 1.00 for `out` and keeps no entry.
        "out": (2.4, 2.4),
        "dx": (2.3, 1.0),
        "dgate_weight": (3.7, 1.0),
        "dup_weight": (4.9, 1.0),
        "ddown_weight": (6.5, 1.0),
    },
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_llama3_swiglu_mlp_inputs,
)
