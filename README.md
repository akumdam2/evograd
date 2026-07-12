# evograd

**Generate, verify, evolve, and benchmark Triton backward kernels from a trusted
PyTorch forward function.**

Evograd turns an operator definition into a training-ready *autograd pair*: a
forward function that chooses what to save and a backward function that consumes
that saved state. It can generate the initial Triton implementation through
three research pipelines, verify it against PyTorch autograd, optimize it with
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve), and
measure its speed and saved-tensor memory.

```text
PyTorch forward reference + operator declaration
                         │
                         ▼
              Pipeline A, B, or C
                         │
                         ▼
             Triton forward/backward seed
                         │
                         ▼
          correctness vs. torch.autograd.grad
                         │
                         ▼
                OpenEvolve optimization
                         │
                         ▼
           latency + saved-memory benchmark
```

> [!IMPORTANT]
> The repository has comprehensive CPU-independent structural tests, but its
> Triton paths have not yet been run on a CUDA machine in this environment. See
> [Project status](#project-status) before using results in experiments.

## What evograd does

Evograd provides one workflow for the parts of backward-kernel research that
otherwise tend to be duplicated per operator:

- A typed declaration of inputs, outputs, shapes, differentiation activity,
  workloads, and tolerances.
- Three ways to generate a Triton autograd-pair seed.
- A PyTorch-autograd oracle derived from the declared forward reference.
- A hard correctness gate for the candidate forward and every requested
  gradient, including shape and dtype checks.
- A generic OpenEvolve evaluator with speed and memory-aware scoring policies.
- A benchmark harness for backward latency, full-step latency, and saved state.
- Automatic operator discovery: adding an operator does not require editing a
  central registry, evaluator, wrapper, or pipeline.

## Quick start

Evograd requires Python 3.10 or newer. Seed generation and benchmarking require
PyTorch, Triton, and a CUDA-capable machine. Pipelines A and C also require an
OpenAI-compatible LLM endpoint.

```bash
git clone https://github.com/akumdam2/evograd.git
cd evograd
pip install -e ".[gpu,llm]"
```

For LLM-backed generation or OpenEvolve:

```bash
export OPENAI_API_KEY="..."
```

List the available operators:

```bash
evograd ops
```

Generate an LLM-free LayerNorm seed with Pipeline B:

```bash
evograd seed b \
    --op layernorm \
    --output-dir /tmp/B_layernorm
```

Verify the emitted autograd pair:

```bash
evograd verify \
    --op layernorm \
    /tmp/B_layernorm/initial_program_autograd_pair.py
```

Optimize it with OpenEvolve:

```bash
evograd evolve \
    --op layernorm \
    --seed /tmp/B_layernorm/initial_program_autograd_pair.py \
    --scoring speed_memory \
    --iterations 10 \
    --output-dir /tmp/evolve_layernorm
```

Benchmark the best candidate:

```bash
evograd bench \
    --op layernorm \
    --candidate /tmp/evolve_layernorm/evolved_best_program.py
```

## Seed-generation pipelines

All three pipelines target the same candidate API and use the same declaration,
oracle, and verification path.

| Pipeline | Inputs | Generation method | LLM required |
|---|---|---|---|
| **A — AtenIR + LLM** | Forward reference and extracted AtenIR backward graph | LLM plans, generates, and repairs fused Triton code | Yes |
| **B — primitive dispatch** | Forward reference and extracted AtenIR backward graph | Handwritten Aten-to-Triton primitive lowering followed by dispatch-free code generation | No |
| **C — forward only** | Forward source and declared contract | LLM derives the backward without seeing AtenIR; used as an ablation | Yes |

```bash
evograd seed a --op rmsnorm --output-dir /tmp/A_rmsnorm --model gpt-5.5
evograd seed b --op rmsnorm --output-dir /tmp/B_rmsnorm
evograd seed c --op rmsnorm --output-dir /tmp/C_rmsnorm --model gpt-5.5
```

A successful Pipeline B run writes:

```text
/tmp/B_rmsnorm/
├── atenir_graph.json
├── lowering_context.md
├── dispatch_program.py
├── initial_program_autograd_pair.py
└── verification_report.json
```

Pipeline B's handwritten primitive library lives in
`src/evograd/atenir/primitive_triton/`. A new operator works with Pipeline B
when every operation in its extracted AtenIR graph has a supported lowering.
Pipelines A and C do not require that primitive coverage.

## Operator declarations

Each operator is a self-contained package:

```text
src/evograd/ops/layernorm/
├── __init__.py       # declaration, workloads, inputs, optional baselines
└── forward_ref.py    # trusted PyTorch implementation
```

The declaration is the single source of truth:

```python
from evograd.opdecl import Active, Inactive, Workload, declare_op


op = declare_op(
    name="layernorm",
    forward="evograd.ops.layernorm.forward_ref:layernorm_forward_ref",
    dims=("rows", "hidden"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Active("bias", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    correctness=(
        Workload(dims={"rows": 8, "hidden": 64}, dtype="float32"),
        Workload(dims={"rows": 32, "hidden": 256}, dtype="float16"),
    ),
    benchmark=(
        Workload(dims={"rows": 1024, "hidden": 1024}, dtype="float16"),
    ),
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
    },
    forward_semantics="Row-wise LayerNorm over the final dimension.",
    backward_semantics="Return dx, dweight, and dbias.",
)
```

In plain language:

- `forward` identifies the trusted PyTorch function.
- `dims` names symbolic dimensions used in tensor shapes.
- `Active` marks a differentiable tensor whose gradient is required.
- `Inactive` marks an input that does not receive a gradient.
- `output` describes the **forward output**, not the backward kernel. Marking it
  `Active` means backward receives its upstream gradient, `dy` here.
- `correctness` contains cases used to reject broken candidates.
- `benchmark` contains realistic cases used to compare correct candidates.
- `tolerances` provides `(absolute_tolerance, relative_tolerance)` per dtype.

### Why `Active` and `Inactive`? What is Enzyme?

[Enzyme](https://enzyme.mit.edu/) is an automatic-differentiation system for
LLVM. It differentiates compiled programs and uses *activity analysis* to
determine which values participate in differentiation. The underlying concepts
are:

- A **constant** value is used by the computation but does not receive a
  derivative.
- A **duplicated** value carries both a primal value and its derivative shadow.

Evograd uses the simpler names `Active` and `Inactive` to expose the same
distinction directly. For example, EvoAttention declares `q`, `k`, `v`, and
`pair_bias` as `Active`, but its mask as `Inactive`; the generated backward returns
four gradients and no mask gradient.

Evograd does **not** depend on or invoke Enzyme. It uses PyTorch autograd as its
oracle and uses the Enzyme-inspired annotations as a compact interface for
generating wrappers, prompts, verification, and evaluator behavior.

## Candidate API

Generated and evolved programs implement a forward/backward pair:

```python
def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    return y, saved_state


def layernorm_backward_from_saved(dy, saved_state, eps=1e-5):
    return dx, dweight, dbias
```

The forward is allowed to choose what it saves. The saved state may mix tensors
with immutable scalar metadata; only tensor storage contributes to the memory
score. The declaration determines the backward return names and ordering, and
the generic autograd binding inserts `None` for every `Inactive` input.

## Correctness and evaluation

For every declared correctness workload, evograd:

1. Constructs deterministic inputs.
2. Runs the trusted forward reference.
3. Computes expected gradients with `torch.autograd.grad` for exactly the
   `Active` inputs.
4. Runs the candidate forward and backward.
5. Checks forward and gradient values, shapes, and dtypes with the declared
   per-dtype and per-output tolerances.

Only candidates that pass every correctness case are benchmarked. The harness
reports candidate forward latency, backward-from-saved latency, forward plus
backward latency, autograd-bound full-step latency, baseline latency, and saved
tensor bytes.

Available scoring policies are:

| Policy | Objective |
|---|---|
| `speed` | Backward speedup |
| `speed_memory` | Weighted backward/full-step speedup with a saved-memory penalty |
| `speed_memory_min` | Minimum of backward and full-step speedup, memory-penalized |
| `speed_memory_min_geomean` | Geometric-mean minimum speedup, memory-penalized |
| `speed_memory_min_weighted_geomean` | Weighted geometric mean with a worst-case guard and memory penalty |

The default is `speed_memory`. LayerNorm also supports an optional Liger
baseline:

```bash
pip install -e ".[baselines]"
evograd bench --op layernorm --candidate best.py --baseline liger
```

## Supported operators

| Operator | Forward | Requested gradients | Benchmark dtypes |
|---|---|---|---|
| `layernorm` | LayerNorm | `dx`, `dweight`, `dbias` | fp32, fp16, bf16 |
| `rmsnorm` | RMSNorm | `dx`, `dweight` | fp32, fp16 |
| `matmul` | `a @ b` | `da`, `db` | fp32, fp16 |
| `linear` | `x @ weight.T + bias` | `dx`, `dweight`, `dbias` | fp32, fp16 |
| `layernorm_linear` | LayerNorm followed by Linear | `dx`, `dlinear_weight`, `dweight`, `dbias` | fp32, fp16 |
| `evoattention` | AlphaFold3-style attention with pair bias | `dq`, `dk`, `dv`, `d_pair_bias` | fp16, bf16 |

LayerNorm includes the legacy and TritonBench shape suites:

```bash
evograd bench \
    --op layernorm \
    --candidate best.py \
    --suite tb_mixed \
    --dtype float16
```

## Adding an operator

Create one package containing a declaration and forward reference:

```text
src/evograd/ops/<name>/
├── __init__.py
└── forward_ref.py
```

The package's `__init__.py` must expose a module-level `op` whose declared name
matches the package name. Evograd discovers it automatically. Once declared,
the operator is available to `evograd ops`, all three seed pipelines,
verification, evolution, and benchmarking without additional evaluator or
wrapper files.

For operators with semantic masks, scaled weights, integer-like inputs, or
special initialization, define a declaration-local `make_inputs` hook. If
Pipeline B encounters an unsupported Aten operation, add its lowering under
`src/evograd/atenir/primitive_triton/`.

The current declaration model targets deterministic, single-tensor-output,
dense floating-point operators. Multiple outputs, sparse gradients, integer
index tensors, and stateful operators require extensions to the declaration and
input/extraction APIs.

## Repository layout

```text
src/evograd/
├── opdecl/                    # declaration types, oracle, binding, verification
├── ops/                       # one self-contained package per operator
├── atenir/
│   ├── extract.py             # PyTorch/autograd graph extraction
│   ├── compose.py             # serialized graph execution
│   └── primitive_triton/      # Pipeline B's handwritten Triton primitives
├── pipelines/
│   ├── a_atenir_llm/          # AtenIR-grounded LLM synthesis
│   ├── b_dispatch/            # LLM-free primitive lowering and code generation
│   ├── c_forward_only/        # forward-only LLM ablation
│   └── shared/
├── evolve/                    # OpenEvolve evaluator, scoring, and run wrapper
├── bench/                     # generic latency and memory harness
└── cli.py                     # evograd command-line interface
```

OpenEvolve is a package dependency rather than vendored or forked source.
Evograd supplies the initial program, evaluator, generated configuration, and
operator-specific environment; OpenEvolve supplies the evolutionary search,
program database, LLM orchestration, and checkpointing.

## Project status

Evograd is the declaration-driven successor to the AtenIR, pipeline, and Triton
backward-benchmark work previously developed in an OpenEvolve fork.

Locally validated:

- 49 unit tests pass.
- Eight torch-facing unit tests are present but skip on machines without
  PyTorch.
- All six declarations match the legacy correctness cases, benchmark cases,
  per-output tolerances, input distributions, and seed recipes.
- The AtenIR implementation is byte-identical to the legacy implementation.
- The OpenEvolve 0.2.27 model-list configuration loads successfully.
- Source compilation, wheel build, clean wheel installation, automatic operator
  discovery, and package-data checks pass.

Still required on a CUDA machine before relying on experimental results:

```bash
PYTHONPATH=src python scripts/gpu_smoke.py --oracle-only
PYTHONPATH=src python -m unittest discover tests

PYTHONPATH=src python scripts/gpu_parity.py \
    --op layernorm \
    --old-repo /path/to/openevolve-fork \
    --candidate /path/to/legacy/layernorm/candidate.py
```

Repeat parity for EvoAttention, then run one end-to-end Pipeline B seed and a
short LayerNorm evolution.

## Testing

```bash
PYTHONPATH=src python -m unittest discover tests
```
