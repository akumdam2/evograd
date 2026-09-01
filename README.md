# evograd

**Generate, verify, evolve, and benchmark Triton backward kernels from a trusted
PyTorch forward function.**

Evograd turns an operator definition into a training-ready _autograd pair_: a
forward function that chooses what to save and a backward function that consumes
that saved state. It can generate the initial Triton implementation through
four research pipelines, verify it against PyTorch autograd, optimize it with
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve), and
measure its speed and saved-tensor memory.

```text
PyTorch forward reference + operator declaration
                         │
                         ▼
            Pipeline A, B, C, or D
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
- Four ways to generate a Triton autograd-pair seed.
- A PyTorch-autograd oracle derived from the declared forward reference.
- A hard correctness gate for the candidate forward and every requested
  gradient, including shape and dtype checks.
- A generic OpenEvolve evaluator with speed and memory-aware scoring policies.
- A benchmark harness for backward latency, full-step latency, and saved state.
- A one-call file/callable interface that trains a generalist and optional
  shape-regime specialists, then emits a measured deployment dispatcher.
- Optional Nsight Compute-guided refinement with a strict keep-only-if-faster gate.
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

Run the complete path for a declared op:

```bash
evograd run \
    --op softmax \
    --iterations 10 \
    --gpus 1 \
    --output-dir /tmp/evograd_softmax
```

This generates and verifies a seed, evolves full/small/large programs when the
declaration defines shape regimes, measures them on the full grid, and writes a
deployed pair plus JSON and Markdown reports. `--gpus 3` runs the three
evolution groups concurrently on three visible GPUs.

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

`--save-programs` keeps every candidate the run evaluates under
`<output-dir>/programs` — one `.py` per distinct program named
`<UTC timestamp>_<score>_<sha1>`, a `.json` sidecar with its metrics, and an
`index.jsonl` line per evaluation (including duplicates and failures). Without
it, OpenEvolve returns only the best program and the rest of the population is
unrecoverable.

Benchmark the best candidate:

```bash
evograd bench \
    --op layernorm \
    --candidate /tmp/evolve_layernorm/evolved_best_program.py
```

Run the whole benchmark suite and get the cross-operator report:

```bash
evograd suite --candidates programs/ --out results/
evograd suite --level 1 --level 2 --out results/   # restrict to a level
```

This writes `suite_report.json` and `SUITE_RESULTS.md` with per-operator,
per-level, and overall full-step speedup, coverage, and saved memory. See
[the benchmark specification](docs/BENCHMARK.md) for what those numbers mean.

`bench` exits non-zero when anything fails and prints the failure to stderr.
With `--out`, the report is written either way: `report["ok"]` is the verdict,
`report["error"]` holds a setup failure (unknown op, candidate that raises on
import, a `--dtype` the declared benchmark suite has no cases for), and
`report["cases"][i]["error"]` holds a per-workload failure. Cases that did run
are still aggregated.

## One-call Python API

Known operators need only their name; the reviewed declaration supplies the
default forward and contract:

```python
from evograd import evograd

result = evograd(op="softmax", output_dir="/tmp/softmax", iterations=10)
print(result.program)
print(result.report)
```

A forward may instead be `path.py:function` or a named, self-contained Python
callable. For a known op it replaces only the forward reference and keeps the
reviewed argument/activity/workload contract. For an unknown name, evograd
first synthesizes a declaration-native contract, imports and validates it, and
records it under `<output_dir>/scaffold/` before seed generation:

```python
import torch
from evograd import evograd

def squared_relu(x):
    return torch.relu(x).square()

result = evograd(squared_relu, op="my_squared_relu", output_dir="/tmp/my_op")
```

Callable snapshots reject lambdas, closures, and globals other than
`math`, `torch`, and `torch.nn.functional as F`; the durable file boundary
makes every later subprocess reproducible.

## Seed-generation pipelines

All four pipelines target the same candidate API and use the same declaration,
oracle, and verification path.

| Pipeline                   | Inputs                                                | Generation method                                                                                      | LLM required |
| -------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------ |
| **A — AtenIR + LLM**       | Forward reference and extracted AtenIR backward graph | LLM plans, generates, and repairs fused Triton code                                                    | Yes          |
| **B — primitive dispatch** | Forward reference and extracted AtenIR backward graph | Handwritten Aten-to-Triton primitive lowering followed by dispatch-free code generation                | No           |
| **C — forward only**       | Forward source and declared contract                  | LLM derives the backward without seeing AtenIR; used as an ablation                                    | Yes          |
| **D — Inductor capture**   | Forward reference only                                | AOTAutograd traces the joint graph, the min-cut partitioner splits it, and Inductor lowers both halves | No           |

```bash
evograd seed a --op rmsnorm --output-dir /tmp/A_rmsnorm --model gpt-5.5
evograd seed b --op rmsnorm --output-dir /tmp/B_rmsnorm
evograd seed c --op rmsnorm --output-dir /tmp/C_rmsnorm --model gpt-5.5
evograd seed d --op rmsnorm --output-dir /tmp/D_rmsnorm
```

### Pipeline D and the saved-tensor contract

Pipelines A–C choose what the forward saves: B's seed keeps only the forward's
own inputs and recomputes everything, and A and C leave the choice to the LLM.
Pipeline D takes PyTorch's answer instead. `min_cut_rematerialization_partition`
solves for the cheapest set of tensors to carry from forward to backward, and
the seed is built around that save-set, so the two halves start from the same
contract `torch.compile` would have used.

That makes D both the strongest available seed and a reference point: it is the
production compiler's answer to the same question the search explores, so a run
that moves off the captured save-set is a measurable result rather than an
assumption. The chosen set is recorded in `partitioner_save_set.json`.

A Pipeline D run emits **one seed per dtype**. Inductor specializes on dtype, so
a single seed handles a single dtype; shapes are generic because the capture uses
dynamic shapes. Each seed holds exactly one forward and one backward kernel set:

```text
/tmp/D_rmsnorm/
├── float32/
│   ├── inductor_raw/{forward,backward}.py   # unmodified Inductor modules
│   ├── partitioner_save_set.json            # the min-cut's choice, and its byte cost
│   ├── initial_program_autograd_pair.py     # 1 forward + 1 backward kernel
│   └── verification_report.json
├── float16/
│   └── ...
└── seeds.json                               # index over the specialists
```

With `--dtype float16` the single seed is written directly into `--output-dir`
instead, so it drops into the existing tooling unchanged.

Evolve a specialist with `--dtype`, which gates correctness on that dtype alone
and measures on it by default:

```bash
evograd evolve --op rmsnorm \
    --seed /tmp/D_rmsnorm/float16/initial_program_autograd_pair.py \
    --dtype float16 --output-dir /tmp/E_rmsnorm_f16
```

Three properties of the capture are worth knowing:

- **Dynamic shapes by default**, so the kernels take sizes as runtime arguments
  and one seed covers the whole workload grid. Dimensions that are _equal_ at
  trace time get unified into one symbol, so the pipeline picks a workload whose
  dims are pairwise distinct, perturbing one if the declaration has none.
- **Scalar arguments are baked in** at trace time (`eps`, and similar). The
  wrapper still accepts them for API compatibility; changing one means
  re-capturing, or rewriting the kernels to take it as a runtime argument.
- **The seed refuses the wrong dtype** rather than silently mis-computing.

`--no-autotune` pins each kernel to the launch config autotuning chose during
capture, swapping `@triton_heuristics.pointwise`/`.reduction` for
`fixed_config`. The sweep at first call disappears, block sizes become explicit
constants, and timings stop depending on which workload happened to run first.
`triton_meta` and `inductor_meta` are carried over untouched -- the first is
what Triton needs to compile, the second carries `grid_type` and hence the
launch grid; neither is a tuning input. Pinning the _winner_ rather than a
default keeps the untuned seed as fast as the tuned one, which makes this the
right mode for isolating what autotuning contributes.

Inductor ships each Triton kernel as source text inside an
`async_compile.triton(...)` call. That text already carries its own imports and
`@triton_heuristics.*` / `@triton.jit` decorators, so the pipeline splices it in
at module level: the same name binds to the same autotuner, the `.run(...)`
launch sites keep working, and the kernel becomes readable code the search can
edit instead of a quoted blob. The result matches Pipeline B's shape — plain
`@triton.jit` kernels plus Python wrappers. Inductor's `call(args)` list
protocol is rewritten to named parameters at the same time, so `_fwd_call` and
`_bwd_call` read like hand-written launchers.

A CPU capture emits C++ instead of Triton, which cannot be inlined as Python and
stays in `async_compile` form — that path exists to check the plumbing without a
GPU, not to produce a seed anyone evolves.

Pipeline D needs no primitive coverage — anything Inductor can lower, it can
seed. Where Inductor falls back to a vendor library, the seed contains an
`extern_kernels` call rather than a kernel, and only the surrounding work is
open to the search. `matmul` is the extreme case: its seed contains **zero**
generated kernels, because both the forward and the backward are cuBLAS calls.
`linear` has one (the `dbias` reduction) and `evoattention` has six.

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
    forward="evograd.ops.level1.layernorm.forward_ref:layernorm_forward_ref",
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
LLVM. It differentiates compiled programs and uses _activity analysis_ to
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

## Action required: regenerate seeds produced before this fix

`program_codegen` emitted every scalar argument of a serialized `aten` fallback
**twice**. `_ordered_arg_exprs` already renders scalars, and
`_generic_fallback_call` re-interleaved them on top of that, so a node with `S`
scalars and `N` other arguments was emitted with `S + N + S` positional
arguments. The generated seed then raised on its first call:

```
RuntimeError: aten::_log_softmax() expected at most 3 argument(s) but received 5 argument(s)
RuntimeError: aten::sort() expected at most 3 argument(s) but received 5 argument(s)
```

Four operators' seeds failed their verification gate for this reason —
`cross_entropy`, `poly_norm`, `sparsemax` and `fused_linear_cross_entropy` — and
`sparsemax` was reported as 0% covered because of it. Any seed generated before
this fix whose graph contains an `aten` fallback with scalar arguments is
affected and should be regenerated.

Check an existing program without regenerating it:

```bash
python tools/check_seed_arity.py --program path/to/initial_program_autograd_pair.py
```

It compares every emitted `_resolve_aten(...)` call against that operator's
schema and exits non-zero on a mismatch. A clean seed prints
`0 arity mismatch(es)`.

Note that an evolved program derived from a broken seed may still be correct:
the search rewrites the block, and three of the four operators above produced
candidates that passed verification independently. The seed being invalid does
not by itself invalidate a downstream result — but it does mean the search
started from a program that could not run.

## Running the benchmark

The benchmark is specified in [docs/BENCHMARK.md](docs/BENCHMARK.md): 25
operators over three levels, 191 timed configurations, every shape traceable to
a layer of a real model. This section is the operational half — how to run it
and how to submit something to it.

### The reference line

Before measuring anything of your own, reproduce the reference line. Any
reviewed pair baseline can stand in as the candidate, which answers "what does a
hand-written production kernel achieve here" without needing a generated program
for all 25 operators:

```bash
evograd suite --candidate-baseline liger --out results/liger/
```

That times Liger-Kernel against eager PyTorch on every operator declaring a
`liger` adapter, and writes `suite_report.json` plus `SUITE_RESULTS.md`. The
report states what it measured in its first line — a suite result is unreadable
without knowing whether the thing being timed was a production library or a
generated kernel.

Operators with no such baseline are reported as **uncovered**, never skipped:
"we did not run it" and "it has no speedup" are different claims.

### Submitting a kernel

A submission is one Python module per operator implementing that operator's
autograd pair (see [Candidate API](#candidate-api) for the two function
signatures). Lay them out by operator name:

```text
my_kernels/
├── layernorm.py           # or my_kernels/layernorm/anything.py
├── rmsnorm.py
└── softmax.py
```

Then run whichever subset you are claiming:

```bash
# everything you have
evograd suite --candidates my_kernels/ --out results/mine/

# one level, or one operator
evograd suite --candidates my_kernels/ --level 1 --out results/mine/
evograd suite --candidates my_kernels/ --op rmsnorm --out results/mine/
```

Rules that follow from the specification rather than from this tool:

- **Correctness is a hard gate.** A candidate that fails any correctness case is
  not timed at all. Check yours first, on CPU if you like:
  `evograd verify --op rmsnorm --device cpu my_kernels/rmsnorm.py`
- **Partial submissions are legitimate**, and are reported as reduced coverage
  rather than as a smaller benchmark. Coverage is never folded into speedup.
- **You choose what the forward saves.** That choice is measured — saved-state
  bytes are reported alongside latency, because trading recomputation against
  retained memory is the thing this benchmark exists to expose.
- **Inputs must not be mutated.** The final-report protocol checks tensor
  content, shape, stride, dtype, and storage offset outside the timed regions.

### Single-operator measurement

For iterating on one kernel, the final-report protocol is available directly,
including its identity control:

```bash
<<<<<<< HEAD
evograd tier1-bench --op rmsnorm --candidate my_kernels/rmsnorm.py
evograd tier1-bench --op rmsnorm --identity-control    # must report ~1.0x
=======
evograd fair-bench --op rmsnorm --candidate my_kernels/rmsnorm.py
evograd fair-bench --op rmsnorm --identity-control    # must report ~1.0x

# static torch.compile specialist versus an evolved pair, across 2^13..2^27
# total elements at hidden=1024; compilation is outside the timed regions
evograd fair-bench --op layernorm --candidate my_kernels/layernorm.py \
    --baseline torch_compile --suite tb_sweep_13_27
>>>>>>> main
```

(`fair-bench` still works; it is the former name of the same command.)

## Evaluation tiers

Two independent axes decide how a kernel is measured, and conflating them is
how a benchmark ends up quoting a number that answers a different question than
the one asked.

**Tier — what is measured.**

| Tier | What runs | Module |
| ---- | --------- | ------ |
| 1 — pair | `y, saved = fwd(x, ...)` then `bwd(dy, saved)`, called directly | `bench/tier1.py` |
| 2 — operator | `y = model(x)` then `y.backward(dy)`, through the autograd engine | `bench/tier2.py` |
| 3 — model | a full training step, evolved kernels patched into a real model | not built |

At tier 1 *you* are the autograd engine: you hold the saved state, you route the
upstream gradient, you receive the gradients as return values. No training loop
does that. It writes `loss.backward()` and PyTorch records a graph, keeps the
saved state alive in its `ctx`, schedules the backward, and accumulates into
`.grad`. That work is real and a training step pays it every iteration, which is
why a tier-1 speedup is not a training speedup and must not be reported as one.
Measured on a GH200, an evolved LayerNorm that is 1.4x faster than eager at tier
1 is *slower* than eager at tier 2 until the rows reach five figures — the
crossover is the result, and only tier 2 can see it.

Tier 2 puts eager PyTorch, `torch.compile`, Liger and the evolved kernel behind
one `nn.Module` interface so all four pay the same framework cost. It needs one
thing tier 1 does not: `parameter_args` on the declaration, naming which
`Active` arguments are module state rather than activations. Both are `Active`
because both take gradients, and passing them positionally makes the question
moot at tier 1 — it only arises once an `nn.Module` has to hold some of them.

**Protocol — how carefully.** Orthogonal to tier. `fast` (`bench/harness.py`) is
what the evolutionary search calls thousands of times per run; it caches the
baseline across candidates and controls neither cache state nor provider order.
`fair` (`bench/tier1.py`, `bench/tier2.py`) remeasures both providers every
time, clears L2 before each timed region, randomizes provider order, rejects
input mutation, and reports bootstrap confidence intervals. **Every published
number comes from `fair`**, and every report records which protocol produced it.

### Running each tier

**Tier 1 — the kernel pair, called directly.**

```bash
# one operator against the strongest declared baseline
evograd tier1-bench --op layernorm --candidate best.py --out results/t1.json

# pick the comparison explicitly
evograd tier1-bench --op layernorm --candidate best.py --baseline pytorch_autograd
evograd tier1-bench --op matmul    --candidate best.py --baseline cublas_pair

# narrow the sweep while iterating
evograd tier1-bench --op layernorm --candidate best.py --dtype bfloat16 --suite tb_mixed

# calibrate the harness: same provider both sides, must report ~1.0x
evograd tier1-bench --op layernorm --identity-control

# confidence intervals: --blocks resamples, so >1 is needed for a non-zero width
evograd tier1-bench --op layernorm --candidate best.py --blocks 5
```

**Tier 2 — the operator through the autograd engine.** Compares four providers
at once: eager, `torch.compile`, the declared pair baseline, and the candidate.

```bash
# all four providers, every shape in the default suite, one process per shape
evograd tier2-bench --op layernorm --candidate best.py --out results/t2.json

# the reference line: no candidate, just what a production library achieves
evograd tier2-bench --op layernorm --out results/t2_reference.json

# drop torch.compile (it dominates wall-clock when iterating)
evograd tier2-bench --op layernorm --candidate best.py --no-compile

# a different declared baseline, and a dtype subset
evograd tier2-bench --op geglu --candidate best.py --baseline liger --dtype fp32

# faster loops while debugging; --no-isolate keeps everything in one process,
# so a hang takes the run down instead of one shape
evograd tier2-bench --op layernorm --candidate best.py --rep-ms 100 --no-isolate

# calibrate: eager against itself, must report ~1.0x
evograd tier2-bench --op layernorm --identity-control
```

An operator is measurable at tier 2 only if its declaration sets
`parameter_args`. All 25 built-ins do; a new or external declaration must say
which `Active` args are module state, or `()` if it has none.

**Tier 3 — a full training step.** Not built. See
[docs/BENCHMARK.md](docs/BENCHMARK.md) for the shape it will take.

**The cross-operator suite** runs tier 1 over every operator and pools the
result by family, then by level:

```bash
evograd suite --candidate-baseline liger --out results/liger/   # the reference line
evograd suite --candidate my_kernels/   --out results/mine/     # a submission
evograd suite --candidate my_kernels/   --out results/quick/ --protocol fast
```

`--protocol fast` selects the evolution harness for iteration only; the report
says so in its header, and published numbers must not come from it.

**Running both tiers on the same operator** is the experiment that justifies
having tiers at all — the gap between them is the framework cost a training
loop pays, and the shape where the two curves cross is the deployment answer:

```bash
evograd tier1-bench --op layernorm --candidate best.py --out results/t1.json
evograd tier2-bench --op layernorm --candidate best.py --out results/t2.json
```

One naming collision to keep straight: `ops/level1/`, `ops/level2/`,
`ops/level3/` and `OpDecl.level` are the **task** hierarchy — primitive, fused,
architectural block — and have nothing to do with evaluation tiers. A level-1
primitive measured at tier 2 is an ordinary thing to want.

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
tensor bytes. After correctness it also makes one untimed call at every selected
benchmark shape, so a shape-dependent deadlock is rejected before timing.

Every evaluation runs in a killable child process. A candidate that hangs or
corrupts its CUDA context is terminated without poisoning the OpenEvolve
worker. Baseline timings are cached persistently by operator, baseline, GPU,
timing settings, dtype, and dimensions; the candidate is always re-timed.

Available scoring policies are:

| Policy                              | Objective                                                          |
| ----------------------------------- | ------------------------------------------------------------------ |
| `speed`                             | Backward speedup                                                   |
| `speed_memory`                      | Weighted backward/full-step speedup with a saved-memory penalty    |
| `speed_memory_min`                  | Minimum of backward and full-step speedup, memory-penalized        |
| `speed_memory_min_geomean`          | Geometric-mean minimum speedup, memory-penalized                   |
| `speed_memory_min_weighted_geomean` | Weighted geometric mean with a worst-case guard and memory penalty |

The default is `speed_memory`. Seventeen declarations expose reviewed optional
Liger baselines. `--baseline auto` selects Liger when its adapter and
`liger-kernel` are available, otherwise PyTorch autograd; an explicit
`--baseline liger` hard-fails rather than silently changing the comparison:

```bash
pip install -e ".[baselines]"
evograd bench --op layernorm --candidate best.py --baseline liger
```

Two built-in `torch.compile` baselines are available for every operator without
any declaration support — `torch_compile` and `torch_compile_max_autotune`:

```bash
evograd bench --op layernorm --candidate best.py --baseline torch_compile
```

They matter for Pipeline D. `pytorch_autograd` measures *eager* PyTorch, while a
D seed is captured from AOTAutograd + Inductor — roughly what `torch.compile`
itself would emit — so beating eager is expected and says little about the
kernel. Both compiled baselines are checked against the eager oracle before
their timings are used, and they are never selected by `--baseline auto`: each
benchmark case compiles a shape specialist, which costs real wall time on the
first run (it is absorbed by warmup and by the persistent baseline timing
cache).

### NCU-guided refinement

The final fork’s Nsight Compute agent is now owned by evograd instead of
patching OpenEvolve internals:

```bash
evograd evolve \
    --op softmax \
    --seed seed.py \
    --output-dir /tmp/evolve_softmax \
    --ncu
```

The pass warms the candidate outside NCU, profiles one representative declared
workload, performs deterministic roofline triage, asks the LLM for a
metric-grounded diagnosis and rewrite, and runs the same correctness/performance
evaluator again. It replaces the best program only when the rewrite is correct
and has a strictly higher combined score. The original/proposed programs,
`.ncu-rep`, metrics, diagnosis, and outcome are stored under `ncu_passes/`.

`evograd ncu` applies the same accepted-only pass to an existing candidate.
OpenEvolve 0.3.2 has no public per-candidate hook, so evograd refines each
evolution group’s returned best at the supported library boundary rather than
retaining the fork’s private every-N worker patch.

## Supported operators

Operators are organized into three benchmark levels. See
[the benchmark specification](docs/BENCHMARK.md) for the metrics, the shape
provenance rules, and how the levels are aggregated.

### Level 1 — primitive operators

| Operator        | Family     | Forward                                   | Requested gradients               | Liger baseline |
| --------------- | ---------- | ----------------------------------------- | --------------------------------- | -------------- |
| `layernorm`     | norm       | LayerNorm                                 | `dx`, `dweight`, `dbias`          | yes            |
| `rmsnorm`       | norm       | RMSNorm                                   | `dx`, `dweight`                   | yes            |
| `poly_norm`     | norm       | polynomial normalization                  | `dx`, `dweight`, `dbias`          | yes            |
| `dyt`           | norm       | dynamic tanh                              | `dx`, `dalpha`, `dgamma`, `dbeta` | yes            |
| `softmax`       | reduction  | row-wise softmax                          | `dx`                              | yes            |
| `sparsemax`     | reduction  | simplex projection                        | `dx`                              | yes            |
| `swiglu`        | activation | SwiGLU activation                         | `da`, `db`                        | yes            |
| `geglu`         | activation | GeGLU activation                          | `da`, `db`                        | yes            |
| `relu_squared`  | activation | `relu(x)²`                                | `dx`                              | yes            |
| `cross_entropy` | loss       | mean cross-entropy loss                   | `dlogits`                         | yes            |
| `kl_div`        | loss       | KL divergence                             | `d_input`                         | yes            |
| `jsd`           | loss       | Jensen-Shannon divergence                 | `dlog_q`                          | yes            |
| `tvd`           | loss       | total-variation distance                  | `dp`, `dq`                        | yes            |
| `matmul`        | gemm       | `a @ b`                                   | `da`, `db`                        | no (cuBLAS)    |
| `linear`        | gemm       | `x @ weight.T + bias`                     | `dx`, `dweight`, `dbias`          | no             |
| `conv2d`        | conv       | dense NCHW convolution                    | `dx`, `dweight`, `dbias`          | no (cuDNN)     |
| `evoattention`  | attention  | AlphaFold3-style attention with pair bias | `dq`, `dk`, `dv`, `d_pair_bias`   | no             |
| `rope`          | positional | rotary position embedding                 | `dx`                              | yes            |

### Level 2 — fused operators

| Operator                     | Family | Forward                                        | Requested gradients                                   | Liger baseline           |
| ---------------------------- | ------ | ---------------------------------------------- | ----------------------------------------------------- | ------------------------ |
| `fused_add_rms_norm`         | norm   | residual add plus RMSNorm                      | `dx`, `dr`, `dweight`                                 | yes                      |
| `fused_linear_cross_entropy` | loss   | lm_head projection fused with the loss         | `dx`, `dweight`                                       | yes                      |
| `layernorm_linear`           | gemm   | LayerNorm followed by Linear                   | `dx`, `dlinear_weight`, `dweight`, `dbias`            | no                       |
| `gemm_leaky_relu`            | gemm   | GEMM with fused Leaky-ReLU epilogue            | `da`, `db`                                            | no (Triton tutorial)     |
| `fused_moe_swiglu`           | moe    | routed grouped GEMM + SwiGLU + down projection | `dx`, `dgate_up_proj`, `ddown_proj`, `dtop_k_weights` | yes                      |

### Level 3 — architectural blocks

| Operator                | Family        | Forward                                                | Gradients |
| ----------------------- | ------------- | ------------------------------------------------------ | --------: |
| `llama3_decoder_layer`  | llm_block     | one Llama-3-8B decoder layer (training forward pass)    | 10        |
| `af3_single_repr_block` | protein_block | AlphaFold3 single-representation update with pair bias  | 13        |

Both blocks compute their correctness reference in float32 while the candidate
runs in bfloat16 (`reference_dtype`). Composing ten operators makes a
same-dtype reference carry as much rounding error as the candidate, at which
point the tolerance stops bounding the candidate's own error. Their declarations
also state what they exclude — the Llama block has no KV cache, and the
AlphaFold3 block is the single-representation update rather than a full
pairformer block, which would need two outputs.

Timed grids are derived from frozen model configurations in
`src/evograd/opdecl/models.py` rather than written by hand, and each workload
records the model and component it came from. `tests/test_provenance.py`
re-derives them, so a declared shape cannot drift from the model it claims to
measure. The pre-v1 hand-picked grids survive as `legacy` ablation suites.

Liger
does not ship generic dense GEMM or convolution kernels: `matmul` therefore
compares against PyTorch/cuBLAS, while `conv2d` compares against PyTorch/cuDNN.
`gemm_leaky_relu` additionally exposes the reviewed `triton_tutorial` pair
baseline; its forward uses the tutorial's grouped GEMM ordering and fused
epilogue, while its backward uses the same GEMM kernel plus a pointwise
activation derivative. `fused_moe_swiglu` is the separate, genuine Liger
grouped-GEMM+activation benchmark; it must not be interpreted as a generic
dense GEMM baseline.

Pipeline B provides handwritten Triton lowerings for the fixed `conv2d`
contract (NCHW/OIHW, stride 1, padding 0, dilation 1, groups 1), including
forward, dX, dWeight, and dBias. Its MoE seed uses a coverage-safe dense-expert
BMM decomposition plus Triton routing/scatter primitives; the Liger baseline
retains sparse grouped-GEMM execution, leaving top-k/grouped fusion as an
explicit evolution target.

LayerNorm also includes the legacy and TritonBench shape suites:

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

To bootstrap a draft from a forward:

```bash
evograd scaffold \
    --op my_op \
    --forward /path/to/forward.py:my_forward \
    --output-dir /tmp/my_op_contract
```

This is the declaration-native replacement for the fork’s generated
`task_spec.py`: the LLM supplies input semantics, activity, shapes, and regimes;
code builds the declaration and PyTorch-autograd oracle mechanically. Review
the generated contract before publishing it as a built-in package.

For operators with semantic masks, scaled weights, integer-like inputs, or
special initialization, define a declaration-local `make_inputs` hook. If
Pipeline B encounters an unsupported Aten operation, add its lowering under
`src/evograd/atenir/primitive_triton/`.

The current declaration model targets deterministic, single-tensor-output
operators. Integer/bool tensor inputs are supported as `Inactive` values
(cross-entropy labels are the worked example). Multiple outputs, sparse
gradients, and stateful operators require declaration/API extensions.

## Repository layout

```text
src/evograd/
├── opdecl/                    # declaration types, oracle, binding, verification
├── ops/                       # one self-contained package per operator,
│   ├── level1/                #   grouped by benchmark level: 18 primitive,
│   ├── level2/                #   5 fused, 2 architectural blocks. The grouping
│   └── level3/                #   follows OpDecl.level, which stays the authority
├── atenir/
│   ├── extract.py             # PyTorch/autograd graph extraction
│   ├── compose.py             # serialized graph execution
│   └── primitive_triton/      # Pipeline B's handwritten Triton primitives
├── pipelines/
│   ├── a_atenir_llm/          # AtenIR-grounded LLM synthesis
│   ├── b_dispatch/            # LLM-free primitive lowering and code generation
│   ├── c_forward_only/        # forward-only LLM ablation
│   ├── d_inductor/           # LLM-free capture of Inductor's own kernels
│   └── shared/
├── evolve/                    # OpenEvolve evaluator, scoring, and run wrapper
├── bench/                     # generic latency and memory harness
├── ncu/                       # NCU profiling, roofline triage, accepted refinement
├── scaffold.py                # forward -> external declare_op contract
├── dispatch.py                # measured generalist/specialist deployment
├── api.py                     # one-call Python workflow
└── cli.py                     # evograd command-line interface
```

OpenEvolve is a package dependency rather than vendored or forked source.
Evograd supplies the initial program, evaluator, generated configuration, and
operator-specific environment; OpenEvolve supplies the evolutionary search,
program database, LLM orchestration, and checkpointing.

See [the final migration audit](docs/MIGRATION_AUDIT.md) for the commit-by-commit
comparison with fork `origin/main` at `e7e1e3a`.

## Project status

Evograd is the declaration-driven successor to the AtenIR, pipeline, and Triton
backward-benchmark work previously developed in an OpenEvolve fork.

Locally validated:

- 63 unit tests pass.
- Eight torch-facing unit tests are present but skip on machines without
  PyTorch.
- All 18 declarations match the final-fork correctness cases, benchmark cases,
  per-output tolerances, input distributions, and seed recipes.
- AtenIR preserves the migrated implementation and includes final-main support
  for inactive integer/bool inputs such as class labels.
- The rendered configuration and `run_evolution` call shape load against
  upstream OpenEvolve 0.3.2.
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
