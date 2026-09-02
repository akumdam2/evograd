# Evaluation

How a kernel is measured, in one place: the tiers, the protocols, the
correctness gates, and how to run each of them.

The benchmark *specification* — which operators, which shapes, how results are
aggregated — is [docs/BENCHMARK.md](../../../docs/BENCHMARK.md). This document is
the operational half.

## Layout

```
bench/
  provider.py      the pair boundary + input-mutation guards      shared
  report.py        canonical schema + per-protocol readers        shared
  suite.py         one adapter -> TaskResult -> pooled report     shared

  harness.py       tier 1, `fast`   -- the evolutionary search's benchmark
  tier1.py         tier 1, `fair`   -- publishable numbers
  tier2.py         tier 2           -- the operator through the autograd engine
  tier3_model.py   tier 3, part 1   -- what is measured
  tier3_patch.py   tier 3, part 2   -- how a kernel gets in
  tier3_runner.py  tier 3, part 3   -- how it is measured
  tier3_llama.py   tier 3           -- the built-in architecture
  tier3.py         tier 3           -- facade
```

## Running the benchmark

25 operators over three levels, 191 timed configurations, every shape traceable
to a layer of a real model.

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

**Tier 3 — a full training step.** Evolved kernels patched into a model, measured
on throughput, peak memory, loss agreement, and how much of the step the GPU was
*not* the bottleneck for.

```bash
# the reference line: eager vs Liger, plus the harness control
evograd tier3-bench --model llama_3_8b_4l --identity-control --baseline liger \
    --out results/t3.json

# your kernel at one site, or several
evograd tier3-bench --candidate rms_norm=my_rmsnorm.py --baseline liger
evograd tier3-bench --candidate rms_norm=a.py --candidate swiglu=b.py

# attribute a blended result to one site
evograd tier3-bench --sites cross_entropy --baseline liger

# the memory experiment: find where eager dies and a fused loss does not
for b in 2 4 8 12 16; do
  evograd tier3-bench --batch $b --tokens 2048 --baseline liger \
      --steps 5 --blocks 1 --out results/t3_b$b.json
done
```

Defaults: `--model llama_3_8b_4l --batch 4 --tokens 1024 --steps 10 --blocks 3
--warmup 3 --loss-steps 5 --dtype bfloat16 --baseline liger`.

`llama_3_8b_4l` is Llama-3-8B with four layers instead of thirty-two. Layer
count is the one dimension that can be cut without changing what a kernel does
to the answer — per-layer effects scale linearly in it, and the loss head's
`[tokens, vocab]` memory does not depend on it at all. Iterate there; report
from `llama_3_8b`. Large runs want
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the eager loss materializes
a `[tokens, vocab]` logits tensor, and fragmentation rather than capacity is
usually what ends a run.

#### Three parts

| Module | Owns | Depends on |
| ------ | ---- | ---------- |
| `tier3_model.py` | **what** is measured — the `TrainingWorkload` protocol, `ModuleWorkload` for bringing your own `nn.Module` | patch |
| `tier3_patch.py` | **how a kernel gets in** — `KernelSet`, the patch sites, `bind` wrapping, module surgery, the identity control | nothing in tier 3 |
| `tier3_runner.py` | **how it is measured** — build, step, time, memory, loss agreement, report | patch, model |

The direction is the design, and `tests/test_tier3.py` enforces it: the patcher
knows nothing about models or measurement, so a new architecture needs neither
the runner nor the facade.

#### Supplying a kernel

Same autograd pair tiers 1 and 2 take, so anything evolution produced already
works. `lookup_pair` accepts these names or the op-prefixed ones.

| `--candidate SITE=` | operator | forward | backward returns |
| ------------------- | -------- | ------- | ---------------- |
| `rms_norm` | `rmsnorm` | `forward_with_saved(x, weight, eps) -> (y, saved)` | `(dx, dweight)` |
| `swiglu` | `swiglu` | `forward_with_saved(a, b) -> (c, saved)` | `(da, db)` |
| `cross_entropy` | `fused_linear_cross_entropy` | `forward_with_saved(x, weight, target) -> (loss, saved)` | `(dx, dweight)` |

`bind` splits `saved` into tensors and plain values so `save_for_backward` is
satisfied whatever you chose to keep, and places each returned gradient in its
declared slot. Two things you must get right: **gradient order** matches
`op.grad_names()`, and tensors are **2D rows** — `[rows, hidden]`, not
`[batch, tokens, hidden]`. The model works in three dimensions and every
declaration is written for rows; `kernel_from_pair` flattens and restores around
your kernel, so write it exactly as the declaration specifies.

#### Supplying a model

The runner never asks what model it is driving. It asks a `TrainingWorkload`
four questions — `units_per_step`, `build(kernels)`, `batch_for(seed)`,
`loss(model, batch)` — and measures whatever comes back. A kernel reaches the
model one of two ways: **by construction**, where the model holds a `KernelSet`
and calls through it so building with a different set *is* the patch (what
`LlamaWorkload` does), or **by module surgery** for a model you did not write:

```python
ModuleWorkload(
    name="llama-3-8b-hf",
    factory=lambda: LlamaForCausalLM(config).cuda(),
    make_batch=lambda seed: tokens_and_labels(seed),
    compute_loss=lambda model, b: model(**b).loss,
    units=batch * seq_len,
    patches=(ModulePatch("rms_norm", is_llama_rms_norm, replace_norm),),
)
```

`patch_modules` walks the built tree and returns the paths it touched. An eager
`KernelSet` touches nothing, so the unpatched provider is the original model
rather than a rebuilt lookalike, and the replacement receives the **original
module** rather than its shapes, so trained weights carry across.

#### What comes out

Per provider, 40 steps: 5 loss-trajectory on fresh batches, 2 peak-memory
probes, 3 warmup, then 3 blocks of 10 timed. The timed region is the whole
step — `loss.backward()`, `optimizer.step()`, `zero_grad()` included — because
training pays all of it.

| Metric | Answers |
| ------ | ------- |
| `step_ms`, `units_per_second` | is training faster |
| `peak_memory_bytes` | can you fit a bigger batch — the lever a fused loss actually pulls |
| `cpu_bound_fraction` | **could a faster kernel even show up** |
| `losses` → `loss_agreement` | does it still train the same |

Read `cpu_bound_fraction` as "the GPU was not the bottleneck", not "this is all
dispatch cost" — the CPU also stops running when it blocks on an implicit
synchronization, and a step allocating a multi-gigabyte logits tensor every
iteration will do exactly that. Near 1.0 on the **unpatched** provider means the
configuration cannot measure kernels at all, whatever the other numbers say.

`--identity-control` adds a provider that patches every site with the eager
kernel it already had: same mathematics, all of the patching machinery. The gap
to plain eager is the harness tax, and nothing smaller than it is a kernel
result. It reads as an upper bound — its backward recomputes the forward where a
real candidate's is a kernel.

#### Known gaps

**Tier 3 runs no correctness gate.** Tier 2 checks every provider against the
autograd oracle before timing it (`check_module`); tier 3 does not. A wrong
kernel here produces numbers rather than an error, and shows up only as
divergence in `loss_agreement` — which is reported, never enforced, because a
defensible threshold depends on dtype and horizon. Until this is closed, verify
before you measure:

```bash
evograd verify --op rmsnorm my_rmsnorm.py
evograd tier1-bench --op rmsnorm --candidate my_rmsnorm.py
evograd tier2-bench --op rmsnorm --candidate my_rmsnorm.py    # gated
evograd tier3-bench --candidate rms_norm=my_rmsnorm.py        # not gated
```

Closing it means reusing `opdecl/verify.py` the way `tier2.check_module` does,
on the shapes the model actually presents.

Three smaller ones. **Providers are measured sequentially in a fixed order**,
where tier 1 randomizes provider order in paired blocks to control for clock
drift; it touches only `tier3_runner.py`. **Results do not reach the suite** —
`report.py` has no tier-3 reader, and the metrics differ enough (throughput and
peak memory rather than a speedup ratio) that `CaseMetrics` needs extending
first. **The harness is unvalidated against a published number**: running Liger
as the candidate and reproducing its published end-to-end figures is the check
that gave confidence in tier 2, and it has not been done here.

**The cross-operator suite** runs tier 1 over every operator and pools the
result by family, then by level:

```bash
evograd suite --candidate-baseline liger --out results/liger/   # the reference line
evograd suite --candidates my_kernels/   --out results/mine/     # a submission
evograd suite --candidates my_kernels/   --out results/quick/ --protocol fast
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

Seventeen declarations expose reviewed optional
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
