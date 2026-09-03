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

  workloads/qwen3/                  -- the Qwen3-0.6B workload, organized by level
    harvest/                        --   the instrumented run and its snapshot
    levels/level4 3 2 1             --   step, captured layer, operators, primitives
    evaluation/tier3/               --   drop-in replacement, gate, calibration
                                    -- see workloads/qwen3/README.md
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

### What the reports contain

`evograd suite` writes `suite_report.json` and `SUITE_RESULTS.md` with
per-operator, per-level, and overall full-step speedup, coverage, and saved
memory. Restrict the scope with `--level` or `--op`:

```bash
evograd suite --candidates programs/ --out results/
evograd suite --level 1 --level 2 --out results/
evograd suite --candidates programs/ --op rmsnorm --out results/
```

`evograd bench` exits non-zero when anything fails and prints the failure to
stderr. With `--out` the report is written either way, because a run that died
partway is still evidence: `report["ok"]` is the verdict, `report["error"]`
holds a setup failure (unknown op, a candidate that raises on import, a
`--dtype` the declared benchmark suite has no cases for), and
`report["cases"][i]["error"]` holds a per-workload failure. Cases that did run
are still aggregated — an operator that works on four shapes out of five is not
an operator with four shapes' worth of speedup, it is one with 80% coverage, and
the report says so rather than averaging the gap away.

### Submitting a kernel

A submission is one Python module per operator implementing that operator's
autograd pair (see [Candidate API](../../../README.md#candidate-api) for the two function
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
evograd tier1-bench --op rmsnorm --candidate my_kernels/rmsnorm.py
evograd tier1-bench --op rmsnorm --identity-control    # must report ~1.0x

# static torch.compile specialist versus an evolved pair, across 2^13..2^27
# total elements at hidden=1024; compilation is outside the timed regions
evograd tier1-bench --op layernorm --candidate my_kernels/layernorm.py \
    --baseline torch_compile --suite tb_sweep_13_27
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
| 3 — model | a full training step, evolved kernels patched into a real model | `bench/tier3*.py` |

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
`fair` (`bench/tier1.py`, `bench/tier2.py`, `bench/tier3_runner.py`) remeasures
every provider, randomizes provider order, rejects input mutation, and reports
bootstrap confidence intervals. **Every published number comes from `fair`**,
and every report records which protocol produced it.

One `fair` rule is tier-specific rather than universal: tiers 1 and 2 clear L2
before each timed region, because they measure one kernel on inputs it would
otherwise find resident. Tier 3 never does — a training step's weights,
activations and optimizer state are exactly as warm as the previous step left
them, and that is the thing being measured.

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
`parameter_args`. All 30 built-ins do; a new or external declaration must say
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

# one process for everything, so a debugger can reach the kernel
evograd tier3-bench --candidate rms_norm=best.py --no-isolate

# raise the per-provider budget for a large model, or skip the gate to time a
# kernel you already know is wrong
evograd tier3-bench --model llama_3_8b --timeout 3600
evograd tier3-bench --candidate rms_norm=wip.py --no-verify
```

Defaults: `--model llama_3_8b_4l --batch 4 --tokens 1024 --steps 10 --blocks 3
--warmup 3 --loss-steps 5 --learning-rate 1e-4 --dtype bfloat16 --baseline liger
--timeout 1800` (`EVOGRAD_TIER3_TIMEOUT` overrides the last).

Every provider runs in its own killable child process, in a **seeded random
order** the report records: a GPU that warms or throttles across a run gives
whichever provider went first a systematic advantage, and with a fixed order
that advantage is indistinguishable from the kernel. A provider that hangs, dies,
or is OOM-killed costs its own row and nothing else — `--no-isolate` gives that
up for a debugger.

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

#### Two workloads, two registries

`llama_3` has three sites and supplies no preflight shapes. `qwen3_0_6b` has
four and supplies its calibrated observed configuration for each, so a candidate
is gated on the shape the model presents before tier 3 will time it:

```bash
# the canonical Qwen3-0.6B training step, batch 2 x sequence 2048, BF16
evograd tier3-bench --model qwen3_0_6b --structural-identity
evograd tier3-bench --model qwen3_0_6b --identity-control      # the bound pair
evograd tier3-bench --model qwen3_0_6b --sites swiglu_mlp --layers 2   # a smoke
```

| site | operator | modules | invocations |
| ---- | -------- | ------- | ----------- |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | `Qwen3Attention` | 28 |
| `attention` | `qwen3_attention` | the same module | 28 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | `Qwen3MLP` | 28 |
| `residual_rmsnorm` | `fused_add_rms_norm` | decoder layers + `model.norm` | 56 |

Both attention sites share **one** adapter on one module, selected
independently; an unselected site runs the production spelling. The residual
site is not a module at all — its add ends decoder layer *i* and its norm starts
layer *i+1* — so the layer hands the un-added pair to the next fusion site and
both returned outputs are used, `summed` as the residual stream and `normalized`
into the next sublayer. Layer 0's `input_layernorm` has no preceding decoder add
and stays unfused, which is why the count is 56 and not 57.

No module is replaced: the adapters rebind `forward` on the existing instances,
so the parameters are the same objects and `state_dict` is untouched by
construction. `--structural-identity` proves it — same submodules, same order,
native autograd, and **bitwise identical logits and loss** on the canonical
model. See [docs/QWEN3_LEVEL4.md](../../../docs/QWEN3_LEVEL4.md) for what the
backward comparison found.

#### Patch sites belong to a workload

There was one module-level `SITE_OPS` dict mapping site names to declared
operators, and it was correct exactly as long as there was one model. With two
it is wrong in both directions: Llama's identity control would patch a site
belonging to another architecture, and a candidate would be accepted for
`rms_norm` because that name happens to exist somewhere.

A `TrainingWorkload` therefore declares a `site_registry`, and tier 3 refuses to
guess when it does not:

```python
LLAMA_SITES = SiteRegistry(
    name="llama_3",
    sites=(
        Site("rms_norm",      "rmsnorm",                     _rms_norm_fused),
        Site("swiglu",        "swiglu",                      _default_swiglu),
        Site("cross_entropy", "fused_linear_cross_entropy",  _eager_cross_entropy),
    ),
)
```

Every tier-3 operation reads it from the kernel set it is handed — candidate
loading, baseline discovery, preflight, the identity control, `restrict`,
provenance, CLI validation, and the report, which serializes the exact mapping
under `site_registry`. An unknown site names the active workload and its valid
sites, so "wrong site" and "wrong model" are distinguishable. `LLAMA_SITE_OPS`
is the former global under a name that says whose it is.

A `Site` can also carry **preflight workloads**: model-derived shapes the
operator's own declared grid does not contain. This is what decides whether an
operator's *observed* configuration can block timing. `rmsnorm`'s declared grid
tops out at 128 rows; a model presents 4096, and a kernel can be right at one
and wrong at the other — `tests/test_tier3_registry.py` proves exactly that,
with a pair that passes the small grid and is caught by a supplied shape. Llama
supplies none, so its behaviour is unchanged.

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

`cpu_bound_fraction` is the **CPU submission-and-blocking fraction** and nothing
narrower. The CPU stops running both when it has finished submitting and when it
blocks — on an implicit synchronization, on the allocator while a step reserves a
multi-gigabyte logits tensor, on a stray `.item()`. All of those are real reasons
a kernel improvement cannot surface, none is separable from the others without a
profile, and reporting the number as dispatch cost claims a decomposition nobody
measured. Near 1.0 on the **unpatched** provider means the configuration cannot
measure kernels at all, whatever the other numbers say.

Per provider the report also carries `per_block_ms`, a `latency` summary
(median/min/q20/q80), the `patch_provenance` of that provider's own build, the
`kernel_sources` behind each patched site, and the `verification` record for it.
`speedup_intervals` gives each provider's step-time ratio against eager with a
block-bootstrap 95% interval. Blocks are resampled within each provider
independently — unlike tier 1, the providers here are not measured in
interleaved paired blocks, since each builds and trains its own model, so there
is no pairing to preserve. With the default three blocks the interval is wide;
`--blocks` is how it narrows.

#### The identity control is a ceiling, not a tax

`--identity-control` adds a provider that patches every site with the eager
kernel it already had: same mathematics, all of the patching machinery —
`bind`, a Python `autograd.Function`, the rank adapter.

Its backward **recomputes the forward** and differentiates it, where a real
candidate's backward is a kernel. So the gap between it and plain eager is an
**upper bound on what the patch plumbing costs**, not a measurement of it: the
bound includes one extra forward per patched site that no candidate pays. Read
it as a ceiling. If it lands near unpatched eager the plumbing is free and a
slowdown belongs to the kernels; if a patched provider is slower than this bound
the kernels are the story regardless; in between it is inconclusive, and
`--sites` narrows it.

#### Correctness before timing

A wrong kernel at this tier does not raise. It returns a throughput — and a
kernel that overflows to NaN reports *faster* than eager, because NaN arithmetic
is not slower. So two gates run before anything is measured, and a provider that
fails either is recorded as failed and never timed:

1. **The tier-1 pair gate.** Every kernel a provider patches in goes through
   `bench.provider.verify_pair_provider` — the same path tier 1's CLI and tier
   2's `check_module` gate on — against its declaration's own correctness
   workloads **plus whatever model-derived shapes the workload's registry adds
   for that site**. The report lists every configuration actually checked, with
   its dims and whether it was declared or workload-supplied. Tier 3 cannot verify the model-shaped call directly: at these
   sites the model's activations are not a declared workload, and the only
   oracle that exists is the declaration's. The kernel is verified where it *is*
   specified, and the rank adapter carries that verdict to the model's shapes. A
   site holding a raw callable that no declaration governs is reported as
   `unverifiable` rather than passed.
2. **Finite scalar losses.** Every loss, in the trajectory and in the timed
   batch, must be a scalar and finite. NaN or Inf marks the provider failed.

`loss_agreement` sits on top of those, not instead of them. It is reported and
**not** gated by default: how much divergence is acceptable depends on dtype and
horizon, and a threshold invented in the harness would be arbitrary in exactly
the way that makes a gate worse than none. A workload that knows its own answer
declares `loss_delta_threshold` and the trajectory becomes a gate for it alone.
Whatever was gated, the report says so in `verification_policy`.

#### Known gaps

**Results do not reach the suite.** `report.py` has no tier-3 reader, and the
metrics differ enough — throughput and peak memory rather than a speedup ratio —
that `CaseMetrics` needs extending first.

**The harness is unvalidated against a published number.** Running Liger as the
candidate and reproducing its published end-to-end Llama figures is the check
that gave confidence in tier 2, and it has not been done here.

**Verification is at the declared shape, not the model's.** The preflight gate
proves the kernel correct on its declaration's correctness workloads; what
carries that to `[batch, tokens, hidden]` is the rank adapter, which is tested
but is not itself an oracle. A kernel that is correct at `[rows, hidden]` and
wrong only at some leading-dimension arrangement would pass.

**`ModuleWorkload` cannot be isolated.** Its factory and loss are closures, so
it is not picklable into a child process. Driving one means `run_tier3`
in-process, where a kernel that hangs or wedges the CUDA context takes the whole
run with it. The CLI's per-provider isolation covers the workloads that can be
named on a command line.

**A Qwen provider is gated on the whole model, not only on its sites.** Before
tier 3 times a non-eager Qwen provider it runs one untimed canonical
forward/backward/AdamW step and checks it against a *calibrated* per-role noise
envelope — derived from the unmodified model compared with itself, validated on
holdout seeds, and bound to the GPU, driver, CUDA, torch, transformers, SDPA
backend and TF32 settings it was measured under. A failure is recorded
`failed_at="model_correctness"` and never timed. There is no global tolerance
constant; see [docs/QWEN3_LEVEL4.md](../../../docs/QWEN3_LEVEL4.md) for the
formula and the three reference comparisons it separates.

**Parameter gradients are not bitwise reproducible on this GPU.** The canonical
Qwen3 model compared against *itself* — same seed, same batch, no patching —
agrees bitwise on logits and loss and differs on a large and run-varying share
of its parameter gradients. Any gradient-level identity claim is therefore
relative to that floor, and a loss-trajectory comparison across providers
inherits it.

**Tier 3 has no reader in the suite and no published number.** The Qwen3
workload exists and is validated; nothing has been timed with intent.

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

## Level 4: a whole-model training workload

Levels 1-3 measure declared operators. Level 4 runs one real model the way
training runs it, so that a later stage can observe a training step instead of
guessing what one contains. So far this means a reference execution and a
harvest of what it invokes -- no provider comparison and no timing.

```bash
pip install 'evograd[qwen3]'        # optional: transformers>=4.51

PYTHONPATH=src python -m evograd.bench.workloads.qwen3 \
    --out results/qwen3-level4/canonical.json
```

The canonical workload is Qwen3-0.6B, batch 2, sequence 2048, BF16, CUDA, SDPA,
`model.train()`, `use_cache=False`, gradient checkpointing off, seed 0,
randomly initialised from config with no weight or tokenizer download:

```python
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

The JSON report records what was requested *and* what the built model actually
reports, plus loss finiteness and per-parameter gradient coverage. Its memory
and wall-time fields are diagnostic only and are not benchmark results.

The same execution can be *harvested*: an observer on a fixed list of stable
Qwen3 boundaries (decoder layer, attention, MLP, `nn.Linear`, RMSNorm, SiLU,
RoPE application, SDPA, causal cross entropy) exports a manifest with the raw
invocation transcript and a structurally deduplicated configuration set carrying
frequency and provenance.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.harvest \
    --out results/qwen3-level4/harvest.json
```

Forward-side semantic invocations only: the loss and backward pass execute and
gradient coverage is validated, but no backward operation and no CUDA kernel is
traced.

One observed decoder layer can then be lifted out into a standalone Level-3
reference. The capture records the arguments Transformers really passed, the
layer's output, the upstream gradient the full-model backward delivered to it,
and the weights and their gradients; a separate process rebuilds a single
`Qwen3DecoderLayer` and reproduces all of it.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.capture \
    --layer 14 --harvest results/qwen3-level4/harvest.json \
    --out results/qwen3-level4/layer14.pt

PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.replay \
    --artifact results/qwen3-level4/layer14.pt \
    --report results/qwen3-level4/layer14-replay.json
```

One `Qwen3MLP` invocation from that verified replay becomes
[`qwen3_swiglu_mlp`](../ops/level2/qwen3_swiglu_mlp/) -- the first
operator here whose benchmark shape was observed rather than chosen. Its
provenance is checkable twice: `Provenance(model="qwen3_0_6b", ...)` re-derives
the dims from the published configuration, and a small tracked snapshot
(`workloads/qwen3/harvest/snapshot.json`) carries the harvested
configuration id, the frequency of 28, and all 28 source module paths as
structured data. Declarations never read `results/`.

Two more Level-2 tasks come from the same artifact:
[`qwen3_attention`](../ops/level2/qwen3_attention/) (causal
grouped-query SDPA plus the output projection) and
[`qwen3_qkv_norm_rope`](../ops/level2/qwen3_qkv_norm_rope/) (the
projections, per-head Q/K RMSNorm and RoPE that feed it). The second returns
three tensors, which is why `OpDecl.output` now accepts an ordered tuple of
`Active` outputs; single-output declarations are unchanged.

The fourth boundary, the residual add plus RMSNorm, corrected the *existing*
generic [`fused_add_rms_norm`](../ops/level2/fused_add_rms_norm/)
rather than adding a Qwen-specific duplicate: a decoder's residual stream keeps
the un-normalized sum, so the task now returns `(normalized, summed)` and its
backward combines both output paths.

Finally, every observed configuration maps onto a generic **Level-1** task:
`linear_no_bias`, `rmsnorm`, `rope`, `swiglu`, `cross_entropy` and the newly
added [`causal_gqa_attention`](../ops/level1/causal_gqa_attention/).
Each gains a `qwen3_0_6b_observed` suite derived from the snapshot; the
Llama-derived defaults are untouched.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping mapping
```

See [docs/QWEN3_LEVEL4.md](../../../docs/QWEN3_LEVEL4.md).

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
