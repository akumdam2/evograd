# Agentic and evolutionary GPU-kernel optimization

## Scope and method

This report audits evolutionary program optimizers, agentic kernel generators,
and kernel benchmarks with a narrow question: what do they *actually* do with
heterogeneous shapes and workload-specific implementations?

The audit was performed on 2026-07-24. Repository behavior was read at the
exact commits in the [source ledger](#source-ledger); paper-only claims are
identified as such. “Shape-aware” is used in seven deliberately separate
senses:

1. one candidate is evaluated on several shapes;
2. scores are aggregated across shapes;
3. candidates that specialize on different shapes are retained;
4. separate shape specialists are optimized;
5. a shape-regime boundary is discovered;
6. a runtime dispatcher selects among specialists;
7. kernels and dispatch policy are optimized jointly.

Passing level 1 or 2 is not evidence for levels 3–7. Likewise, “evolution” is
reserved for population, lineage, or archive-based search; sequential
generate-test-repair loops are described as iterative refinement.

## 1. Executive summary

The field separates into four practical families:

- **Population evolution:** OpenEvolve, ShinkaEvolve, FunSearch, AlphaEvolve,
  and KernelFoundry maintain populations, islands, archives, clusters, or
  quality-diversity maps. GEPA is a reflective evolutionary optimizer with
  unusually strong per-instance/Pareto retention.
- **Beam or stochastic search:** KernelAgent, Autocomp, and POLCA retain
  several candidates, but use beam, priority-queue, or stochastic search
  rather than a conventional island evolutionary algorithm.
- **Iterative and multi-agent refinement:** GEAK, K-Search, AutoKernel,
  KernelSkill, KernelBlaster, CuTeGen, TritonForge, and KernelPro repeatedly
  rewrite one task or coordinate specialists. Some retain a best-so-far
  candidate or replay memory; that alone does not make them population
  evolution.
- **Benchmark/deployment infrastructure:** KernelBench, KernelBench-X, both
  projects named TritonBench, KernelGYM, and FlashInfer-Bench define
  correctness and performance measurements. FlashInfer-Bench additionally
  builds a deployable exact-key solution table.

No verified public system in this audit combines all of: evolutionary
shape-specific niches, independent specialist evolution, data-derived regime
boundaries, a generated runtime dispatcher, and joint kernel/router search.
That is a statement about the inspected public sources—not a “first” or
novelty claim.

The strongest **component-level precedents** are:

- **GEPA** for retaining candidates that are best on individual validation
  instances or instance/objective cells. Treating shapes as GEPA instances
  could preserve shape specialists, but this is an interpretation; GEPA does
  not present a GPU-shape dispatcher.
- **GEAK** for production-shape capture, workload weighting, per-shape AITER
  tuning, regime-specific implementations, exact guards/fallbacks, and
  end-to-end validation. It is the closest kernel-deployment precedent, but
  its current workflow is multi-agent optimization rather than
  population/Pareto evolution.
- **FlashInfer-Bench** for selecting the fastest correct solution per exact
  workload key and deploying a runtime lookup table with fallback. The
  selection is post-hoc over submitted solutions, not an evolutionary
  specialist search or learned boundary.
- **KernelFoundry** for genuine MAP-Elites kernel evolution. Its public paper
  describes behavioral niches such as memory and parallelism strategy, not
  niches indexed by input shape.
- **TVM MetaSchedule/Ansor** for cost-model-guided evolutionary schedule search
  and a database keyed by serialized workload, tensor shapes/dtypes, and
  hardware target. Static shapes can receive independently tuned schedules,
  although this is compiler scheduling rather than LLM evolution of arbitrary
  autograd-pair source or a learned dynamic-shape router.
- **Triton autotune** for benchmarking configuration variants whenever declared
  key arguments change and caching the best configuration. This is exact-key
  shape/config specialization inside a kernel, not a population of separately
  generated implementations.

Accordingly, **TVM MetaSchedule/Ansor is the strongest direct precedent if
“shape-specialized evolution” includes evolutionary schedule search performed
separately for static shape workloads**. No audited public source demonstrates
the narrower end-to-end combination of arbitrary evolved forward/backward
programs, per-shape frontier retention, learned regime boundaries, and a
co-optimized runtime router.

The most defensible description of Evograd is therefore:

> a declaration-driven backward-kernel system that places a shape-distribution
> evaluator around OpenEvolve, separately searches a generalist and
> declaration-defined small/large specialists, measures their crossovers, and
> emits a threshold dispatcher while optimizing backward speed, full-step
> speed, correctness, and saved-state memory.

The unsafe claims would be “the first specialist kernel optimizer,” “the first
per-shape dispatcher,” “the first evolutionary kernel search,” or “jointly
evolves kernels and dispatch.” GEAK/FlashInfer-Bench already provide direct
specialization/dispatch precedents, KernelFoundry provides evolutionary
quality-diversity kernel search, TVM provides evolutionary per-workload schedule
search, Triton provides exact-key configuration autotuning, and current Evograd
searches its threshold *after* the kernel runs rather than co-evolving it.

### Immediate Evograd findings

Current Evograd at `d21dbe1` is more shape-aware than the original six-operator
handoff:

- it has 18 declared forward/backward operators;
- 12 declarations define full/small/large suites and a scalar regime feature;
- the one-call API launches independent full, small, and large OpenEvolve runs;
- all returned programs are re-evaluated over the full grid;
- a geometric midpoint sweep finds a small/large threshold;
- the emitted forward stores a non-tensor route tag so backward uses the same
  specialist;
- correctness gates forward output and every requested gradient, including
  shape and dtype;
- the evaluator measures backward-from-saved, raw full step,
  autograd-bound full step, and saved tensor bytes.

Five immediate limitations should be treated as experiment-design work rather than
papered over:

1. `case_weight` is declared but the one-call specialist path does not pass
   those weights to `EVOGRAD_GEOMEAN_WEIGHTS`; the nominally weighted specialist
   score is currently a uniform geometric mean.
2. Numerical correctness runs only on the correctness suite; benchmark-only
   shapes receive an execution smoke test before timing, not an oracle
   comparison.
3. The generalist is reported but is not an eligible arm in threshold routing,
   and the threshold deployment is not compared against the generalist before
   choosing the final artifact.
4. The dispatch score is reconstructed from component measurements using raw
   full-step speedup; it does not time the emitted dispatcher or include
   backward-only speed, memory, case weights, or routing overhead.
5. Specialist regimes and their single scalar feature are declared in advance.
   OpenEvolve populations remain scalar-scored within each run; no per-shape
   frontier, shape-indexed MAP-Elites archive, medium regime, or joint
   kernel/router mutation exists.

## Source ledger

Cloned repository links below are commit-pinned. Documentation-only compiler
entries name the official rolling documentation audited on 2026-07-24.

| ID | Source | Audited revision |
|---|---|---|
| OE | [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve/tree/411fb59c886c18704caaffb611e17cf9e7d824d2) | `411fb59c886c18704caaffb611e17cf9e7d824d2` |
| SE | [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve/tree/b67a07328ab7e21e999d9e20a44f4f0054a4b83c) | `b67a07328ab7e21e999d9e20a44f4f0054a4b83c` |
| GEPA | [GEPA](https://github.com/gepa-ai/gepa/tree/f919db0a622e2e9f9204779b81fe00cc1b2d808f), [paper](https://arxiv.org/abs/2507.19457) | `f919db0a622e2e9f9204779b81fe00cc1b2d808f`; paper v2 |
| FS | [FunSearch public implementation](https://github.com/google-deepmind/funsearch/tree/cc53f274237d7ab05c19df939edbc1f9616a7c19) | `cc53f274237d7ab05c19df939edbc1f9616a7c19` |
| AE | [AlphaEvolve white paper](https://arxiv.org/abs/2506.13131), [official post](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | public paper/blog; implementation unreleased |
| AEV | [A-Evolve](https://github.com/A-EVO-Lab/a-evolve/tree/c9d4789f2be499589d543aa08e74d05d10d93177), [paper](https://arxiv.org/abs/2602.00359) | `c9d4789f2be499589d543aa08e74d05d10d93177` |
| GEAK | [GEAK](https://github.com/AMD-AGI/GEAK/tree/ab7fa983d4c94c2b3b50426a57ea60e2b30909a8), [paper](https://arxiv.org/abs/2507.23194) | `ab7fa983d4c94c2b3b50426a57ea60e2b30909a8` |
| KA | [KernelAgent](https://github.com/meta-pytorch/KernelAgent/tree/e0647170da36ef9b059ac0bd3d60103aa4ed378b) | `e0647170da36ef9b059ac0bd3d60103aa4ed378b` |
| KB | [KernelBench](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1), [paper](https://arxiv.org/abs/2502.10517) | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` |
| KBX | [KernelBench-X](https://github.com/BonnieW05/KernelBenchX/tree/fd4192293bf9a8c645327a9d46aa1e807f1f9cf2), [paper](https://arxiv.org/abs/2605.04956) | `fd4192293bf9a8c645327a9d46aa1e807f1f9cf2` |
| FIB | [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench/tree/40e6ca7844b514eb4b1c7edba6d6a7377df57870), [paper](https://arxiv.org/abs/2601.00227) | `40e6ca7844b514eb4b1c7edba6d6a7377df57870` |
| MTB | [Meta PyTorch TritonBench](https://github.com/meta-pytorch/tritonbench/tree/ad8e430730919be4bfb4524eff09ad5faf919afa) | `ad8e430730919be4bfb4524eff09ad5faf919afa` |
| TTB | [THUNLP TritonBench](https://github.com/thunlp/TritonBench/tree/603e28a5050e8c268f6883a69709d477a272d49a), [paper](https://arxiv.org/abs/2502.14752) | `603e28a5050e8c268f6883a69709d477a272d49a` |
| KS | [K-Search](https://github.com/caoshiyi/K-Search/tree/53c8fab9a5e8fab2c86610d24fbec5067f90e115) | `53c8fab9a5e8fab2c86610d24fbec5067f90e115` |
| AK | [AutoKernel](https://github.com/RightNow-AI/autokernel/tree/78435821cc3d5756ba6ee1785c397f6d8fa8c90d) | `78435821cc3d5756ba6ee1785c397f6d8fa8c90d` |
| KBLA | [KernelBlaster](https://github.com/NVlabs/KernelBlaster/tree/84237f91a391971e566cd9066bfb7e9514e957ee) | `84237f91a391971e566cd9066bfb7e9514e957ee` |
| TF | [TritonForge](https://github.com/RLsys-Foundation/TritonForge/tree/b61331ad2c6fd0c6b315b4270621474ff7120d6b) | `b61331ad2c6fd0c6b315b4270621474ff7120d6b` |
| KM | [KernelSkill/KernelMem](https://github.com/0satan0/KernelMem/tree/8b57ccc9adc2ae2f11fc487fd458a7ecc1ea014d), [paper](https://arxiv.org/abs/2603.10085) | `8b57ccc9adc2ae2f11fc487fd458a7ecc1ea014d` |
| POLCA | [POLCA paper](https://arxiv.org/abs/2603.14769), [project](https://github.com/rlx-lab/POLCA/tree/356d0177d034df8c70bf351f5f62c93dbb226b41), [Trace optimizer](https://github.com/xuanfeiren/Trace/tree/4ee52f12f0bb328dfee2f16b6c1801232c6ccf46) | project `356d017…`; optimizer `4ee52f1…` |
| CG | [CuTeGen](https://github.com/taratt/cutegen/tree/0ebe185b9f9d50cf8720878695ce937c2853caae), [paper](https://arxiv.org/abs/2604.01489) | `0ebe185b9f9d50cf8720878695ce937c2853caae` |
| SG | [SpecGen paper](https://arxiv.org/abs/2606.17518) | paper only; no official code linked |
| KF | [KernelFoundry paper](https://arxiv.org/abs/2603.12440) | paper only; no official code found |
| AC | [Autocomp](https://github.com/ucb-bar/autocomp/tree/a56ce8154c6992648348517489ff5db3d8267798), [paper](https://arxiv.org/abs/2505.18574) | `a56ce8154c6992648348517489ff5db3d8267798` |
| KGYM | [KernelGYM](https://github.com/hkust-nlp/KernelGYM/tree/3a84417f8c0efaadb215ef638b37d12e71ed20f3) | `3a84417f8c0efaadb215ef638b37d12e71ed20f3` |
| AVO | [AVO paper](https://arxiv.org/abs/2603.24517) | paper only; no official code found |
| KPRO | [KernelPro paper](https://arxiv.org/abs/2606.26453) | paper only; no official code found |
| EVO | [Evograd](https://github.com/akumdam2/evograd/tree/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df) | `d21dbe10a47ecb31e19ce6af9b18a43ca91a32df` |
| TRI | [Triton `autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html) | official main documentation, accessed 2026-07-24 |
| TVM | [MetaSchedule RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md), [database API](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1s__tir_1_1meta__schedule_1_1Database.html) | official rolling documentation, accessed 2026-07-24 |
| PTI | [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) | official stable documentation, accessed 2026-07-24 |

## 2. Comparison matrix

The requested matrix is split vertically into three tables so every system
still has one row and all requested fields remain readable.

### 2A. Search and candidate mechanics

| System | Repository or paper | Optimization target | Search category | Unit of mutation | Candidate population/archive | Selection mechanism | Feedback to model |
|---|---|---|---|---|---|---|---|
| OpenEvolve | OE | Complete files or marked code blocks | Island evolutionary search + MAP-Elites-style feature map | LLM diff or full rewrite; inspirations can act as crossover context | Per-island population, per-island feature-map owners, global scalar top archive | Random exploration; archive exploitation; otherwise fitness-weighted parent; inspirations from elites/map/random | Metrics, evaluator artifacts, prior code/diff, feature coordinates |
| ShinkaEvolve | SE | Source programs/code regions; optionally prompts | Island evolution, optional dynamic islands, async evaluation | Diff/full/cross patch; optional system-prompt mutation | SQLite population, per-island populations, scalar/embedding archive | Weighted, power-law, beam, best-of-N, sequential, or bandit/model routing | Metrics, evaluator text, genealogy, incorrect-program repair context |
| GEPA | GEPA | Prompt/program text components | Reflective evolution with instance/objective Pareto frontiers | Reflection-driven component rewrite; optional merge | All candidates plus instance/objective/hybrid/cartesian frontiers | Candidate chosen in proportion to frontier coverage (“dominator”); optional top-k Pareto | Per-example trajectories, scores, objective scores, actionable side information |
| FunSearch | FS | Body of one decorated function | Island evolutionary program search | LLM completion from sampled programs | Clusters by per-test behavior signature within islands | Prefer high-scoring clusters/programs; periodically reset weaker islands | Program text and scalar evaluator results |
| AlphaEvolve | AE | Whole codebases/algorithms | Evolutionary coding agent (public high-level description) | Gemini-generated code changes | Evolutionary database; unreleased details | Multiple evaluators guide selection; exact public mechanics incomplete | Automated evaluator scores and program context |
| A-Evolve | AEV | Agent workspace: prompts, skills, memory, tools | Iterative workspace evolution with held-out gate | Workspace edits | Git state/current workspace; no required population archive | Solve → observe → evolve → gate → reload; rollback on failed gate | Task traces, performance observations, held-out gate |
| GEAK | GEAK | AMD kernels, configs, backends, serving stack | Multi-agent profiler-guided refinement | File/config/backend/overlay changes | Verified incumbent patches and learned knowledge; not a population | Director prioritizes Amdahl-weighted bottlenecks; specialists iterate and integrate | E2E/operator latency, shape profiles, rocprof, correctness, regime weights |
| KernelAgent | KA | Triton/CUDA kernels and composed subgraphs | Beam search with profiler-guided mutation | Kernel code rewrite for selected bottleneck | Top-N runtime beam; PTX-fingerprint dedup; scalar program DB | Top runtime kernels × top bottlenecks × models/samples; half-top/half-random inspirations | Correctness, runtime, NCU, roofline/bottleneck diagnosis, SOL |
| Triton autotune | TRI | Meta-parameter configs for one `triton.jit` kernel | Online exhaustive/pruned autotuning per declared key | `triton.Config` values such as block size/warps | Best timed config cached per key; no evolutionary population | Benchmark every/pruned config when key changes; take fastest | Timing only; optional hooks/reset/restore, not LLM feedback |
| TVM MetaSchedule / Ansor | TVM | TensorIR schedule traces/decisions for a workload and target | Cost-model-guided evolutionary schedule search plus replay | Schedule-trace sampling decisions/mutators | Measured tuning records and top-K schedules per serialized workload/target | Cost-model ranking with epsilon-greedy exploration; compile and measure selected schedules | On-device runtime, trace features, validity/resource postprocessors |
| TorchInductor max-autotune | PTI | Compiler-generated Triton/template algorithms and configs | Profile-guided compiler autotuning, not population evolution | Algorithm/template/config choice | Compiler caches guarded specializations/selected algorithms; version-dependent | Profile candidate matmul/conv choices and take fastest valid option | Runtime timing and compiler correctness assumptions, no model |
| KernelBench | KB | Benchmark submissions, usually one `ModelNew` per task | Benchmark, not a search algorithm | N/A | N/A | N/A | Correctness and timing are available to external agents |
| KernelBench-X | KBX | Triton implementation of API-level tasks | Benchmark plus iterative evaluation controller, not population evolution | Whole generated implementation | Per-task current output/history; no specialist population | External generator/controller may revise failed/slow task | Recursive correctness errors, total runtime, speedup, throughput |
| FlashInfer-Bench | FIB | Kernel solutions for definitions × workloads | Benchmark and post-hoc solution selection; optional external agents | Submitted solution package | All traces; best correct trace per exact apply key | Maximum speedup per key; most per-key wins defines fallback solution | Correctness, latency, error, optional NCU/sanitizer feedback |
| Meta PyTorch TritonBench | MTB | Provider implementations of operators | Benchmark, not search | N/A | N/A | N/A | Per-input latency/speedup/throughput and optional profiler metrics |
| THUNLP TritonBench | TTB | LLM-generated Triton code | Generation benchmark, not a built-in optimizer | Whole output program | Generated samples, no evolving population | External model/sample choice | Code similarity, call/execution accuracy, efficiency |
| K-Search | KS | Kernel source plus an intrinsic decision-tree “world model” | Iterative search with co-evolving reasoning model; not population evolution | Action-conditioned code rewrite and tree refinement | One global best plus frontier action nodes | Deterministic frontier action scoring by expected value/difficulty/rating | Full/partial workload metrics, failure trace, world-model outcomes |
| AutoKernel | AK | One `kernel.py` in a model or KernelBench task | Autoresearch-style single-incumbent iteration | Direct file edit | Current and best result log | Keep/revert against fixed benchmark; primary size determines score | Five-stage correctness, runtime, VRAM, profiler/roofline summaries |
| KernelBlaster | KBLA | CUDA kernel plus C++ harness | Parallel rollout refinement with replay memory; not evolutionary population | Sequential code change inside each rollout | Best step across rollouts; optimization-strategy replay DB | Strategies sampled by relevance; best elapsed-cycle candidate returned | Compile/run correctness, NCU cycles and bottleneck knowledge |
| TritonForge | TF | Triton/CUDA kernels generated by a trained model | SFT/RL multi-turn generation | Model response/repaired kernel | Samples/turns, not an evolutionary archive | RL policy and up to configured repair turns | Correctness and performance reward, compiler/runtime errors |
| KernelSkill / KernelMem | KM | Kernel source and reusable optimization memory | Multi-agent iterative refinement | Kernel rewrite/repair | Current/base/best lineage plus short/long-term strategy memory | Update base after material gain; retain scalar best | Correctness, speedup, NCU/NSYS, prior strategy outcomes |
| POLCA | POLCA | General programs/prompts; linked experiment optimizes CUDA | Stochastic generative priority search | LLM proposal from queued parent(s) | Priority queue + epsilon-net diversity over code embeddings | Mean/UCB priority; top candidates become parents | Scalar objective, correctness/failure, summarized prior trials |
| CuTeGen | CG | CuTe DSL kernels | Structured sequential generate-test-refine | One node/code rewrite | Single chain with current/global best | Advance with passed proposal; bounded repair retries | Compile, correctness, mean runtime, delayed NCU feedback |
| SpecGen | SG | GPU kernel candidates produced during one reasoning trace | Speculative parallel generation, not population evolution | Fork a non-reasoning candidate from partial trace | Parallel candidates tied to one parent reasoning trajectory | Validate/profile in parallel; stop reasoning when satisfactory | Validation and profiling results |
| KernelFoundry | KF | GPU kernels and meta-prompts | MAP-Elites quality-diversity evolution | Kernel/meta-prompt variation and template-parameter tuning | Quality-diversity map indexed by kernel behavior | Elite per behavior cell | Correctness/performance and behavior descriptors |
| Autocomp | AC | Accelerator programs across CUDA/JAX/Pallas/etc. | Hardware-in-the-loop beam search | Plan then implementation rewrite; optional pairwise combine | Candidate repository/top-N beam | Scalar correct-performance rank; optional parent pairing | Compilation, correctness, measured hardware performance |
| KernelGYM | KGYM | Kernel-generation policies/interactions | Distributed evaluation/RL environment, not evolution itself | Agent response/kernel | Training rollouts, not a built-in evolutionary archive | Defined by external RL/agent trainer | Isolated compile, random-trial correctness, timing, profiling |
| AVO | AVO | GPU kernel code | Paper-described agent-as-variation evolutionary search | Propose/repair/critique kernel edits | Evolutionary population/lineages and knowledge base | Paper-described lineage/fitness selection | Execution, correctness, performance, accumulated knowledge |
| KernelPro | KPRO | CuTe/raw CUDA kernels | Multi-agent MCTS | Search-tree action/code expansion | MCTS tree with progressive widening and search memory | Asymmetric branching/log reward/dead-end pruning | Correctness, runtime, NCU/SASS/NSYS; paper also reports energy |

### 2B. Objectives, inputs, specialization, and deployment

| System | Objective structure | Multiple-input handling | Performance aggregation | Shape-aware objective | Specialist preservation | Runtime dispatch |
|---|---|---|---|---|---|---|
| OpenEvolve | Evaluator-defined metrics; `combined_score` is the scalar fitness, else mean numeric metrics excluding feature dimensions | Whatever evaluator runs | Framework does not aggregate cases itself | Only if evaluator emits it | Generic MAP cells can preserve user-defined niches, but no built-in shape semantics | No |
| ShinkaEvolve | Weighted normalized archive criteria/scalar combined score; optional embedding crowding | Whatever evaluator runs | Evaluator-defined | Only if evaluator supplies it | Islands are independent populations, not semantic shape niches | No |
| GEPA | Aggregate candidate score plus per-instance/per-objective frontiers | Validation examples/minibatches; test set outside optimization budget | Arithmetic mean of evaluated examples | **Potentially:** make each shape an example (interpretation) | Yes, per-instance or instance/objective frontier | No |
| FunSearch | Scalar reducer plus vector behavior signature | Evaluator may run multiple test inputs | Public example uses the last test score as scalar; signature contains sorted per-test scores | Indirect behavior vector, not shape objective | Clusters can retain distinct score signatures, not explicit shape elites | No |
| AlphaEvolve | Multiple automated evaluators; public scalar/multiobjective internals incomplete | Demonstrated across varied algorithm tasks | Unreleased | Unclear | Evolutionary diversity is explicit; shape specialists are not publicly established | No public evidence |
| A-Evolve | Task reward plus held-out gate | Repeated task instances/held-out checks | Task-defined | No kernel-shape mechanism | No | No |
| GEAK | Primary weighted end-to-end speed ratio, regime gates/floors; secondary geomean | Production traces, shapes, dtypes, call frequencies | `Σw / Σ(w/speedup)` with baseline-time × call-count weights | Yes | Agents may build regime variants; per-shape AITER configs persist | Yes: existing AITER lookup tables and exact guards/fallbacks; not co-evolved |
| KernelAgent | Correctness constraint then minimum runtime; SOL retained diagnostically | Generation dedups static shape signatures; optimization task is a concrete kernel/input | Scalar `time_ms` per optimization target | Static-signature aware, not distribution fitness | Top-N implementation beam for one target, not shape niches | Composition by subgraph signature, not a learned specialist crossover |
| Triton autotune | Minimum measured latency among supplied configs | Each distinct tuple of named key arguments triggers tuning | No cross-key aggregate; independent winner per exact key | Yes, at configuration level | Best config survives per key | Yes, keyed cached config selection inside the JIT wrapper; no learned boundary |
| TVM MetaSchedule / Ansor | Correctness/validity constraints then predicted/measured schedule latency; task scheduler can weight tasks | Static operator workloads include shape/dtype and target | Per-workload tuning records; task scheduler allocates a global trial budget | Yes for separately represented static workloads | Top schedules retained per workload/target | Database is queried when compiling a matching workload; not a general dynamic runtime threshold router |
| TorchInductor max-autotune | Fastest profiled correct/eligible algorithm; compiler modes also expose shape padding | Compiler specializes under input guards and can recompile for changed shapes | Selection is local to compiled specialization, not a published cross-shape aggregate | Yes at compiler specialization/config level | Cached compiled choices, not an objective-space archive | Guarded compiled specialization/cache, not an explicit user-visible specialist portfolio |
| KernelBench | Correctness then runtime/fast_p by task | Usually one concrete shape per task; random values across trials | Per-task speedup; benchmark summaries can use geometric mean over correct tasks | No within-task shape distribution in the standard tasks | No | A submission may contain manual guards, but dispatcher quality is not a separate metric |
| KernelBench-X | Correctness constraint; task speedup from summed case runtime | Deterministic multi-case/multi-shape suite per task | `Σ reference time / Σ candidate time`; global sum ratio and per-task arithmetic mean | Level 1–2 only | No | One submitted implementation can branch, but no portfolio selection |
| FlashInfer-Bench | Correctness gate then latency/speedup; failed trace-set workload zeros score | Production-like workload traces, axes/dtypes | Mean per-workload speedup; mean of per-definition best scores | Yes, per exact workload | Post-hoc fastest correct solution retained per exact key | **Yes:** exact axis-value key → solution, with fallback/definition-best |
| Meta PyTorch TritonBench | Accuracy plus selectable latency/speedup/throughput/profiler metrics | Generator-defined shape sequence; fwd/bwd/fwd+bwd modes | Per-input metrics; arithmetic mean summary | Level 1–2 | No | Provider may dispatch internally; benchmark has no learned router |
| THUNLP TritonBench | Staged similarity, API, execution, efficiency | G/T suites include multiple calls/shapes | Per-op total reference time / total generated time; arithmetic mean across ops | Level 1–2 | No | No portfolio/dispatcher evaluation |
| K-Search | Correctness gate + scalar workload score | Fixed/random subsets for FlashInfer; several TriMul configs | FlashInfer arithmetic mean speedup; TriMul geometric mean | Aggregated distribution objective | No specialist archive | No |
| AutoKernel | Correctness gate; primary-size throughput/latency; reports memory | Shape sweep for correctness and benchmark sizes | Primary largest/biggest size drives keep/revert | Mostly no; multiple cases are guards/reports | One incumbent | No |
| KernelBlaster | Correctness then NCU elapsed cycles | One fixed KernelBench-CUDA task per run | Scalar cycles | No | Strategies persist across tasks, not kernels per shape | No |
| TritonForge | Correctness reward + capped performance reward | KernelBench task inputs; multiple hardware backends | Task scalar | No shape-specialist archive | No | No |
| KernelSkill / KernelMem | Correctness then scalar speedup | One fixed KernelBench task/shape per run | Scalar speedup | Explicit fixed-shape specialization, but no distribution objective | Knowledge/lineage retained, not shape portfolio | No |
| POLCA | Task-supplied scalar; correctness failure penalized | Linked KernelBench experiment runs 16 fixed matmul tasks separately | Scalar speedup per task | No within-run distribution | Embedding-diverse code can survive, not objective niches | No |
| CuTeGen | Correctness then mean wall time | One fixed KernelBench problem in main search; benchmark scripts can sweep | Mean runtime of repetitions | No distribution search | One chain/global best | No |
| SpecGen | Correctness/profiling satisfaction under time/token budget | Candidate evaluation inputs are task-defined | Paper’s kernel evaluator decides | No public specialist mechanism | Parallel speculative alternatives only | No |
| KernelFoundry | Quality and behavior descriptors | KernelBench/robust/custom inputs | Paper-defined fitness per niche | Behavior-aware, but public niches are not input shapes | **Yes**, by MAP-Elites behavioral cell | No reported shape router |
| Autocomp | Correctness constraint + scalar measured performance | Task/hardware-specific tests | Scalar score | No explicit shape portfolio | Top-N implementations for a task | No |
| KernelGYM | Correctness and speed reward | Multiple random-value trials for a fixed task shape | Task scalar | No | External trainer-dependent | No |
| AVO | Correctness + performance over evaluated attention configurations | Several evaluated configurations/hardware setup in paper | Paper-defined fitness | Multi-configuration evaluation is reported; public detail insufficient for shape niches | Population diversity, but shape-specific survival is unverified | No public evidence |
| KernelPro | Correctness + runtime; energy also evaluated in paper | Task/configuration dependent | Search reward is scalar/log-scaled | No explicit shape portfolio reported | MCTS branches, not deployed specialists | No |

### 2C. Validation, profiling, cost, and relationship to Evograd

| System | Correctness validation | Hardware profiling | Held-out evaluation | Search-cost reporting | Similarity to Evograd | Difference from Evograd | Supporting source |
|---|---|---|---|---|---|---|---|
| OpenEvolve | Evaluator-defined; invalid candidates retain low metrics/artifacts | Evaluator-defined | User-defined | Iterations, evaluations, token/cost accounting hooks | Evograd directly uses its LLM evolution/database | No native kernel oracle, per-shape archive, specialist groups, or dispatcher | [config](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/openevolve/config.py), [database](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/openevolve/database.py), [fitness](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/openevolve/utils/metrics_utils.py) |
| ShinkaEvolve | Evaluator-defined; explicit incorrect-fix mode | Evaluator-defined | User-defined | Evaluations/LLM accounting and async/Slurm controls | Closest general island alternative | More selection/model/island policies; still no shape semantics/deployment | [configuration](https://github.com/SakanaAI/ShinkaEvolve/blob/b67a07328ab7e21e999d9e20a44f4f0054a4b83c/docs/configuration.md), [parents](https://github.com/SakanaAI/ShinkaEvolve/blob/b67a07328ab7e21e999d9e20a44f4f0054a4b83c/shinka/database/parents.py), [islands](https://github.com/SakanaAI/ShinkaEvolve/blob/b67a07328ab7e21e999d9e20a44f4f0054a4b83c/shinka/database/islands.py) |
| GEPA | Adapter-defined per example | Adapter-defined | Explicit `test_set`, outside optimization budget | Metric-call budget; paper reports rollout savings | Per-instance frontiers are a strong design precedent | Optimizes text components; emits one aggregate-best candidate, no kernel dispatcher | [API](https://github.com/gepa-ai/gepa/blob/f919db0a622e2e9f9204779b81fe00cc1b2d808f/src/gepa/api.py), [state](https://github.com/gepa-ai/gepa/blob/f919db0a622e2e9f9204779b81fe00cc1b2d808f/src/gepa/core/state.py), [paper](https://arxiv.org/abs/2507.19457) |
| FunSearch | Sandboxed evaluator | No built-in GPU profiler | Problem-defined | Paper-level distributed budget; public code is illustrative | Islands and behavior-vector clustering could inspire diversity | Scalar reducer and domain differ; no GPU/full-step/memory/dispatch | [database](https://github.com/google-deepmind/funsearch/blob/cc53f274237d7ab05c19df939edbc1f9616a7c19/implementation/programs_database.py), [evaluator](https://github.com/google-deepmind/funsearch/blob/cc53f274237d7ab05c19df939edbc1f9616a7c19/implementation/evaluator.py) |
| AlphaEvolve | Multiple automated evaluators | Domain evaluators; kernel claims in paper | Task-dependent | Paper reports large deployments/results, not reproducible public config | Whole-code evolutionary optimization with evaluators | Closed implementation; shape-specialist claims cannot be verified | [white paper](https://arxiv.org/abs/2506.13131), [official post](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) |
| A-Evolve | Task tests plus held-out gate/rollback | Not kernel-specific | Yes, explicit gate | Task/model-run reporting | Evolves the optimizer/agent around tasks | Agent workspace target, no GPU shape objective | [repository](https://github.com/A-EVO-Lab/a-evolve/tree/c9d4789f2be499589d543aa08e74d05d10d93177), [paper](https://arxiv.org/abs/2602.00359) |
| GEAK | Recorded-I/O/random parity, graph/compile and E2E checks | rocprof and workload-shape capture | Warm E2E A/B on actual serving workload | Workflow/budget documented operationally; not a uniform candidate-count benchmark | Closest end-to-end shape weighting, regimes, and deployment precedent | AMD serving optimization, not OpenEvolve population; router not jointly evolved | [weighting](https://github.com/AMD-AGI/GEAK/blob/ab7fa983d4c94c2b3b50426a57ea60e2b30909a8/e2e_workflow/scripts/attribute_weights.py), [shape parsing](https://github.com/AMD-AGI/GEAK/blob/ab7fa983d4c94c2b3b50426a57ea60e2b30909a8/e2e_workflow/scripts/parse_profile.py), [workflow](https://github.com/AMD-AGI/GEAK/blob/ab7fa983d4c94c2b3b50426a57ea60e2b30909a8/e2e_workflow/knowledge/e2e_optimization.md) |
| KernelAgent | Task-specific tests before timing | NCU, bottleneck analyzer, SOL/roofline | No standard held-out shape set | Beam width/workers/warmup/repeats/configurable rounds | Hardware-guided multi-candidate kernel optimization | Optimizes concrete targets; no backward saved-state objective or shape router | [beam config](https://github.com/meta-pytorch/KernelAgent/blob/e0647170da36ef9b059ac0bd3d60103aa4ed378b/examples/configs/beam_search.yaml), [database](https://github.com/meta-pytorch/KernelAgent/blob/e0647170da36ef9b059ac0bd3d60103aa4ed378b/kernel_perf_agent/kernel_opt/database/base.py) |
| Triton autotune | Assumes supplied configs preserve kernel semantics; repeated writes require reset/restore care | Internal benchmarking, optional user `do_bench` | No hidden-key protocol | Cost is configs benchmarked per new key; pruning and disk timing cache are optional | Direct prior for shape-keyed config selection | Tunes meta-parameters of one kernel, not arbitrary source/autograd pairs; no population/full-step/memory | [official API](https://triton-lang.org/main/python-api/generated/triton.autotune.html) |
| TVM MetaSchedule / Ansor | Build/run validity and user integration correctness | On-device measurements plus learned cost model | Workload/target records are reusable, but no automatic held-out-shape study | Explicit global/per-task trial budgets and records | Strongest classical precedent for evolutionary per-workload schedule specialists | Schedule IR rather than LLM source; generally compile-time workload lookup, no inferred dynamic boundary | [RFC search](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md#42-exploring-the-design-space), [database](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1s__tir_1_1meta__schedule_1_1Database.html) |
| TorchInductor max-autotune | Compiler/runtime validation and fallbacks; exact internals are release-dependent | Profiles supported matmul/convolution choices | No standard held-out-shape benchmark | Compile/autotune overhead is cached but workload-dependent | Production compiler precedent for guarded shape specialization | Not an agent/evolution system and does not expose Evograd’s saved-state/full-step objective | [official `torch.compile` modes](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) |
| KernelBench | `torch.allclose` across randomized trials, dtype tolerances | Optional external profiling | Standard test inputs vary values, not unseen shapes | Evaluation counts and benchmark paper | Common kernel-generation benchmark | Mostly forward/fixed shape; no portfolio, memory, or dispatch score | [repo](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1), [paper](https://arxiv.org/abs/2502.10517) |
| KernelBench-X | Recursive output contract, exact shape/dtype, dtype/task tolerances | Golden throughput/bandwidth metrics | Deterministic task cases; hardware-specific goldens expose transfer issues | Benchmark generation/evaluation reporting | Multi-shape candidate evaluation and total-time aggregation | API breadth/forward orientation; no specialist retention or runtime router | [correctness](https://github.com/BonnieW05/KernelBenchX/blob/fd4192293bf9a8c645327a9d46aa1e807f1f9cf2/EVAL/1_exe_acc.py), [efficiency](https://github.com/BonnieW05/KernelBenchX/blob/fd4192293bf9a8c645327a9d46aa1e807f1f9cf2/EVAL/2_efficiency.py) |
| FlashInfer-Bench | Definition-specific tolerance, sampling/low-bit/DSA validators | Optional NCU and sanitizer agents | Workload traces can be separated by user; not inherently a hidden split | Trace/evaluation counts and timing settings | Strongest exact workload-specialist deployment precedent | No population search or inferred continuous threshold; primarily forward/inference | [scoring](https://github.com/flashinfer-ai/flashinfer-bench/blob/40e6ca7844b514eb4b1c7edba6d6a7377df57870/flashinfer_bench/data/trace_set.py), [table](https://github.com/flashinfer-ai/flashinfer-bench/blob/40e6ca7844b514eb4b1c7edba6d6a7377df57870/flashinfer_bench/apply/table.py), [runtime](https://github.com/flashinfer-ai/flashinfer-bench/blob/40e6ca7844b514eb4b1c7edba6d6a7377df57870/flashinfer_bench/apply/runtime.py) |
| Meta PyTorch TritonBench | Accuracy against selected baseline | PyTorch profiler, NCU, power, compile-time options | Input sampling can reserve IDs manually; no standard search split | Benchmark runtime/config, not search cost | Excellent harness precedent for fwd/bwd/full-step-like modes and shape grids | No generation/search/specialist deployment | [repository](https://github.com/meta-pytorch/tritonbench/tree/ad8e430730919be4bfb4524eff09ad5faf919afa), [operators](https://github.com/meta-pytorch/tritonbench/tree/ad8e430730919be4bfb4524eff09ad5faf919afa/tritonbench/operators) |
| THUNLP TritonBench | Call + execution comparison; legacy G path compares printed outputs | Peak GB/s/TFLOPS computation, not iterative profiler feedback | Retrieval corpus/eval sets, not shape generalization within an optimized kernel | Dataset/model evaluation | Triton generation and multi-call efficiency precedent | Separate benchmark from Meta TritonBench; no autograd-pair evolution/dispatch | [G evaluator](https://github.com/thunlp/TritonBench/tree/603e28a5050e8c268f6883a69709d477a272d49a/EVAL/eval_G), [T evaluator](https://github.com/thunlp/TritonBench/tree/603e28a5050e8c268f6883a69709d477a272d49a/EVAL/eval_T), [paper](https://arxiv.org/abs/2502.14752) |
| K-Search | All selected workloads must pass; final full-workload pass | Detailed trace/benchmark feedback; world model | Random/fixed search subset then full evaluation | Defaults include 20 FlashInfer, 300 TriMul, 50 KernelBench rounds | Searches kernels across workload sets | One scalar best and no shape elites/router | [repository](https://github.com/caoshiyi/K-Search/tree/53c8fab9a5e8fab2c86610d24fbec5067f90e115) |
| AutoKernel | Smoke, shape sweep, numerical stability, determinism, edge tests | Profiler/roofline summaries | Correctness sweep beyond primary benchmark size | Roughly 90 s/experiment and overnight guidance in repo | Full-model profiling and accepted-only iterative kernel work | One incumbent and primary-size objective; no specialists | [repository](https://github.com/RightNow-AI/autokernel/tree/78435821cc3d5756ba6ee1785c397f6d8fa8c90d) |
| KernelBlaster | Compile and executable comparison in task harness | NCU elapsed cycles/bottleneck metrics | No held-out shapes | Default workflow iterations/rollout steps; paper-level budget | Profiler-grounded repeated CUDA optimization | Fixed forward task; rollout/replay method, no shape distribution | [repository](https://github.com/NVlabs/KernelBlaster/tree/84237f91a391971e566cd9066bfb7e9514e957ee) |
| TritonForge | KernelBench-derived correctness/evaluation | Optional PyTorch profiler scripts | Benchmark splits/model evaluation | Turn count and RL training budget | Multi-turn Triton/CUDA repair | Learned generator rather than online evolutionary population; fixed shapes | [repository](https://github.com/RLsys-Foundation/TritonForge/tree/b61331ad2c6fd0c6b315b4270621474ff7120d6b) |
| KernelSkill / KernelMem | Configurable tolerance against KernelBench reference | NCU/NSYS | No unseen-shape protocol in main loop | Paper/model-call experiments | Persistent optimization memory and accepted best kernel | Fixed-shape forward tasks; no full-step/saved-state or dispatch | [repository](https://github.com/0satan0/KernelMem/tree/8b57ccc9adc2ae2f11fc487fd458a7ecc1ea014d), [paper](https://arxiv.org/abs/2603.10085) |
| POLCA | Task evaluator; incorrect kernels receive poor score | Kernel experiment uses runtime, not a profiler-centric loop | No shape holdout in linked matmul config | Linked run: 11 steps, 5 candidates, 1 proposal | Multi-candidate LLM search with diversity | Diversity is code-embedding distance, not per-shape performance | [kernel benchmark](https://github.com/rlx-lab/POLCA/blob/356d0177d034df8c70bf351f5f62c93dbb226b41/benchmarks/kernelbench.md), [priority search](https://github.com/xuanfeiren/Trace/blob/4ee52f12f0bb328dfee2f16b6c1801232c6ccf46/opto/optimizers) |
| CuTeGen | Compile then numerical test | NCU begins only at deeper search nodes | Benchmark evaluation, no search/train shape split | Depth 10, generation/repair retry limits in code | Structured kernel repair and profiling | CuTe, single chain, fixed task, no dispatcher | [repository](https://github.com/taratt/cutegen/tree/0ebe185b9f9d50cf8720878695ce937c2853caae), [paper](https://arxiv.org/abs/2604.01489) |
| SpecGen | Parallel validation/profiling | Yes | Paper task split | Fixed time/token comparisons and H200 setup | Reduces wall time of candidate exploration | Scheduling innovation, not shape-specialist retention | [paper](https://arxiv.org/abs/2606.17518) |
| KernelFoundry | Kernel benchmark correctness | Distributed hardware evaluation | Robust benchmark variants | Paper reports distributed search budget | Genuine quality-diversity kernel search | Public niches are behavioral, not shape regimes; no dispatcher | [paper](https://arxiv.org/abs/2603.12440) |
| Autocomp | Task-defined correctness constraints | Real hardware measurements | Benchmark/test-defined | Beam/iteration budget in configs/paper | Beam search and candidate crossover are relevant alternatives | Cross-accelerator forward programs, no autograd-pair memory/router | [search](https://github.com/ucb-bar/autocomp/tree/a56ce8154c6992648348517489ff5db3d8267798/autocomp/search), [paper](https://arxiv.org/abs/2505.18574) |
| KernelGYM | Isolated random-trial correctness and tolerance checks | Profiling backend support | External benchmark split | Distributed worker/rollout accounting | Evaluation substrate Evograd experiments could target | Environment, not a specialist optimizer | [repository](https://github.com/hkust-nlp/KernelGYM/tree/3a84417f8c0efaadb215ef638b37d12e71ed20f3) |
| AVO | Paper evaluator | Hardware execution feedback | Not enough public implementation detail | Paper reports seven-day B200 experiment | Evolutionary agent as variation operator | Attention-focused paper-only system; shape routing unverified | [paper](https://arxiv.org/abs/2603.24517) |
| KernelPro | Task correctness gates | NCU/SASS/NSYS micro-profiling | Paper benchmark setup | MCTS/search budget in paper | Rich search tree and low-level evidence | No per-shape archive or dispatcher reported | [paper](https://arxiv.org/abs/2606.26453) |

## 3. Detailed system notes

### 3.1 OpenEvolve

OpenEvolve is the search engine Evograd currently calls. Its
[`DatabaseConfig`](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/openevolve/config.py)
defaults to a population of 1,000, archive of 100, five islands,
`exploration_ratio=0.2`, `exploitation_ratio=0.7`, and migration every 50
generations at rate 0.1. Evograd overrides the main sizes to 30/20/two islands.

The database contains three related but distinct diversity mechanisms:

- a set of programs for each island;
- a feature map for each island, where each discretized cell owns one program;
- a global archive of the highest scalar-fitness programs.

Cell replacement compares scalar fitness. The global archive is also sorted by
scalar fitness; it is not a Pareto frontier despite loose “Pareto” wording in
some top-level descriptions. The exact fitness helper first uses
`combined_score`; if that key is absent, it averages numeric metrics after
excluding configured feature dimensions. Consequently, returning per-shape
metrics does not by itself preserve specialists: Evograd would need either
shape-derived MAP coordinates or a different database/selection policy.

Parent selection is island-local. A random draw chooses exploration from the
island, exploitation from the archive/elites, or fitness-weighted selection.
Prompt inspirations may include island best/top programs, nearby feature-map
cells, and random same-island programs. The LLM can receive multiple programs
as context, but this is not a syntax-aware genetic crossover operator.

The evaluator API accepts arbitrary numeric metrics plus larger textual
artifacts. This is exactly where Evograd implements the autograd oracle,
per-shape timing, memory measurement, and scalar objective. OpenEvolve provides
parallel/asynchronous orchestration, checkpointing, islands, migration, LLM
mutation, and persistence. It does **not** provide kernel semantics,
shape-regime islands, per-input Pareto retention, runtime dispatch, or
joint kernel/router evolution.

### 3.2 ShinkaEvolve

ShinkaEvolve stores candidates and genealogy in SQLite and supports a richer
menu of search policies than OpenEvolve. Its database modules implement
weighted, power-law, beam, best-of-N, sequential, and bandit-like parent/model
selection. It supports full rewrites, diffs, cross patches, explicit repair of
incorrect programs, optional system-prompt evolution, and asynchronous or
Slurm-backed evaluation.

Its archive criteria are weighted and normalized, with `combined_score` as the
default criterion. An embedding-crowding option promotes source diversity.
Neither mechanism is an objective-space Pareto frontier. Dynamic islands react
to stagnation and maintain independent populations; their IDs do not encode
small/medium/large shapes. A user could run separate shape evaluators or add
shape-aware archive logic, but that is external design work just as it is for
OpenEvolve.

ShinkaEvolve is therefore a credible alternate *search engine* for Evograd,
especially if dynamic islands, repair mode, or model routing matter. It is not
an existing implementation of shape-specialist deployment.

### 3.3 GEPA

GEPA is the clearest precedent for preserving a candidate because it is good on
a subset of inputs. The API supports four frontier types:

- `instance`: programs best on each validation example;
- `objective`: programs best on each named objective;
- `hybrid`: both sets;
- `cartesian`: programs best on each example/objective cell.

The engine records per-example scores, outputs, trajectories, and optional
objective scores. Aggregate validation fitness is an arithmetic mean. A
candidate selector samples “dominator” programs in proportion to their
frontier coverage, so a program that owns several examples is more likely to
be mutated. Reflection turns execution trajectories and other actionable side
information into a component rewrite. Strict improvement on a sampled
minibatch gates a proposal; an accepted candidate receives broader validation
evaluation. Optional merge combines complementary text components from
descendants with a common ancestor—it is not a runtime composition of their
executables.

Shapes *could* be represented as validation examples and metrics such as
backward speed, full-step speed, and memory as objectives. In that
configuration GEPA’s instance or cartesian frontier would preserve
shape-specialized candidate programs. This is a concrete design transfer from
GEPA, not functionality advertised or evaluated as GPU shape specialization.
GEPA ultimately exposes an aggregate-best candidate and does not infer or emit
a dispatcher. Its explicit `test_set` is evaluated outside the optimization
budget, which is directly useful for held-out-shape methodology.

### 3.4 FunSearch

The public FunSearch repository is a simplified, illustrative implementation,
not DeepMind’s full distributed service. It evolves the body of a marked
function across islands. An evaluator can score a program on several inputs;
the database forms a behavior signature from the sorted per-test scores and
clusters programs with the same signature.

That signature can maintain behavioral diversity, but the public scalar
reducer uses a designated/last test score for ranking and sampling. It does not
perform Pareto dominance across tests. Weaker islands are periodically reset
and reseeded from retained high-quality programs. The mechanism is conceptually
relevant to performance-vector clustering, but it neither assigns shapes to
niches nor emits multiple deployed functions.

### 3.5 AlphaEvolve

AlphaEvolve combines Gemini models, an evolutionary program database, and
multiple automated evaluators to optimize whole codebases. The white paper
reports applications to data-center scheduling, matrix multiplication, and
low-level FlashAttention instructions, including production-relevant kernel
improvements. The public material establishes evolutionary program search and
multi-evaluator feedback.

The implementation is not open source. Public sources do not expose enough
database, selection, per-test retention, or dispatch detail to assert
shape-specific Pareto archives or learned routing. Claims about those
capabilities should remain “unknown,” not inferred from the fact that the
optimized programs support varied inputs.

### 3.6 A-Evolve

A-Evolve’s object of optimization is an *agent workspace*, not a GPU kernel.
It cycles through solving a task, observing performance, editing prompts,
skills, memory, or tools, applying a held-out gate, and reloading or rolling
back the workspace. Its use of the word “evolve” is broader than a conventional
population evolutionary algorithm.

It is relevant to improving Evograd’s optimizer policy or accumulated kernel
knowledge, and its held-out gate is a useful engineering pattern. It is not a
precedent for per-shape kernel populations or dispatch.

### 3.7 GEAK

GEAK’s current v4 repository is a multi-agent AMD optimization workflow. A
director profiles the end-to-end system, attributes cost to operators and
shape/dtype regimes, and delegates high-value work to config tuners, kernel
surgeons, backend specialists, and an integrator. This is not the simple
single-agent Reflexion loop implied by older summaries.

The shape path is unusually concrete:

- production profiles are parsed into shape/dtype distributions;
- cases are assigned weights derived from baseline milliseconds and estimated
  call counts;
- per-shape AITER configurations can be swept and stored;
- regime-specific overlays or exact `(N, K)` guards can select optimized
  implementations;
- unsupported cases fall back to the original implementation;
- warm end-to-end A/B tests and output-parity gates validate integration.

The primary lifecycle speed ratio in
[`attribute_weights.py`](https://github.com/AMD-AGI/GEAK/blob/ab7fa983d4c94c2b3b50426a57ea60e2b30909a8/e2e_workflow/scripts/attribute_weights.py)
is equivalent to `Σ weight / Σ(weight / speedup)`, with weights reflecting
baseline cost and analytic call count. This directly prevents a synthetic
equal-shape average from misrepresenting production benefit, although it
deliberately gives expensive/frequent shapes more influence. Regime floors,
gates, and a secondary geometric mean temper regressions.

GEAK is the closest direct precedent for the **deployed result** Evograd wants:
shape evidence → specialized variants → guarded routing → full-system
validation. The difference is search organization. GEAK retains verified
patches/configurations and expert knowledge, not a per-shape Pareto population,
and its routing tables/guards are generated through existing tuners and agent
integration rather than jointly evolved as genomes.

### 3.8 KernelAgent

KernelAgent has separate generation and optimization paths. Generation
decomposes a graph, deduplicates static subgraph signatures, synthesizes kernels
in parallel, and composes them. That is shape/signature specialization at graph
construction time, but not an objective over a distribution.

Optimization uses a genuine runtime beam. At each round it takes top runtime
kernels, selects bottlenecks, samples model rewrites, checks correctness,
deduplicates PTX fingerprints, and keeps the fastest `N`. NCU metrics feed a
bottleneck analyzer and roofline/SOL reasoning. A divergence threshold can
revert unproductive paths. The program database sorts by `time_ms`; inspiration
sampling mixes top and random entries.

The audited example beam uses width two, two bottlenecks, four workers, 25
warmups, 100 repeats, and a 50% divergence threshold on H100. Those are config
defaults/examples rather than a universal paper budget. KernelAgent is a strong
baseline for “does a beam outperform islands for the same GPU evaluations?”
It does not retain one beam member per shape, optimize saved tensors, or build a
shape router.

### 3.9 Classical autotuning and compiler dispatch

#### Triton autotune

Triton’s official `@triton.autotune` decorator is direct prior art for
shape-dependent specialization at the configuration level. The author supplies
a list of `triton.Config` objects and a list of argument names as the key. When
the key values change, Triton benchmarks all configurations (or a
performance-model/early-pruned subset), chooses the fastest, and can cache
autotuning results to disk.

This achieves levels 1, 3, and 6 of the report’s shape ladder for *configuration
variants*: different exact shape keys retain and use different block sizes,
warp counts, or other constexpr parameters. It does not aggregate a
distribution, discover continuous boundaries, preserve arbitrary source-level
kernel specialists in an evolutionary archive, or jointly optimize a router.
The autotuner assumes configurations are semantically equivalent. Its
`reset_to_zero` and `restore_value` hooks are necessary when benchmarking
mutating kernels because every configuration runs during tuning.

Any Evograd paper should compare its generated specialist portfolio with a
strong single-source Triton kernel that uses `@triton.autotune` over the same
shape keys. Otherwise an apparent gain from separate LLM evolution may simply
be ordinary launch-configuration autotuning.

#### TVM MetaSchedule / Ansor

TVM is the strongest classical precedent for **evolutionary shape-specific
schedule search**. MetaSchedule represents schedule decisions as replayable
TensorIR traces. Its cost-model-guided evolutionary strategy mutates trace
decisions, applies postprocessors for validity/resource rules, predicts
performance, compiles promising schedules, measures them on hardware, and uses
epsilon-greedy exploration.

The tuning database records:

- a serialized TensorIR workload;
- hardware target;
- tensor argument shapes and dtypes;
- the schedule trace and decisions;
- measured runtime and log version.

The database exposes top-K and best-schedule queries. When distinct static
shapes become distinct workloads, they can receive independently evolved
schedule specialists, and compilation queries the matching workload/target
record. This is not the same as retaining shape specialists in one candidate
population: each static workload is a compiler tuning task. The original RFC
describes argument-type fields as useful for future dynamic-shape workloads, so
it would be unsafe to portray MetaSchedule’s database lookup as a learned
runtime boundary for arbitrary unseen dynamic shapes.

The proper distinction is:

- TVM evolves *schedule traces* per compiler workload and selects them during
  compilation;
- Evograd evolves *complete forward/backward source programs* over declared
  workload distributions and emits a runtime Python-level threshold router.

#### TorchInductor max-autotune

PyTorch’s documented `torch.compile(mode="max-autotune")` profiles supported
Triton/template matrix-multiplication choices and Triton convolution choices;
compiler shape guards and caches create input specializations. This is
production evidence that profiling and guarded specialization are normal
compiler techniques. Internal selection/cache behavior changes across PyTorch
versions, so this report relies only on the official stable API claim rather
than inferring a particular candidate database or dynamic-shape router.

TorchInductor is an important non-agent baseline: compare against
`torch.compile` with declared mode/options on the same hardware, and report
compile/autotune time separately from steady-state execution.

### 3.10 Benchmark infrastructure

#### KernelBench

KernelBench is a code-generation benchmark, not an optimizer. The current
repository contains three standard levels (100/100/50 problems) plus an
additional Hugging Face level. A problem normally fixes the concrete dimensions
in `get_inputs`; repeated correctness trials vary tensor values, not dimensions.
Submissions implement `ModelNew` and may use Triton, CUDA, CuTe, TileLang,
ThunderKittens, or HIP depending on evaluator support.

Correctness uses reference execution and dtype-appropriate `allclose`
tolerances. Performance is measured on the task input and commonly reported as
speedup or `fast_p`; aggregate reports can use a geometric mean across correct
tasks. Since each task is a separate program, success on two differently shaped
tasks does not demonstrate one multi-shape kernel or runtime dispatch.

#### KernelBench-X

KernelBench-X has 176 API-oriented tasks in 15 categories. Each task source
usually contains several deterministic test cases, often with different shapes,
dtypes, options, or nested outputs. Correctness recursively checks the entire
output structure, exact shapes and dtypes, and numerical values. Default
float32 tolerances are tight; half/bfloat16 are looser, and tasks can supply
cosine, L1, or RMSE thresholds.

Efficiency times every case and sums runtimes. A task’s speedup is:

`sum(reference_case_ms) / sum(candidate_case_ms)`.

That is a total-time objective, so expensive cases dominate by design. The
summary additionally exposes a global sum ratio and an arithmetic mean of
per-task speedups. Golden performance is hardware-specific; the project
explicitly exposes cross-GPU variability rather than assuming transfer.
KernelBench-X evaluates one submitted implementation per task and has no
specialist portfolio.

#### FlashInfer-Bench

FlashInfer-Bench separates definitions, solutions, workloads, and traces. The
default service timing settings are 10 warmups, 50 iterations, and three trials.
Correctness supports definition-specific `rtol`/`atol` and specialized
validators for sampling, low-bit, and DSA-style operators.

For a trace set, a solution’s score is the arithmetic mean of per-workload
speedups; one failed workload zeros the complete score. Author ranking keeps
the best solution for each definition and averages those definition scores.
This is a generalist score, but the deployment builder performs a second,
specialist operation:

1. form an `ApplyKey` from exact sorted axis values;
2. discard incorrect traces;
3. choose the trace with maximum speedup for each exact key;
4. store `definition → key → solution`;
5. define a `def_best` fallback from the solution with the most key wins;
6. at runtime, dispatch by exact key or use configured fallback behavior.

Frequent winners can be AOT-warmed. The current feature hook uses axes rather
than learned continuous embeddings. Thus this is **post-hoc exact workload
selection**, not inferred boundary discovery and not joint search. It is still
the strongest public implementation reference for Evograd’s eventual
deployment table.

#### Meta PyTorch TritonBench

Meta’s TritonBench is a performance harness for operator/provider pairs. Input
generators can enumerate many shapes; the runner can take first-`k`, equally
spaced, random-`k`, or explicit input IDs. It supports forward, backward,
forward+backward, and forward-without-grad modes, along with latency, speedup,
throughput, profiler, NCU, power, and compilation metrics.

It reports per-input results and arithmetic summaries. It does not generate
kernels, keep candidate populations, or deploy a router. Its chief relevance
is evaluator methodology and ready-made shape distributions.

#### THUNLP TritonBench

THUNLP’s identically named TritonBench is a different project: an LLM Triton
code-generation benchmark. TritonBench-G collects real GitHub kernels;
TritonBench-T pairs PyTorch interfaces with generation tasks; an 8K corpus
supports retrieval. Evaluation stages cover source similarity, call accuracy,
execution accuracy, and efficiency.

The audited G execution path compares serialized program outputs and is less
robust than a declaration-native tensor oracle. Efficiency runs several input
calls and computes a per-op ratio of summed reference time to summed generated
time, then an arithmetic average across operations. G additionally estimates
peak GB/s or TFLOPS. T contains safeguards for suspicious extreme speedups.
Neither suite implements online evolutionary search or dispatch.

### 3.11 Additional kernel optimizers

#### K-Search

K-Search alternates kernel edits with refinement of an intrinsic
decision-tree-like world model. Frontier action nodes represent optimization
decisions and are selected by deterministic value/difficulty/rating logic.
The loop works on one global best, not a population.

Its FlashInfer evaluation can use a fixed or random workload subset; every
selected workload must pass, and the score is an arithmetic mean of workload
speedups. A full workload pass is used at the end. The TriMul setup uses a
geometric mean over seven configurations. These are robust distribution
objectives but do not preserve specialists or route among them.

#### AutoKernel

AutoKernel follows an autoresearch edit-benchmark-keep/revert loop on one
`kernel.py`. It can profile a model, extract bottlenecks, optimize kernels, and
reverify the application. Correctness progresses through smoke, shape sweep,
numerical stability, determinism, and edge cases. It reports latency,
throughput, VRAM, and roofline-like information, but the primary
largest/“biggest” benchmark size normally drives acceptance. Its README budgets
roughly 90 seconds per experiment, about 40 experiments/hour and about 320
overnight, with larger KernelBench runs in the tens to hundreds of experiments.

#### KernelBlaster

KernelBlaster launches parallel optimization rollouts from an initial CUDA
kernel. Within a rollout, changes are sequential. NCU elapsed cycles and
bottleneck knowledge guide rewrites; a persistent strategy database records
historical outcomes and samples relevant strategies. The best step across
rollouts is returned only when it beats the initial program. This resembles
multi-start reinforcement/refinement with replay memory, not an evolving
population of executable specialists.

#### TritonForge

TritonForge trains kernel-generation models with supervised fine-tuning and
reinforcement learning and evaluates multi-turn correction. The reward combines
correctness and performance, and the evaluation stack targets Triton/CUDA on
NVIDIA and AMD. Online inference is a small bounded repair conversation, not
island or Pareto evolution.

#### KernelSkill / KernelMem

KernelSkill uses collaborating agents plus long- and short-term memories of
optimization strategies. Each run maintains current, base, and best code;
material improvement updates the base, while the scalar best is retained.
NCU/NSYS evidence and failed/correct rewrites enter memory. The prompt
explicitly encourages exploiting the fixed benchmark shape, making it a real
specialization system—but one specialist per independent KernelBench task, not
a portfolio over a shared runtime interface.

#### POLCA

POLCA builds on the Trace stochastic generative optimizer. Candidates live in a
priority queue, accumulate mean/UCB-like scores, and become parents for further
proposals. An epsilon-net rejects near-duplicates using L2 distance between
768-dimensional text embeddings. A summarizer transfers lessons across trials.

The linked kernel experiment treats 16 matrix-multiplication tasks as separate
runs. Its example configuration uses 11 steps, five candidates, one proposal
per step, and epsilon 0.02. The diversity criterion is *source embedding
distance*, not shape-performance diversity, and no per-shape dispatcher is
produced.

#### CuTeGen

CuTeGen performs a structured generate-test-refine chain in the CuTe DSL. The
audited implementation searches to depth 10, retries initial generation up to
three times, repairs up to five times, and delays NCU until later nodes. It
normally advances from the most recent passing program while separately
recording the global best. Although its reporting scripts can sweep shapes, the
main optimization target is one fixed KernelBench task.

#### SpecGen

SpecGen is a scheduling/speculation technique. During one expensive reasoning
trajectory it forks cheaper non-reasoning kernel candidates, validates and
profiles them in parallel, stops the parent reasoning when a satisfactory child
arrives, rebalances GPU pools, and can use spare GPU memory for remote KV cache.
It improves search throughput under fixed wall-clock/token budgets. It does not
introduce per-shape candidate retention or routing.

#### KernelFoundry

KernelFoundry is the one additional work found that clearly uses MAP-Elites for
GPU kernels. It evolves kernels in a quality-diversity map, co-evolves
meta-prompts, and tunes structured template parameters on distributed
hardware. The public paper’s behavior dimensions describe implementation
strategy—such as memory access and parallelism—not input-shape cells. No
official code was found in this audit, so finer claims about archive replacement
or deployment cannot be source-verified.

#### Autocomp

Autocomp plans and implements accelerator code with real-hardware feedback. Its
current repository uses a top-N beam/candidate repository and can combine two
parents, making it a useful crossover and beam baseline across CUDA, Pallas,
Metal, and other accelerators. It ranks correct candidates by a scalar task
score and does not expose a shape-specialist router.

#### KernelGYM, AVO, and KernelPro

KernelGYM is reusable distributed evaluation/RL infrastructure with process
isolation, random-trial correctness, timing, and profiling. It is relevant as
an experimental substrate, not as a prescribed evolutionary method.

AVO’s paper describes an evolutionary optimizer in which an agent acts as the
variation operator and consults lineage, a knowledge base, and execution
feedback. It reports long B200 attention experiments. Without public code,
per-configuration retention and dispatch details remain uncertain.

KernelPro’s paper combines multi-agent CuTe/raw-CUDA generation with MCTS,
progressive widening, asymmetric branches, logarithmic rewards, dead-end
pruning, search memory, and NCU/SASS/NSYS evidence. It also reports an energy
result. No per-shape archive or runtime dispatch mechanism is reported.

## 4. Taxonomy

| Category | Systems | Defining retention/search behavior |
|---|---|---|
| Island/population evolution | OpenEvolve, ShinkaEvolve, FunSearch, AlphaEvolve, AVO | Multiple executable programs with lineage, mutation, and population/database selection |
| Pareto or quality-diversity evolution | GEPA, KernelFoundry; OpenEvolve only when user-supplied MAP features are meaningful | Retain candidates by per-instance/objective frontier or behavior-map cell rather than only scalar rank |
| Beam and priority search | KernelAgent, Autocomp, POLCA | Keep top-N or queued candidates; branch from selected parents without island dynamics |
| Single-incumbent/Reflexion-style refinement | AutoKernel, K-Search, CuTeGen, KernelSkill, KernelBlaster | Repeated rewrite/repair around current or best program, sometimes with replay/knowledge memory |
| Profiler-guided kernel agents | GEAK, KernelAgent, KernelBlaster, KernelSkill, KernelPro, Evograd’s optional final NCU pass | NCU/rocprof/NSYS/SASS/roofline evidence informs proposed changes |
| Multi-agent generation/integration | GEAK, KernelSkill, KernelPro, TritonForge | Specialized agents coordinate planning, generation, diagnosis, or integration |
| Learned/offline generators | TritonForge | SFT/RL trains a generator; online candidate loop is not evolutionary |
| Classical schedule/config autotuning | TVM MetaSchedule/Ansor, Triton autotune, TorchInductor max-autotune | Search or profile schedule/config choices for a workload/key; retain compiler/JIT selections |
| Search-throughput infrastructure | SpecGen, KernelGYM | Parallel speculation or distributed evaluation makes another optimizer faster |
| Benchmarks | KernelBench, KernelBench-X, Meta TritonBench, THUNLP TritonBench | Define tasks, inputs, correctness, and performance; do not prescribe population search |
| Deployment selection | FlashInfer-Bench, GEAK, Evograd | Choose among variants at runtime using exact keys, guards, or a measured threshold |

These categories overlap. KernelAgent is both beam search and profiler-guided;
GEAK is both multi-agent and deployment-oriented. The table describes the
search mechanism, not branding.

## 5. Shape-handling analysis

### 5.1 Capability ladder

| System | 1: same candidate, many shapes | 2: aggregate | 3: retain shape specialists | 4: separately evolve specialists | 5: discover boundary | 6: runtime dispatch | 7: joint kernel + router |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenEvolve alone | evaluator-defined | evaluator-defined | no, unless custom MAP/frontier logic | no | no | no | no |
| ShinkaEvolve alone | evaluator-defined | evaluator-defined | no | no | no | no | no |
| GEPA configured with shapes as examples | yes | mean | **yes, by interpretation** | not inherently | no | no | no |
| FunSearch | evaluator-defined | scalar + behavior vector | indirect clusters only | no | no | no | no |
| GEAK | yes | production weighted | configs/verified variants | agent-directed regimes | agent/profile-directed, not a generic learned sweep | yes | no |
| KernelAgent | generally one concrete optimization target | scalar | beam variants for that target | no | no | static-signature composition only | no |
| Triton autotune | each exact key is independently timed | none across keys | best config per key | supplied configs are tuned per key | exact-key partition | yes, cached config choice | no |
| TVM MetaSchedule / Ansor | separately represented static workloads | task scheduler allocates trials; records remain per workload | top schedules per workload/target | yes, separate tuning tasks | workload identity, not learned boundary | compile-time database selection | no |
| TorchInductor max-autotune | guarded compiler specializations | local best per specialization | cached compiler choice | compiler-driven | shape guards, version-dependent | guarded compiled cache | no |
| KernelBench | usually no within task | across tasks | no | tasks are separate, not a portfolio | no | no | no |
| KernelBench-X | yes | summed time | no | no | no | no | no |
| FlashInfer-Bench | yes | mean speedup | yes, post-hoc per exact key | external generator may make many solutions | exact key, no continuous boundary | **yes** | no |
| Meta TritonBench | yes | arithmetic summaries | no | no | no | provider-internal only | no |
| THUNLP TritonBench | yes within operation calls | total-time ratio | no | no | no | no | no |
| K-Search | yes | arithmetic/geometric mean | no | no | no | no | no |
| KernelFoundry | benchmark-dependent | yes | behavior specialists, not shape specialists | niches evolve together | no shape boundary reported | no | no |
| Current Evograd | **yes** | total ratio or geometric mean | separate run winners only | **yes: predefined small/large** | **yes: one measured scalar threshold** | **yes** | no |

### 5.2 Why aggregation choice matters

For candidate case latencies `c_i`, baseline latencies `b_i`, and speedups
`s_i=b_i/c_i`, the common aggregations answer different questions:

- **Total-time ratio:** `Σb_i / Σc_i`. Slow/large/frequent cases dominate. This
  is appropriate for a trace where every listed call occurs once, but a
  synthetic grid silently treats “one row in the grid” as “one production
  call.”
- **Arithmetic mean speedup:** `mean(s_i)`. Every listed case has equal count,
  but extreme speedups can dominate and ratios are asymmetric.
- **Geometric mean:** `exp(mean(log s_i))`. Every case has equal log-weight;
  reciprocal gains/regressions are symmetric. It does not prevent a severe
  single-case regression.
- **Weighted geometric mean:** `exp(Σw_i log(s_i)/Σw_i)`. This makes the case
  distribution explicit; weights must have a documented semantic meaning.
- **Minimum/worst-case gate:** `min(s_i)` or a constraint `s_i ≥ floor`.
  Protects tails but can make noisy microbenchmarks control the search.
- **Pareto/per-case frontier:** does not collapse cases during retention.
  Necessary when the goal is to preserve complementary specialists before a
  dispatcher exists.

Evograd’s full-grid `speed_memory_min` uses total-time ratios for backward and
raw full step, then takes their minimum. Large/slow shapes therefore dominate
both numerator and denominator. Its specialist policy uses a geometric mean and
a soft worst-case multiplier. GEAK instead uses observed call count and baseline
cost to make production dominance intentional. GEPA avoids early scalar
collapse for retention, while FlashInfer-Bench retains the fastest correct
solution independently for each exact workload key.

### 5.3 Specialist survival

There are three materially different ways to keep a locally good candidate:

1. **Code/behavior diversity:** OpenEvolve MAP features, FunSearch signatures,
   POLCA embeddings, and KernelFoundry behavior cells. A candidate survives
   because its code or behavior descriptor is different, not necessarily
   because it wins a shape.
2. **Objective-local superiority:** GEPA instance/cartesian frontiers and
   FlashInfer-Bench’s per-key winners. A candidate survives specifically
   because it wins an input/objective.
3. **Independent search budgets:** Evograd’s small/large runs and GEAK’s
   agent-assigned regimes. A specialist does not compete against the generalist
   during its search, but regime definitions consume prior knowledge and total
   search cost rises.

For Evograd’s research question, method 2 is the missing evolutionary control.
A shape-indexed frontier would reveal whether independent specialist runs are
needed or whether one population naturally contains complementary kernels.

### 5.4 Boundary and dispatch mechanisms

The verified deployment mechanisms are:

- **FlashInfer-Bench:** exact workload axes → fastest correct submitted
  solution; miss policy falls back or uses a definition-wide winner.
- **GEAK:** existing AITER configuration tables and integrator-written exact
  shape/regime guards with fallback.
- **Evograd:** one declared scalar feature; benchmark adjacent observed feature
  values; try geometric midpoints plus all-small/all-large sentinels; maximize
  component-derived raw-full-step geometric mean; emit one threshold.

None of those co-evolves a routing tree with kernel source. FlashInfer’s exact
table has no interpolation; GEAK’s agent reasoning is not a generic boundary
learner; Evograd’s one-threshold model cannot express islands such as “small and
very large use A, medium uses B,” interactions among multiple dimensions, or a
generalist fallback in only a subregion.

### 5.5 Benchmark audit

| Benchmark | Task/shape structure | Dtypes and randomness | Correctness | Performance/aggregation | Shape dominance, specialization, transfer |
|---|---|---|---|---|---|
| KernelBench | Standard task usually defines one concrete `get_inputs` shape; tasks are separate | Task/evaluator precision; default seed 42; deterministic derived seeds for repeated random-value trials | All trials must pass reference comparison; tolerance chosen from precision/task settings | Per-task runtime/speedup; aggregate `fast_p` and geometric means over correct tasks | No within-task shape distribution. A manually branching submission is allowed but not rewarded as a portfolio. Hardware must be reported; transfer is not assumed |
| KernelBench-X | 176 tasks/15 categories; usually 3–4 or more explicit cases per file, often different shapes/options | Task-specific float/int/half/etc.; evaluator resets Python/Torch/CUDA seed, default `KERNELBENCHX_SEED=0` | Recursive type/container/value checks, exact tensor shape/dtype; f32/f64 `1e-5`, half/bf16 `5e-3`, or task precision thresholds | Task speedup `Σref_ms/Σgen_ms`; total GB/s/TFLOPS; global sum ratio and per-task arithmetic mean | Slow cases dominate task score. No specialist archive/dispatcher. Golden results are GPU-specific, exposing cross-hardware variance |
| FlashInfer-Bench | Definition × solution × workload trace; workload axes represent production-like shapes/configs | Definition/workload-specific dtypes; deterministic serialized workloads; timing defaults 10/50/3 | Configurable tolerances and specialized validators; any failed trace zeros a solution’s trace-set score | Mean per-workload speedup; per-definition best then mean across definitions | Generalist score equal-weights workloads, but apply table independently selects each exact-key winner. Exact-key deployment supports specialists and fallback; transfer requires retiming on target GPU |
| Meta PyTorch TritonBench | Operator input generator can yield arbitrary-length shape grid; selectable explicit/first/equal/random input IDs | Operator-defined; random-`k` can take a reproducibility seed | Provider output/gradients compared with chosen baseline using configured accuracy metrics | Per-input latency/speedup/throughput; arithmetic summary; fwd/bwd/fwd+bwd modes and many profiler metrics | Harness evaluates one provider across shapes. No specialist retention or dispatch; results are target-hardware measurements |
| THUNLP TritonBench | G real-kernel calls and T PyTorch-interface cases; multiple calls/shapes per operation | Dataset-defined; no uniform cross-suite dtype/seed policy | Staged call and execution accuracy; audited G path compares serialized outputs, T has more task logic | Sum-reference/sum-generated time per op, arithmetic mean across ops; G peak GB/s/TFLOPS | Slow calls dominate per-op ratio. One generated file per operation, no portfolio/router. Hardware affects reported efficiency |

No benchmark in this table supplies all of the experiment Evograd needs:
hidden shape split, multiple candidate implementations, per-shape retention,
router training, router timing, and equal-budget universal-vs-portfolio
comparison. Those must be layered on top.

### 5.6 Held-out shapes

Held-out *values* are not held-out *shapes*. KernelBench’s multiple randomized
correctness trials test value robustness at the same dimensions. A defensible
shape-generalization protocol should:

- partition by feature intervals, not randomly split near-duplicate grid rows;
- include interpolation and extrapolation sets;
- keep dtypes represented in both search and test, plus an explicit
  cross-dtype extrapolation test if claimed;
- select dispatch boundaries only on the search/validation grid;
- measure the frozen router and kernels on hidden shapes;
- rerun the entire comparison per GPU architecture rather than reuse one
  architecture’s threshold.

GEPA’s separate test set is the clearest software pattern for enforcing this
boundary. FlashInfer-Bench supplies the trace abstraction needed to replay a
frozen portfolio.

### 5.7 Forward, backward, full-step, memory, and energy scope

| System/family | Execution scope | Memory or energy objective |
|---|---|---|
| OpenEvolve, ShinkaEvolve, GEPA, FunSearch | Completely evaluator-defined | Possible through a custom evaluator; no built-in GPU saved-state model |
| AlphaEvolve | Varied algorithms and reported low-level kernel/codebase tasks | Public sources do not establish a standard kernel-memory objective |
| A-Evolve | End task/agent workspace | Task-defined, not kernel-specific |
| GEAK | Operator microbenchmarks plus complete serving E2E | Memory feasibility and system behavior are checked; primary lifecycle objective is time |
| KernelAgent | Generated subgraphs and concrete kernel optimization | SOL/resource metrics inform diagnosis; runtime is the retained scalar |
| Triton autotune | One JIT kernel under several launch/meta-parameter configurations | Latency chooses a config; no explicit memory objective |
| TVM MetaSchedule / Ansor | TensorIR operator schedule for a workload/target | Runtime is primary; resource validity and cost-model features guide search |
| TorchInductor max-autotune | Compiled graph/operator implementations | Profiles fastest supported choices; compiler options expose memory/performance tradeoffs |
| KernelBench | `ModelNew.forward` task boundary, which may contain fused operations | Runtime correctness/performance; no saved-tensor objective |
| KernelBench-X | API/forward implementation across test cases | Runtime/throughput/bandwidth; no autograd saved state |
| FlashInfer-Bench | Inference kernels | Correctness/latency; no training saved-state objective |
| Meta TritonBench | Explicit `fwd`, `bwd`, `fwd_bwd`, and `fwd_no_grad` modes | Can measure memory/power/profiler metrics, but is not a search objective by default |
| THUNLP TritonBench | Dataset-defined Triton functions, including some backward kernels | Efficiency/throughput, not a systematic fused training-step memory objective |
| AutoKernel | Kernel and full-model execution | Reports VRAM/resource use; audited keep/revert objective is primarily performance |
| KernelBlaster, CuTeGen, KernelSkill, K-Search, TritonForge | Usually forward/fixed KernelBench or inference task | Resource metrics may guide prompts, but retained fitness is correctness/performance |
| KernelPro | Kernel task | Paper explicitly evaluates an energy objective/result in addition to speed |
| Evograd | Generated forward-with-saved + backward-from-saved; backward-only and two full-step paths | Explicit saved-tensor bytes and saved/input ratio enter the scalar score |

Within the inspected public implementations, Evograd’s exact combination of a
generated saved-state contract, backward-from-saved timing, raw full-step
timing, and saved-tensor penalty was not found elsewhere. That is a narrow
implementation comparison, not a claim that memory-aware training-kernel
optimization is new; the generic evolutionary frameworks can accept such an
evaluator, Meta TritonBench can measure backward modes, and GEAK optimizes
end-to-end systems where memory constraints matter.

## 6. Direct comparison with Evograd

### 6.1 What comes from OpenEvolve

Evograd inherits from OpenEvolve:

- LLM diff/full-program mutation;
- island populations and migration;
- scalar-fitness parent and elite selection;
- inspiration sampling;
- generic MAP feature maps;
- asynchronous/process evaluation orchestration;
- checkpoints, program genealogy, artifacts, and best-program return.

OpenEvolve sees only the `combined_score` and auxiliary metrics Evograd returns.
It does not know that a metric came from a shape, gradient, saved tensor, or
full training step.

### 6.2 What Evograd adds in the evaluator

The declaration-native evaluator adds:

- deterministic construction of declared correctness and benchmark inputs;
- a `torch.autograd.grad` oracle over exactly the `Active` arguments;
- forward and per-gradient numerical, shape, and dtype checks;
- a hard correctness gate and killable subprocess timeout;
- one untimed smoke invocation on every benchmark shape;
- median CUDA-event timing for forward, backward-from-saved, raw forward plus
  backward, autograd-bound full step, and the selected baseline;
- saved tensor byte accounting and saved/input memory ratio;
- five scalar policies spanning backward speed, backward/full-step balance,
  geometric means, worst-case protection, and a saved-memory penalty.

The exact current aggregates are:

```text
B = sum(baseline_backward_ms) / sum(candidate_backward_ms)
F = sum(baseline_full_ms) / sum(candidate_raw_full_ms)
M = sum(saved_bytes) / sum(memory_input_bytes)

speed_memory_min = min(B, F) / (1 + 0.05 M)
```

For the nominal weighted-geometric policy:

```text
G = weighted_geomean_i(min(B_i, F_i))
W = min(1, min_i(min(B_i, F_i)))
score = G * W / (1 + 0.05 M)
```

The evaluator is therefore multi-metric in measurement but scalar in
OpenEvolve selection.

### 6.3 What Evograd adds as a specialization strategy

For the 12 regime-enabled declarations, Evograd creates three independent
OpenEvolve jobs from the same seed:

- `full`: all declared benchmark cases, scored by total-time
  `speed_memory_min`;
- `small`: cases below a declaration-provided scalar split;
- `large`: cases at or above that split.

The small/large searches use the weighted-geometric policy name and a
worst-case guard. They do not share population members, migrate candidates, or
cross over across regimes. The initial split is expert-declared, not learned.
With three GPUs the jobs can run concurrently; with one GPU they run
sequentially.

This is **separate specialist evolution**, not specialist retention inside one
population. It spends approximately three times the evolutionary evaluations
of a one-run generalist before deployment remeasurement, so search-budget
normalization is mandatory in any scientific comparison.

### 6.4 What Evograd adds at deployment

Evograd re-evaluates every returned program on the full benchmark suite and
chooses the best full/small/large tag within each prefix by raw-full-step
geometric mean. For the best small and large program it evaluates thresholds at
geometric midpoints between observed positive feature values, plus all-small
and all-large endpoints. It emits either one winning program or a Python
dispatcher. The forward route is saved as a plain integer beside the candidate
saved state, ensuring backward follows the same implementation without adding
tensor bytes.

This is data-derived threshold selection and deployable routing. It is not
joint evolution: threshold search occurs after all kernel searches are
finished, cannot influence which candidates survive OpenEvolve, and is limited
to two arms and one scalar feature.

### 6.5 Current gaps discovered in the code audit

The following are present at Evograd commit `d21dbe1` and should be fixed or
made explicit before claiming a completed shape-specialization study:

1. **Declared weights are disconnected.** `OpDecl.workload_weight()` exists,
   but the automatic `_evolve_group` path never serializes selected weights to
   `EVOGRAD_GEOMEAN_WEIGHTS`. `weighted_geomean(..., None)` becomes a uniform
   geometric mean.
2. **Benchmark shapes are smoke-tested, not numerically verified.** Full
   forward/gradient correctness runs only on `op.correctness`. Benchmark cases
   receive an untimed execute/synchronize smoke pass before timing, so a
   shape-conditional wrong answer outside the correctness set can be scored.
3. **The full generalist cannot win routing selection.** It is measured and
   reported, but threshold scoring considers only the chosen small and large
   programs. Even when the full program has a better full-grid geomean, the
   generated final artifact follows the small/large decision.
4. **The emitted dispatcher is not benchmarked end to end.** The reported
   dispatch score combines prior component timings. Python routing, module
   lookup, branch, and any compilation/cache behavior are excluded.
5. **Dispatch optimizes a narrower objective than evolution.** Boundary
   selection uses raw-full-step speedup only. It ignores backward-only speed,
   saved memory, declared case weights, and the evaluator’s worst-case factor.
6. **One-dimensional, two-arm boundary.** There is no medium specialist,
   non-monotone assignment, multi-dimensional tree/table, uncertainty band, or
   generalist fallback arm.
7. **No shape-local population retention.** OpenEvolve’s MAP features remain
   its defaults; per-shape metrics are artifacts/case reports rather than
   archive dimensions or Pareto objectives.
8. **NCU refinement is not regime-local.** Each group’s optional final NCU pass
   chooses the largest representative from the declaration’s default benchmark
   rather than the selected group suite, and acceptance uses the generic full
   evaluator. It is also one post-search rewrite, not feedback during
   population evolution.
9. **No hidden-shape protocol.** Correctness, benchmark, threshold fitting, and
   deployment reporting are declaration-defined but not split into
   search/validation/test shape sets.
10. **CUDA status is unverified in this checkout.** The repository reports
    structural/unit validation but states that its Triton paths have not been
    run on a CUDA machine in this environment. All runtime conclusions above
    are source-level capabilities until the GPU checklist passes.

Relevant Evograd sources:

- [evaluator and correctness gate](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/evolve/evaluator.py)
- [benchmark aggregates](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/bench/harness.py)
- [scoring formulas](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/evolve/scoring.py)
- [generalist/specialist orchestration](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/api.py)
- [threshold search and generated dispatcher](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/dispatch.py)
- [declaration regime metadata](https://github.com/akumdam2/evograd/blob/d21dbe10a47ecb31e19ce6af9b18a43ca91a32df/src/evograd/opdecl/activity.py)

### 6.6 Closest comparisons

There is no single nearest neighbor on every axis:

- Compare **search mechanics** to OpenEvolve and ShinkaEvolve.
- Compare **per-input retention** to GEPA.
- Compare **quality-diversity kernel evolution** to KernelFoundry.
- Compare **evolutionary per-workload schedules** to TVM MetaSchedule/Ansor.
- Compare **shape-keyed launch/config specialization** to Triton autotune.
- Compare **production compiler autotuning** to TorchInductor max-autotune.
- Compare **hardware-guided candidate search** to KernelAgent.
- Compare **production-weighted regimes and guarded integration** to GEAK.
- Compare **per-workload portfolio deployment** to FlashInfer-Bench.
- Compare **multi-shape total-time evaluation** to KernelBench-X.
- Compare **backward/fwd+bwd harness design** to Meta TritonBench.

The uncertain area is unpublished infrastructure: AlphaEvolve and paper-only
systems may have internal workload-specialization behavior not described
publicly. The report should say “not established in public sources,” not “does
not exist.”

## 7. Additional repositories and papers found

| Work | Relevance | Related-work priority |
|---|---|---|
| Triton autotune | Exact-key timing and retained launch/meta-parameter configuration for one kernel | **Mandatory baseline/prior art** for shape specialization |
| TVM MetaSchedule / Ansor | Cost-model-guided evolutionary schedule search and workload/target database | **Primary related work** for evolutionary static-workload specialists |
| TorchInductor max-autotune | Profile-selected compiler algorithms/configs under guarded specialization | **Mandatory production compiler baseline** where supported |
| K-Search | Co-evolving intrinsic world model; robust all-workload kernel evaluation | **Primary adjacent baseline** for iterative multi-workload search |
| KernelFoundry | MAP-Elites quality-diversity search directly for GPU kernels | **Primary related work** for population diversity, though niches are not shapes |
| AutoKernel | Full-model bottleneck extraction and accepted-only autoresearch loop | Primary engineering comparison; not a shape-specialist method |
| KernelBlaster | NCU-guided parallel rollouts and persistent strategy replay | Primary profiler-agent comparison |
| TritonForge | SFT/RL multi-turn Triton/CUDA generator on NVIDIA/AMD | Primary learned-generator comparison, not online evolution |
| KernelSkill / KernelMem | Multi-agent persistent optimization memory and explicit fixed-shape specialization | **Primary related work** for specialist knowledge transfer |
| POLCA / Trace | Priority search with code-embedding epsilon-net diversity | Primary search baseline; embedding diversity must not be mislabeled shape diversity |
| CuTeGen | Structured CuTe generate-test-refine with delayed NCU | Primary CuTe agent comparison |
| SpecGen | Parallel speculative kernel candidates under fixed time/token budget | Adjacent infrastructure; relevant to search-cost experiments |
| Autocomp | Real-hardware beam search and optional candidate combination across accelerators | Primary beam/crossover and cross-hardware comparison |
| KernelGYM | Distributed, isolated kernel evaluation/RL environment | Adjacent experimental infrastructure |
| AVO | Agent-as-variation evolutionary kernel optimizer | Potentially primary, but paper-only implementation limits reproducibility |
| KernelPro | Multi-agent MCTS with low-level profiling and an energy objective | Primary search/profiling comparison; paper-only |

“AutoKernel” is overloaded in older compiler literature. This report refers to
RightNow-AI’s agentic repository at the pinned commit, not unrelated scheduling
systems with the same name.

## 8. Suggested experiments

### Priority 0: establish a trustworthy measurement boundary

1. **Run numerical correctness on every timed shape.** Keep the small
   correctness suite for fast rejection, but require the frozen candidate and
   final dispatcher to match the oracle across the complete benchmark grid
   before performance is accepted.
2. **Time the emitted artifact.** Benchmark the actual routed forward/backward
   pair, including routing overhead and saved route state. Report predicted
   component score and measured deployment score side by side.
3. **Wire and log case weights.** Materialize declaration weights in every run
   artifact and assert their length/order against selected cases. Publish both
   uniform and production-weighted results.
4. **Make the generalist an arm.** Optimize routing over `{generalist, small,
   large}` and permit a no-dispatch result. Never deploy a portfolio that loses
   to the best universal kernel after measured overhead.
5. **Add non-agent baselines.** Run a single-source `@triton.autotune` kernel,
   TVM/Ansor where the operator can be represented fairly, and
   `torch.compile(mode="max-autotune")` where supported. Separate their tuning
   cost from steady-state timing.

### Experiment 1: equal-budget universal versus specialists

Use a fixed total number of evaluated candidates or GPU-seconds:

- universal: one run with budget `3B`;
- specialists: three runs with budget `B` each;
- shared population: one run with budget `3B` and per-shape retention.

Repeat seeds and report search-cost confidence intervals, final full-grid
backward/full-step speed, memory, and router overhead. Also report an
“oracle-per-shape” upper bound that selects the fastest correct candidate for
each shape; this quantifies the maximum available specialization value before
choosing a router model.

### Experiment 2: GEPA-style per-shape frontier inside kernel evolution

Store the full vector:

```text
(backward_speedup_i, full_step_speedup_i, saved_ratio_i, correct_i)
```

for every candidate and shape. Retain:

- the aggregate-best candidate;
- top candidates for each shape;
- candidates on a backward/full-step/memory frontier;
- optionally candidates for each `(shape, objective)` cell.

Select parents proportional to the number or production weight of cells they
own, analogous to GEPA’s frontier coverage. Compare this with OpenEvolve’s
default scalar archive and with source-diversity MAP features. The key outcome
is not only final speed; measure how often complementary candidates survive
long enough to improve or enter the final router.

### Experiment 3: shape-aligned islands

Compare:

- generic OpenEvolve islands with migration;
- fixed small/medium/large islands;
- dynamic clustering of shapes by candidate performance vectors;
- no migration, low migration, and periodic migration;
- shared elites versus regime-local elites.

A shape-aligned island should score its local distribution but periodically
receive full-grid correctness and regression checks. Migration can be triggered
by cross-regime utility: move a candidate when it wins cases outside its home
regime, not merely every fixed number of generations.

### Experiment 4: routing models

Fit and validate increasingly expressive routers on the same candidate pool:

1. one scalar threshold;
2. two thresholds for small/medium/large;
3. exact-key table with fallback, matching FlashInfer-Bench;
4. shallow decision tree over symbolic dimensions, dtype, divisibility, and
   layout;
5. cost-sensitive tree whose leaf objective includes measured routing overhead;
6. jointly mutated router plus kernel references.

Use nested validation: kernel search shapes, router-fit shapes, and a hidden
test set. Penalize program size, compile latency, and number of resident
variants. Enforce a generalist fallback for unseen keys.

### Experiment 5: crossover detection and stability

For each candidate pair:

- collect repeated latency samples per shape;
- bootstrap the probability that A beats B;
- fit boundaries only where the win probability exceeds a confidence threshold;
- introduce a hysteresis/uncertainty band that routes to the generalist;
- repeat on different clocks, process launches, and days.

Report whether inferred boundaries survive noise and whether geometric
midpoints are better than direct empirical cut points. A speed curve crossing
once on the observed grid is not evidence of global monotonicity.

### Experiment 6: objective ablation

Compare, with identical candidate evaluations:

- total-time ratio;
- arithmetic mean speedup;
- uniform geometric mean;
- production-weighted geometric mean;
- GEAK-style baseline-time × call-count lifecycle score;
- worst-case hard floor versus Evograd’s soft multiplier;
- scalar score versus per-shape frontier.

Report per-shape curves, not only aggregate scores. Include backward-only,
raw full-step, autograd-bound full-step, saved bytes, compile time, and
dispatcher overhead. This will show precisely which shapes each objective
sacrifices.

### Experiment 7: held-out and adversarial shapes

Create:

- interpolation shapes between training grid points;
- extrapolation just outside each regime;
- awkward prime/non-power-of-two dimensions;
- alignment/divisibility transitions;
- dtype transitions;
- minimal/degenerate legal cases;
- memory-pressure cases near device limits.

Freeze kernels and router before evaluation. Track correctness, regression
rate, fallback rate, and performance calibration. Compare exact-key fallback
with threshold/tree generalization.

### Experiment 8: cross-hardware specialization

Repeat complete search and routing on at least two materially different GPU
architectures. Evaluate four deployments:

- kernel/router searched and tested on hardware A;
- A artifacts tested unchanged on B;
- A kernels with router refit on B;
- full B search.

This separates kernel transfer from router transfer. A hardware ID can become a
dispatch feature only if the deployment package carries separately verified
artifacts and does not silently use A’s timing table on B.

### Experiment 9: search-cost versus deployment benefit

For every method, record:

- LLM calls, input/output tokens, and dollar cost;
- compile attempts and correctness executions;
- timed GPU evaluations and GPU-seconds;
- profiler invocations and NCU replay overhead;
- wall-clock time and parallel hardware count;
- final code size, compile/startup time, and number of variants;
- full-step milliseconds saved per representative training step.

Plot deployment speedup and saved training time against search cost. Include a
break-even estimate:

`search_cost_seconds / seconds_saved_per_training_step`.

This prevents a three-specialist search from appearing superior merely because
it received three times the GPU/LLM budget.

## Bottom line

Direct precedents exist for every *piece* of shape-specialized kernel
optimization:

- per-instance frontier retention (GEPA);
- quality-diversity kernel evolution (KernelFoundry);
- evolutionary per-workload schedule tuning (TVM MetaSchedule/Ansor);
- exact-key launch/config selection (Triton autotune);
- production-weighted shape regimes and guarded deployment (GEAK);
- exact per-workload winner selection and runtime lookup (FlashInfer-Bench);
- multi-shape total-time benchmarking (KernelBench-X);
- backward and forward+backward performance modes (Meta TritonBench).

The research opportunity is to combine those pieces rigorously around
forward/backward Triton pairs, with hidden-shape validation and equal search
budgets. The current Evograd implementation is a useful first composition, but
its disconnected weights, benchmark-only smoke checks, two-arm post-hoc router,
and lack of generalist competition or measured routing overhead should be fixed
before making strong empirical or novelty claims.
