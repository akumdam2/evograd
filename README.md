# evograd

**Evolved backward kernels.** Given a forward reference for an operator,
evograd generates a training-ready autograd-pair seed (forward + backward
Triton kernels) through one of several pipelines, optimizes it with
[OpenEvolve](https://github.com/codelion/openevolve), and benchmarks the
result against the PyTorch-autograd baseline.

Successor to the backward-benchmark work in the `openevolve` fork
(`atenir/`, `pipeline/`, `benchmark/`); OpenEvolve itself is consumed as a
package dependency, not forked.

## The core abstraction: activity-annotated operator declarations

Every operator is declared once, with Enzyme-style activity annotations:

```python
op = declare_op(
    name="evoattention",
    forward="evograd.ops.evoattention.forward_ref:evoattention_forward_ref",
    dims=("B", "S", "H", "N", "D"),
    args=(
        Duplicated("q",         "[B, S, N, H, D]"),   # active: gets a gradient
        Duplicated("k",         "[B, S, N, H, D]"),
        Duplicated("v",         "[B, S, N, H, D]"),
        Const("res_mask",       "[B, S, 1, 1, N]"),   # inactive: no gradient
        Duplicated("pair_bias", "[B, 1, H, N, N]", grad="d_pair_bias"),
    ),
    output=Duplicated("o", "[B, S, N, H, D]"),        # its shadow is the upstream grad
    correctness=(...), benchmark=(...), tolerances={...},
)
```

`Duplicated` marks an active tensor (primal + gradient "shadow"); `Const`
marks an inactive one. The declaration is the single source of truth —
everything else derives from it:

| Consumer | Derivation | Replaced |
|---|---|---|
| Pipeline A/C prompts | rendered from the typed contract | `<op>_spec.json` files |
| Pipeline B wrapper codegen | `Duplicated` positions → grad selection | `grad_reorder()`, `"d"+name` matching |
| Ground-truth oracle | `torch.autograd.grad`, `requires_grad` on exactly the `Duplicated` args | `backward_ref.py` × 6 |
| Training wrapper | `opdecl.bind` builds the `autograd.Function`; `Const` → `None` slots | `autograd_wrapper.py` × 6 |
| Verifier | per-`Duplicated` comparison at declared workloads/tolerances | `test_correctness.py`, evaluator correctness halves |
| OpenEvolve evaluator | one generic evaluator × scoring policy | per-bench evaluator families (7 files for layernorm) |
| Extraction inputs | `example_input_spec(op)` | hand-typed `--example-input` strings |

## Usage

```bash
pip install -e ".[gpu,llm]"

evograd ops                                              # list the 6 declared operators

# Seed generation (GPU node; a/c need an LLM API, b is LLM-free)
evograd seed b --op rmsnorm --output-dir /tmp/B_rmsnorm --dtype float32 --dtype float16
evograd seed a --op rmsnorm --output-dir /tmp/A_rmsnorm --model gpt-5.5
evograd seed c --op rmsnorm --output-dir /tmp/C_rmsnorm --model gpt-5.5

# Verify any candidate against the autograd oracle
evograd verify --op rmsnorm /tmp/B_rmsnorm/initial_program_autograd_pair.py

# Optimize with OpenEvolve (scoring: speed | speed_memory | speed_memory_min
#                                    | speed_memory_min_geomean | ..._weighted_geomean)
evograd evolve --op rmsnorm --seed /tmp/B_rmsnorm/initial_program_autograd_pair.py \
    --scoring speed_memory --iterations 10 --output-dir /tmp/evolve_rmsnorm

# Benchmark a candidate against the PyTorch-autograd baseline
evograd bench --op rmsnorm --candidate /tmp/evolve_rmsnorm/evolved_best_program.py

# LayerNorm includes every legacy/TritonBench suite and the optional Liger baseline
evograd bench --op layernorm --candidate best.py --suite tb_mixed --dtype float16
pip install -e ".[baselines]"
evograd bench --op layernorm --candidate best.py --baseline liger
```

Adding operator #7 = one `src/evograd/ops/<name>/` package containing an
`__init__.py` declaration and `forward_ref.py`. Related input builders,
baselines, or operator-specific helpers stay in that package. The registry
discovers operator packages automatically; prompts, codegen, oracle, verifier,
evaluator, and bench all follow without editing a central list.

## Layout

```text
src/evograd/
├── opdecl/        # declare_op/Duplicated/Const; oracle, bind, verify, inputs, verify_cli
├── ops/           # one self-contained package per operator
│   ├── layernorm/
│   │   ├── __init__.py       # declaration, workloads, input builder, baselines
│   │   └── forward_ref.py
│   ├── evoattention/
│   │   ├── __init__.py
│   │   └── forward_ref.py
│   └── ...
├── atenir/        # AtenIR graph extraction + primitive Triton dispatch (ported byte-identical)
├── pipelines/
│   ├── a_atenir_llm/     # AtenIR summary + LLM plan/codegen/repair loop
│   ├── b_dispatch/       # LLM-free: dispatch-free Triton program + native wrapper codegen
│   ├── c_forward_only/   # forward-source-only ablation (tracks LLM cost)
│   └── shared/           # llm client, graph summaries, subprocess runner, CLI args
├── evolve/        # generic evaluator, scoring policies, config template, openevolve-run wrapper
├── bench/         # latency/memory harness + bench CLI
└── cli.py         # evograd ops|seed|verify|evolve|bench
```

## Migration status

- [x] Phase 1: `opdecl` core + 6 operator declarations. The transitional
      `OperatorSpec` bridge and JSON fixtures have now been removed; A/C prompts,
      Pipeline B, and OpenEvolve config rendering consume `OpDecl` directly.
- [x] Phase 2: `atenir/` ported byte-identical; forward refs; `oracle()`,
      `bind()`, `verify()`, declaration-driven inputs.
- [x] Phase 3: pipelines A/B/C ported and declaration-native — `--op <name>`
      replaces `--op-spec <json>`; `grad_reorder()`, the `eps` special-case,
      and `"d"+name` matching are gone; verification runs against the oracle
      via `verify_cli` (no per-bench evaluator paths).
- [x] Phase 4: `evolve/` — one generic evaluator × 5 scoring policies (metric
      names preserved for cross-repo comparability), config template, run
      wrapper that re-implements the fork's `--save-best-to`.
- [x] Phase 5: `bench/` harness + unified CLI + `scripts/gpu_parity.py`.
- [x] Completeness audit: automatic operator discovery; output-specific legacy
      tolerances; legacy-exact input distributions/seeds; forward and gradient
      dtype/shape correctness gates; flexible saved state; all LayerNorm shape
      suites; optional Liger baseline; OpenEvolve 0.2.27 model-list config API.
- [ ] **GPU validation** (blocking before first real use — dev boxes have no
      CUDA; all torch-facing code is validated by CPU-runnable unit tests and
      compile checks only):
      1. `PYTHONPATH=src python scripts/gpu_smoke.py --oracle-only`
      2. `PYTHONPATH=src python -m unittest discover tests` (runs the 8
         torch tests that skip on CPU-less boxes)
      3. `PYTHONPATH=src python scripts/gpu_parity.py --op layernorm
         --old-repo <fork> --candidate <old seed>` (repeat for evoattention)
      4. One end-to-end `evograd seed b` + `evograd evolve` on layernorm.
- [ ] PR the LLM parameter fallbacks from the fork upstream to openevolve.

## Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```

49 tests; the 8 torch-dependent ones self-skip on machines without torch and
run on any torch install (CPU is enough — they use a pure-torch toy op).
