# evograd

**Evolved backward kernels.** Given a forward reference for an operator,
evograd generates a training-ready autograd-pair seed (forward + backward
Triton kernels) through one of several pipelines, optimizes it with
[OpenEvolve](https://github.com/codelion/openevolve), and benchmarks the
result against strong baselines.

Successor to the backward-benchmark work in the `openevolve` fork
(`atenir/`, `pipeline/`, `benchmark/`); OpenEvolve itself is consumed as a
package dependency, not forked.

## The core abstraction: activity-annotated operator declarations

Every operator is declared once, with Enzyme-style activity annotations:

```python
op = declare_op(
    name="evoattention",
    forward="evograd.ops.evoattention_forward_ref:evoattention_forward_ref",
    dims=("B", "S", "H", "N", "D"),
    args=(
        Duplicated("q",         "[B, S, N, H, D]"),   # active: gets a gradient
        Duplicated("k",         "[B, S, N, H, D]"),
        Duplicated("v",         "[B, S, N, H, D]"),
        Const("res_mask",       "[B, S, 1, 1, N]"),   # inactive: no gradient
        Duplicated("pair_bias", "[B, 1, H, N, N]", grad="d_pair_bias"),
    ),
    output=Duplicated("o", "[B, S, N, H, D]"),        # its shadow is the upstream grad
    ...
)
```

`Duplicated` marks an active tensor (primal + gradient "shadow"); `Const`
marks an inactive one (the harness emits `None` for its grad slot). The
declaration is the single source of truth — everything else derives from it:

| Consumer | Derivation |
|---|---|
| Pipeline A/C prompts | rendered from typed args, shapes, semantics |
| Pipeline B wrapper codegen | grads = `Duplicated` args in declared order; `Const` → `None` |
| Ground-truth oracle | `torch.autograd.grad` over `forward` with `requires_grad` on exactly the `Duplicated` args |
| Correctness verifier | per-`Duplicated` comparison at declared workloads/tolerances |
| OpenEvolve evaluator | one generic evaluator × pluggable scoring policy |

## Planned layout

```text
src/evograd/
├── opdecl/        # declare_op, Duplicated, Const  [done]
│                  # oracle, bind (autograd.Function builder), verify  [next]
├── ops/           # one declaration per operator  [6 ops ported]
├── atenir/        # graph extraction + primitive Triton dispatch  [to port]
├── pipelines/     # a_atenir_llm / b_dispatch / c_forward_only  [to port]
├── evolve/        # generic evaluator, scoring policies, openevolve-run wrapper  [to port]
└── bench/         # latency / memory / ncu harness, strong baselines  [to port]
```

Target CLI: `evograd seed | verify | evolve | bench --op <name> ...`

## Migration status

- [x] Phase 1: `opdecl` core + 6 operator declarations + legacy `OperatorSpec`
      bridge (`to_operator_spec`), diff-tested against snapshots of the old
      repo's `<op>_spec.json` files (`tests/fixtures/`).
- [ ] Phase 2: port `atenir/` unchanged; `op.oracle()` + `op.bind()`.
- [ ] Phase 3: port pipelines A/B/C (first via the bridge, then native).
- [ ] Phase 4: generic evaluator + scoring policies; run wrapper
      (re-implements the fork's `--save-best-to`; PR the LLM param fallbacks
      upstream).
- [ ] Phase 5: bench harness; GPU parity gate vs the old repo on layernorm +
      evoattention.

## Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```

No GPU or torch needed for the phase-1 tests. Anything touching kernels or
the oracle must be smoke-tested on a GPU node (dev boxes have no CUDA).
