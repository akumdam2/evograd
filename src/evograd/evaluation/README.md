# Evaluation

The evaluation package executes providers, checks correctness and measures
performance. Benchmark declarations own cases, shapes and provenance.

| Tier | Execution context |
| --- | --- |
| T1 | Direct forward/backward pair |
| T2 | Operator module through PyTorch autograd |
| T3 | Providers installed in an architectural block or whole model |

`fast` and `fair` are measurement modes, separate from benchmark levels and tiers.

## Tier 3 scopes

`--scope block` evaluates an L3 block using supplied output gradients, without
an optimizer. Qwen3-0.6B and Llama-3.2-1B adapters support captured and
config-derived inputs. `block.py` drives execution, `gate/block.py` validates
results and policies, and `block_cli.py` handles the CLI and provider workers.

`--scope model` retains the existing L4 training-step behavior: task loss,
backward, AdamW and gradient reset. Each workload supplies its model-specific
gates. Qwen supports local output/gradient checks, model prediction and gradient
comparisons, and training/validation diagnostics according to the selected policy.

The report reader distinguishes `evograd-tier3-block-v1` from
`evograd-tier3-model-v2`; these timing boundaries are not pooled. Block numerical
policies use `evograd-t3-block-policy/2` and bind to the case and environment.
Old block policies require recalibration.

## Workload adapters

Shared execution and timing live in `tier3/`. Model construction, patch sites,
input preparation and local semantics live in `tier3/workloads/<model>/`.
`workloads/<model>/` contains checks for cases extracted from model captures.
Adding an architecture should reuse the shared runner and provide an adapter.

## Commands

```bash
evograd tier1-bench --help
evograd tier2-bench --help
evograd tier3-bench --help
evograd suite --help
```

See [Qwen usage](../../../docs/QWEN3_LEVEL4.md),
[block correctness](../../../docs/L3_CORRECTNESS_CHECKS.md) and
[result paths](../../../docs/RESULTS_LAYOUT.md).
