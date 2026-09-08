# Evaluation

`evograd.evaluation` defines **how a candidate is evaluated**: how providers
run, which execution context checks correctness and timing, and how stable
reports are produced. Benchmark Level and Evaluation Tier are orthogonal axes.

## Three-Tier Evaluation Protocol

| Tier | Execution context | Purpose |
| ---: | --- | --- |
| T1 | direct pair | Call the candidate forward/backward pair directly |
| T2 | operator/autograd | Compare providers through a common `nn.Module` and autograd path |
| T3 | end-to-end model training | Patch the candidate into a real model training step |

`fast` and `fair` are measurement modes, not another Tier or a Benchmark
Level. `fast` supplies evolution fitness; `fair` supplies publishable direct-pair
measurements.

## Package layout

```text
evaluation/
├── common/       providers, canonical reports, benchmark-report adapters
├── tier1/        fast.py, fair.py, cli.py
├── tier2/        runner.py, integrated.py, cli.py
└── tier3/
    ├── model.py, patch.py, runner.py, cli.py
    ├── gate/     workload-neutral gate primitives
    └── workloads/
        ├── qwen3_0_6b/
        └── alphafold3/
```

The Qwen3-0.6B T3 checks cover local outputs and input/weight gradients,
valid-token KL, whole-model gradient-vector relative L2, and consecutive
training/validation loss behavior. The first three may screen a provider after
their thresholds have been calibrated and frozen; training behavior remains
diagnostic until the evidence justifies a gate. An internal legacy filename
does not define an additional public protocol.

This reorganization changes module ownership and imports only. It does not
change T3 thresholds, gates, report schemas, default measurement behavior, or
exit codes.

## Stable CLI

The command names and their arguments remain compatible:

```bash
evograd tier1-bench --help
evograd tier2-bench --help
evograd tier3-bench --help
evograd suite --help
```

New evaluation output belongs under
`results/evaluation/tier<N>/<workload>/...`; historical output is not moved.
See [`docs/RESULTS_LAYOUT.md`](../../../docs/RESULTS_LAYOUT.md).
