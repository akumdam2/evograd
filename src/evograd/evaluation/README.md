# Evaluation

`evograd.evaluation` defines **how a candidate is evaluated**: how providers
run, which execution context checks correctness and timing, and how stable
reports are produced. Benchmark Level and Evaluation Tier are orthogonal axes.

What it does not define is *what* to run. The tasks, their cases, the grids and
the provenance belong to `evograd.benchmark`; this package consumes them. The
two are put together one level up, in `evograd.cli` and `evograd.suite_cli`, so
that neither package has to import the other to run a command.

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
├── tier3/
│   ├── model.py, patch.py, runner.py, cli.py
│   ├── gate/     workload-neutral gate primitives
│   └── workloads/
│       ├── qwen3_0_6b/
│       └── alphafold3/
└── workloads/    per-workload checks that are not tier-specific
    └── qwen3_0_6b/
        ├── level1/  verify.py, calibrate.py, cli.py
        ├── level2/  one module per site, plus calibrate.py,
        │            negative_controls.py
        └── level3/  replay.py
```

`workloads/` and `tier3/workloads/` answer different questions and are not a
duplication. `tier3/workloads/` is how a candidate is *patched into a live
model* and gated there. `workloads/` is how one model's captured cases are
judged at all — reference-versus-production comparison, the tolerance each
result is held to, the repeated-noise measurement that tolerance is calibrated
from, and the negative controls that show it still rejects a wrong kernel.

The cases those checks read belong to the benchmark
(`evograd.benchmark.topdown.<model>`), which captures and describes them but
never decides whether an implementation passes. That is the boundary this
package exists to hold: a threshold is not derived by the module that defines
the case it gates.

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
