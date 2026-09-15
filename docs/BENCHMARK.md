# Benchmark

Benchmark levels describe the computation being evaluated. Evaluation tiers
describe how an implementation is executed.

| Level | Scope | Example |
| --- | --- | --- |
| L1 | Primitive operator | RMSNorm, linear projection, RoPE |
| L2 | Composite operator | QKV + normalization + RoPE, SwiGLU MLP |
| L3 | Architectural block | One decoder layer with selected L2 providers |
| L4 | Whole model | Model forward, loss and backward |

L3 uses Tier 3 with `--scope block`; L4 uses `--scope model`. Block timing
covers forward and backward from supplied output gradients. Model timing
includes loss, backward, AdamW and gradient reset. Results from these scopes
are reported separately.

## Cases and ownership

- `evograd.ops` owns reusable primitive contracts and correctness cases.
- `benchmark/operator_suite` owns operator performance grids and aggregation.
- `benchmark/topdown/<model>` owns model configurations, harvested cases and
  captured blocks. Current families are Qwen3-0.6B, Llama-3.2-1B and AlphaFold3.
- `evograd.evaluation` owns execution, numerical checks, calibration and timing.

Qwen and Llama each declare their own four L2 tasks. Attention with output
projection, SwiGLU MLP and residual RMSNorm share reference computations but
have different model dimensions. Qwen's QKV task additionally includes Q/K
normalization.

The L3 adapters construct native decoder layers and install providers at their
declared sites. They are registered through `TIER3_ADAPTERS`; they are not
monolithic `OpDecl` kernels. L1/L2 tasks remain in `benchmark.TASKS`.

## Provenance

Captured cases identify the model execution, layer, weights, inputs and incoming
gradients through verified artifacts. Config-derived cases specify their
architecture and seeds and can run without a full-model capture. Reports retain
this distinction. Changing dimensions, dtype, data or backend changes the case.

The `llama_3_8b` configuration remains registered for existing operator-suite
grids. It is distinct from the Llama-3.2-1B model workload. Select named suites
explicitly when an operator has both generic and model-observed grids.

## Measurement

Correctness gates run before timing. Speedup is reference latency divided by
candidate latency for the same case and execution boundary. Failed providers
remain in coverage reports and have no valid timing result. Memory is reported
separately from speedup.

See [block correctness](L3_CORRECTNESS_CHECKS.md),
[Qwen usage](QWEN3_LEVEL4.md), the [evaluation guide](../src/evograd/evaluation/README.md)
and [result paths](RESULTS_LAYOUT.md).
