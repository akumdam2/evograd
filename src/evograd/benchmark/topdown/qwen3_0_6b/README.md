# Qwen3-0.6B top-down benchmark

This is the current Qwen instance of the paper's benchmark hierarchy. Its
stable name is `qwen3_0_6b`; it is not the planned Qwen3-Next workload.

## Ownership

```text
qwen3_0_6b/
├── declaration.py       L4 facts shared by top-down tools
├── levels/
│   ├── level4/          model spec, build, smoke, and report
│   ├── level3/          captured decoder-layer artifact and replay
│   ├── level2/          QKV, attention, SwiGLU, residual/RMSNorm boundaries
│   └── level1/          primitive-to-composite mapping
└── harvest/             observation, manifest, and frozen snapshot
```

Model patching and the local/KL/gradient/training-behavior checks are evaluation
concerns and live under:

```text
evograd.evaluation.tier3.workloads.qwen3_0_6b
```

## Current scope

- L4: the Qwen3-0.6B training workload, smoke, harvest, and snapshot.
- L2: `qkv_norm_rope`, `attention`, `swiglu_mlp`, and
  `residual_rmsnorm`/`fused_add_rms_norm` model boundaries.
- L1: the primitive mapping for those boundaries.
- L3: representative decoder-layer capture/replay infrastructure only. It is
  not a completed top-down L3 evolution target until the layer-level saved-state
  contract, complete return contract, and pretrained-activation coverage exist.

Benchmark Level describes integration scope. T1/T2/T3 are evaluation execution
contexts; `fast`/`fair` are measurement modes.

## Commands

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.cli smoke
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/benchmark/topdown/qwen3_0_6b/harvest.json
evograd tier3-bench --model qwen3_0_6b --help
```

Historical `results/qwen3-level4/` and external cache paths are not moved. Only
new runs use the new layout.

Qwen T3 numerical thresholds must be calibrated and frozen from trusted
references, repeated noise, and metric-specific floors before a candidate is
run, then enforced on an independent holdout. Training behavior remains
diagnostic under the current policy. This package move does not alter those
algorithms or gate behavior.
