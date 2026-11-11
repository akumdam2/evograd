# Qwen3-0.6B top-down benchmark

This is the Qwen3-0.6B instance of the paper's benchmark hierarchy. Its
stable name is `qwen3_0_6b`.

## Ownership

```text
qwen3_0_6b/
├── declaration.py       L3 (whole-model) facts shared by top-down tools
├── levels/
│   ├── level3/          model spec, build, smoke, and report
│   ├── level2/          manifest.py plus one package per site:
│   │                    <site>/{task,reference}.py
│   └── level1/          manifest.py -- which primitives this model runs
└── harvest/             observation, manifest, and frozen snapshot
```

Everything above describes *the cases*: where a boundary sits, which tensors
cross it, in which layout, how often, and whether what was captured still
matches the frozen harvest. `levels/level2/manifest.py` is the single source
for the site names, task keys, frequencies and module paths -- it reads the
snapshot rather than restating it.

Deciding whether an implementation of one of those cases is acceptable is a
separate concern with a separate owner: the verification package's Gates 1/2/3
(`evograd.verification`). Model patching lives in
`evograd.evaluation.tier3.workloads.qwen3_0_6b`.

## Current scope

- L3: the Qwen3-0.6B whole-model training workload, smoke, harvest, and
  snapshot.
- L2: `attention_block` (the whole `Qwen3Attention`), `swiglu_mlp`,
  `residual_rmsnorm`/`fused_add_rms_norm` and `lm_head_ce` (the LM head with the
  causal-LM loss) model boundaries. Their payloads are the real-payload
  capture's (`evograd.verification.integrations.qwen3_0_6b`).
- L1: the primitive mapping for those boundaries.

Benchmark Level describes integration scope. T1/T2/T3 are evaluation execution
contexts; `fast`/`fair` are measurement modes.

## Commands

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.cli smoke
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/benchmark/topdown/qwen3_0_6b/harvest.json
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot \
  --harvest results/benchmark/topdown/qwen3_0_6b/harvest.json --write
```

The harvest observes the real-payload capture's first step: the pinned checkpoint
on the first WikiText-2 TRAIN batch of data order 101, built by
`evograd.verification.integrations.qwen3_0_6b` under the verification settings
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`). The snapshot records that run (`run`), and
a capture refuses a snapshot that does not record its own run. `--synthetic`
harvests the random-weight, synthetic-token canonical spec instead (debug
overrides apply to it) and never backs a capture. Regenerate the tracked
snapshot from the real run whenever `harvest/snapshot.py`'s `TASK_SOURCES`
change (see "Regenerating the snapshots" in `src/evograd/verification/README.md`).
