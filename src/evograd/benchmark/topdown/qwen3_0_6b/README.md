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
│   │                    <site>/{task,reference,capture}.py
│   └── level1/          manifest.py -- which primitives this model runs
├── layer/               artifact.py, capture.py, prepare.py -- the captured
│                        decoder layer the L2 sites draw their real
│                        invocations from, and what it takes to replay it
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
- L2: `qkv_norm_rope`, `attention`, `swiglu_mlp`, and
  `residual_rmsnorm`/`fused_add_rms_norm` model boundaries.
- L1: the primitive mapping for those boundaries.
- `layer/`: the representative decoder layer (layer 14), captured; a data
  source for L2, not a level.

Benchmark Level describes integration scope. T1/T2/T3 are evaluation execution
contexts; `fast`/`fair` are measurement modes.

## Commands

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.cli smoke
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/benchmark/topdown/qwen3_0_6b/harvest.json
```
