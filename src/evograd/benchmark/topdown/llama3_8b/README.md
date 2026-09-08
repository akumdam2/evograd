# Llama-3-8B top-down workload

This package owns the Meta-Llama-3-8B L4 specification, model builder, smoke,
harvest, and snapshot tooling. Shared machinery comes from
`evograd.benchmark.topdown.common`; model-specific classes, observation plan,
and configuration remain here.

```text
llama3_8b/
├── declaration.py
├── levels/level4/
└── harvest/
```

L4 and the harvest path are implemented and tested. Top-down L3/L2/L1
declarations and a Tier-3 patch adapter are not complete, so
`evograd tier3-bench` does not register Llama-3-8B. This remains an explicit
coverage gap. The legacy `llama3_decoder_layer` direct-block task that used to
sit under `ops/level3` could not stand in for it and has been deleted.

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.harvest.harvest \
  --out results/benchmark/topdown/llama3_8b/harvest.json
```

Historical output remains at its original read-only path.
