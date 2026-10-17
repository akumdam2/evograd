# TorchRec DLRM generated-input benchmark

This workload reproduces the single-device TorchRec DLRM example architecture
with local PyTorch modules. It does not import from `references/` or require
TorchRec at runtime. The whole-model Level-3 path trains on seeded generated
dense features, categorical IDs, and binary labels. `--real-input` is reserved
and raises an explicit error.

The canonical configuration uses 13 dense features, 26 embedding tables of
100,000 rows each, embedding width 64, batch 32, dense MLP
13→512→256→64, and over MLP 415→512→512→256→1. This is over 160 million
parameters. Each table uses summed `EmbeddingBag`; feature interaction forms
strict lower-triangular pairwise dot products. The Level-1 bindings are
`embedding_bag` and `linear`; Level 2 binds `dlrm_interaction`.

The benchmark uses dense embedding gradients and AdamW. It does not implement
TorchRec distributed sharding or its fused sparse optimizer, so results should
be described as a single-device architecture workload.

## Canonical generated-input identity check

```bash
python -m evograd.benchmark.topdown.torchrec_dlrm.harvest.harvest \
  --out results/torchrec_dlrm/harvest.json --write-snapshot

python -m evograd.evaluation.tier3.workloads.common.generated.gate \
  --model torchrec_dlrm \
  --config src/evograd/benchmark/topdown/torchrec_dlrm/benchmark.json \
  --sites embedding_bag,dlrm_interaction --steps 5 \
  --out results/torchrec_dlrm/calibration.json

python -m evograd.evaluation.tier3.cli --model torchrec_dlrm \
  --calibration results/torchrec_dlrm/calibration.json \
  --structural-identity --providers eager,structural_identity \
  --baseline none --out results/torchrec_dlrm/identity.json
```

For a reduced check, supply a JSON configuration that lowers the number and
size of embedding tables. Pass it to harvest and gate with `--config`, and to
Tier 3 with `--model-config`. Omit `--write-snapshot` for a reduced run.
