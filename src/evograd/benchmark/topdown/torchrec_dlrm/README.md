# TorchRec DLRM benchmark

This workload reproduces the single-device TorchRec DLRM example architecture
with local PyTorch modules. It does not import from `references/` or require
TorchRec at runtime. The whole-model Level-3 path trains on seeded generated
dense features, categorical IDs, and binary labels. `--real-input` reads fixed
Criteo Kaggle training rows from TorchRec's preprocessed NumPy format.

The canonical configuration uses 13 dense features, 26 embedding tables of
100,000 rows each, embedding width 64, batch 32, dense MLP
13→512→256→64, and over MLP 415→512→512→256→1. This is over 160 million
parameters. Each table uses summed `EmbeddingBag`; feature interaction forms
strict lower-triangular pairwise dot products. The Level-1 bindings are
`embedding_bag` and `linear`; Level 2 binds `dlrm_interaction`.

The benchmark uses dense embedding gradients and AdamW. It does not implement
TorchRec distributed sharding or its fused sparse optimizer, so results should
be described as a single-device architecture workload.

## Real-input identity check

Install the benchmark runtime with `pip install -e ".[dlrm]"`. From the
repository root, run:

```bash
python scripts/prepare_benchmark_inputs.py --benchmark torchrec_dlrm
```

On a cluster without outbound network access, download the Criteo Kaggle
archive on an internet-connected machine, copy it to the cluster, and pass
`--criteo-archive /path/to/archive.tar.gz`. Alternatively, place an extracted
`train.txt` in `data/torchrec_dlrm/raw/` and rerun the same command.

The script downloads a revision- and SHA256-pinned Criteo Kaggle archive from
Hugging Face, extracts `train.txt`, and converts it to TorchRec-compatible
`train_dense.npy`, `train_sparse.npy`, and
`train_labels.npy` in `data/torchrec_dlrm/processed/`. If the raw file or processed arrays
already exist, it reuses them. The benchmark imports no TorchRec or reference
code at runtime. It reads the
already transformed dense features and binary click labels, and maps each
categorical ID modulo its table size, as TorchRec's in-memory loader does.

The source is the labeled first half of `train`, excluding the half TorchRec
uses for its test stage. `data_seed` selects distinct rows deterministically.
Calibration and evaluation use disjoint fixed batches. The prepared artifact
records selected row IDs, the source-row digest, file metadata, and the mapping
policy; its file hash is bound into calibration. No pretrained checkpoint is
needed for structural identity because both providers start from the same
seeded weights. The 100,000-row tables make this a capped-cardinality Criteo
workload, not TorchRec's full-cardinality distributed benchmark.

```bash
ls data/torchrec_dlrm/processed/train_{dense,sparse,labels}.npy
mkdir -p results/torchrec_dlrm
python -m evograd.evaluation.tier3.workloads.common.generated.gate \
  --model torchrec_dlrm \
  --config src/evograd/benchmark/topdown/torchrec_dlrm/benchmark.json \
  --real-input \
  --sites embedding_bag,dlrm_interaction --steps 5 \
  --out results/torchrec_dlrm/real_calibration.json

python -m evograd.evaluation.tier3.cli \
  --model torchrec_dlrm --real-input \
  --calibration results/torchrec_dlrm/real_calibration.json \
  --structural-identity --providers eager,structural_identity \
  --baseline none --out results/torchrec_dlrm/real_identity.json
```

Harvest can use `--real-input`; prepared batches live under
`data/torchrec_dlrm/batches/` by default.
Real-input harvests do not publish the generated-input canonical snapshot.

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
