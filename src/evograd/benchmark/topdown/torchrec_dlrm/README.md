# TorchRec DLRM benchmark

This workload reproduces the single-device TorchRec DLRM example architecture
with local PyTorch modules. It does not import from `references/` or require
TorchRec at runtime. The benchmark spec (`spec.canonical`) trains on seeded
generated dense features, categorical IDs, and binary labels. The harvest and
the verification stream Criteo Kaggle training rows from TorchRec's
preprocessed NumPy format. Both start from the seed-0 initialization: the
model has no published weights.

The canonical configuration uses 13 dense features, 26 embedding tables of
100,000 rows each, embedding width 64, batch 32, dense MLP
13→512→256→64, and over MLP 415→512→512→256→1. This is over 160 million
parameters. Each table uses summed `EmbeddingBag`; feature interaction forms
strict lower-triangular pairwise dot products. The Level-1 bindings are
`embedding_bag` and `linear`; Level 2 binds `dlrm_interaction`.

The benchmark uses dense embedding gradients and AdamW. It does not implement
TorchRec distributed sharding or its fused sparse optimizer, so results should
be described as a single-device architecture workload.

## Real input

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
The prepared artifact records selected row IDs, the source-row digest, file
metadata, and the mapping policy. The 100,000-row tables make this a capped-cardinality Criteo
workload, not TorchRec's full-cardinality distributed benchmark.

```bash
ls data/torchrec_dlrm/processed/train_{dense,sparse,labels}.npy
```

Prepared batches (a workload's `data_path`) live under
`data/torchrec_dlrm/batches/` by default; the harvest and the verification
read the processed arrays directly (`dlrm_data.CriteoStream`, override the
directory with `EVOGRAD_DLRM_CRITEO_DIR`).

## Harvest

The harvest runs the step the verification capture records first: the
verification workload (`evograd.verification.integrations.torchrec_dlrm`:
the streamed Criteo rows), a fresh native model at the seed-0 initialization,
the first training batch of data order 101, under the verification's
deterministic settings (`integrations.harvest_run.first_step`). Its snapshot
(schema `/2`) names that run under `run`: `weights` (`random:seed0`), `data`
(the data identity the capture records, without its per-order batches),
`order`, `batch_digest` and `parameter_state_sha256`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.benchmark.topdown.torchrec_dlrm.harvest.harvest \
  --out results/torchrec_dlrm/harvest.json --write-snapshot
```

`python -m evograd.benchmark.topdown.torchrec_dlrm.harvest.snapshot --harvest
HARVEST.json [--write]` derives the snapshot from a saved harvest without a
GPU. For a reduced check, call `collect(spec)` from Python on a spec with
fewer and smaller embedding tables; it never writes a tracked snapshot.
