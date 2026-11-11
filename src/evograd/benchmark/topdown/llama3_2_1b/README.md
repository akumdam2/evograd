# Llama-3-8B top-down benchmark

The second top-down workload, and the one that made the first one's machinery
generic. Its stable name is `llama3_2_1b`; the public model key — the
`Provenance.model` value and the `--model` argument — is `llama_3_2_1b`.

Almost everything here is shared with Qwen3: the spec, the builder, the
observer, the manifest, the snapshot reader and its extraction all live in
[`../common/`](../common/). What is in this package is the list of facts only
Llama can supply.

## Ownership

```text
llama3_2_1b/
├── declaration.py       L3 (whole-model) facts shared by top-down tools
├── levels/
│   ├── level3/          model spec, build, smoke, and report
│   ├── level2/          QKV, attention, SwiGLU, residual/RMSNorm boundaries
│   └── level1/          primitive-to-composite mapping
├── layer/               captured decoder-layer artifact (the L2 sites' real
│                        invocations; not a level)
└── harvest/             observation, manifest, and snapshot extraction
```

Model patching lives in `evograd.evaluation.tier3.workloads.llama3_2_1b`;
correctness is judged by the verification package's Gates 1/2/3.

The legacy `llama3_decoder_layer` direct-block task that used to sit under
`ops/level3` is gone. It was not deleted *because of* this workload and it did
not survive into it: the two coexisted for a while — the ownership refactor
removed it on `main` while this workload was being developed on a branch — and
the branch's own decoder-layer material is the capture and replay under
`layer/`, not that declaration. Nothing here inherits from it.

## Canonical workload

- L2: `llama3_qkv_rope`, `llama3_attention`, `llama3_swiglu_mlp` and
  `llama3_residual_rmsnorm` model boundaries. 32 layers → **160 invocations**.
  Each Level-2 identity is this model's own: an architecture does not borrow
  another's task key, even where the mathematics is the same.
- L1: the primitive mapping for those boundaries, bound as this model's own
  observed suites onto the shared primitive contracts. **The harvest has not
  been run**, so every one of those suites is presently empty and no primitive
  declaration is changed by them; they appear the moment the snapshot is
  tracked, with no edit here and none in any primitive.
- L3: the whole training step (`levels/level3/`).
- `layer/`: representative decoder-layer capture, on the same footing as
  Qwen3's — a data source for L2, not a level.

Benchmark Level describes integration scope. T1/T2/T3 are evaluation execution
contexts; `fast`/`fair` are measurement modes.

## Commands

```bash
# resolve and print the workload id and hashes; runs nothing
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_2_1b --print-spec

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_2_1b \
  --out results/llama3-level3/canonical.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  python -m evograd.benchmark.topdown.llama3_2_1b.harvest.harvest \
  --out results/benchmark/topdown/llama3_2_1b/harvest.json
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot \
  --harvest results/benchmark/topdown/llama3_2_1b/harvest.json --write
```

The harvest observes the real-payload capture's first step: the pinned (gated)
checkpoint on the first WikiText-2 TRAIN batch of data order 101, built by
`evograd.verification.integrations.llama3_2_1b` under the verification settings
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`). The snapshot records that run (`run`), and
a capture refuses a snapshot that does not record its own run. `--synthetic`
harvests the random-weight, synthetic-token canonical spec instead (debug
overrides apply to it); the layer capture replays that step and reads such a
harvest, until the tracked snapshot is regenerated from the real run (see
"Regenerating the snapshots" in `src/evograd/verification/README.md`).

The smoke entry point takes no subcommand: `--out` and the override flags go
directly to `python -m evograd.benchmark.topdown.llama3_2_1b`.

## Not present, because each must be derived from a run

In the order they unblock each other:

1. `harvest/snapshot.json`. Until it is tracked, `load_snapshot("llama_3_2_1b")`
   refuses with the commands to produce one, and no declaration carries a
   `llama_3_2_1b_observed` suite — the hooks are wired and inert, so the suites
   appear when the file lands, with no edit to any operator.

   Llama-3-8B in BF16 is ~16 GiB of weights and ~16 GiB of gradients before
   activations. The Level-3 step takes no optimizer step, so it fits an
   80–120 GiB card; `--layers 4` shrinks it for a smoke.

2. The layer capture (`layer16.pt`). Every level-2 derivation consumes it and
   refuses by name rather than substituting synthetic tensors.

## Real text: the pinned checkpoint on WikiText-2

The canonical workload above is random weights on synthetic tokens. A second,
separate workload identity trains the pinned `meta-llama/Llama-3.2-1B`
checkpoint (`spec.PRETRAINED`) on WikiText-2 raw (`spec.REAL_TEXT`, the same
revision Qwen3 uses). Its workload id carries `.pretrained.realtext`, so no
synthetic calibration or snapshot binds to it, and it exists only at the
published architecture (`--layers` is refused).

The checkpoint is gated. Accept Meta's Llama 3.2 license on the Hub with your
account and log in once; the preparation step is then the only thing that
touches the network. Benchmark runs read the local cache
(`data/llama_3_2_1b/hf_cache`, or `EVOGRAD_HF_CACHE`).

```bash
pip install -e ".[llama3]"
huggingface-cli login
python scripts/prepare_benchmark_inputs.py --benchmark llama_3_2_1b
```

Non-empty WikiText lines are tokenised with the checkpoint's tokenizer, joined
with its EOS token and packed into `seq_len` blocks; the trailing partial block
is dropped. Training batches follow a seeded permutation of the train split
(a single step uses the first batch of order `data_seed + seed`); validation
is the published validation split in one fixed order. Reports record the
dataset revision, parquet and tokenizer hashes and block counts
(`data_identity`).
