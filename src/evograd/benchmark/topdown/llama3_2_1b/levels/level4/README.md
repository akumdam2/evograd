# Level 4 — the canonical Meta-Llama-3-8B training step

Level 4 declares no operator. It is one whole model executed the way training
executes it, and it exists so that every level below can be *derived from* a
real step rather than inferred from one. Nothing here is evolved; nothing here
is timed as a benchmark result.

## The canonical workload

| field | value |
| --- | --- |
| model | Meta-Llama-3-8B, 8.03B parameters (6.98B excluding the untied embedding and `lm_head`), randomly initialised |
| batch size | 2 |
| sequence length | 2048 |
| tokens per step | 4096 |
| dtype | bfloat16 |
| device | cuda |
| attention backend | sdpa |
| mode | `model.train()` |
| `use_cache` | False |
| gradient checkpointing | disabled |
| optimizer step | none |
| seed | 0 |
| workload id | `meta-llama-3-8b.train.bs2.seq2048.bf16.cuda.sdpa.5f08b9e7` |

The executed step is exactly:

```python
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

`labels = input_ids.clone()`, and `input_ids` are synthetic, drawn from an
explicit CPU generator seeded with the workload seed so the token stream is
identical on any device and does not depend on how many random numbers model
construction consumed.

Sequence 2048 rather than the 8192 the architecture permits, so the observed
shapes line up with Qwen3's and the two workloads can be read side by side. It
is a canonical choice, not a limit.

## What is in this package

| file | what it declares |
| --- | --- |
| `spec.py` | `LLAMA_3_2_1B` — the published architecture, written out — and the canonical batch/sequence/dtype/device/backend as `WorkloadSpec`'s field defaults |
| `model.py` | two class names, `transformers:LlamaConfig` and `transformers:LlamaForCausalLM`, bound to the shared builder |
| `smoke.py` | runs the step once and describes what happened |
| `report.py` | the report schema string, `evograd-llama3-smoke/1` |

Everything that *acts* — building, seeding, stepping, verifying, reporting —
is in `../../common/`. What is here is the short list only Llama can answer.

## Three settings no command-line flag can change

`--batch-size`, `--seq-len`, `--dtype`, `--device`, `--attn`, `--seed` and
`--layers` all shrink or shift the workload; any of them changes the workload
id, sets `"canonical": false`, and prints a banner to stderr. But three
settings are refused by `WorkloadSpec.validate` rather than by the parser, so
no flag can produce a run that claims to be this workload while executing a
different graph:

- **`use_cache` must be False.** A KV cache is a decode-time structure: with it
  on, the model allocates and returns per-layer caches a training step never
  reads, so the executed graph is not the training graph.
- **`gradient_checkpointing` must be False.** Checkpointing re-executes forward
  regions inside backward, which would double-count every operator invocation
  the observation stage records.
- **`training` must be True.** This workload is defined as a training step.

## No Hub token

Llama-3 is a gated repository, but nothing here fetches a checkpoint, a config
or a tokenizer. The architecture is written out in `spec.py` and the model is
built with random weights, because a reference execution must not depend on
what the Hub served that day or on a node having network access.

## Running it

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_2_1b \
    --out results/llama3-level4/canonical.json
```

Exit code 0 means the step ran and every check passed. `--print-spec` resolves
the workload id and hashes without running anything.

## What the report contains

Schema `evograd-llama3-smoke/1`:

- **`workload`** — id, hashes, `canonical` flag, the full architecture config,
  batch/sequence/token counts, dtype, device, backend, cache and checkpointing
  state, seed
- **`environment`** — Python, platform, torch, transformers, CUDA, cuDNN, GPU
  name and capability, total memory, driver version
- **`effective`** — what the *built model* reports, as distinct from what was
  requested: backend at the root and at every submodule, `use_cache`,
  checkpointing, `model.training`, parameter dtypes and devices, buffer dtypes,
  input dtype/shape/checksum, loss and logits dtypes, whether a KV cache came back
- **`result`** — loss, whether it is finite, trainable parameter count, how many
  received gradients, the names of any that did not or whose gradient is not finite
- **`diagnostics`** — wall time and peak allocated/reserved memory

**Everything under `diagnostics` is diagnostic only.** One unwarmed step, no
repetition, no L2 flush, no median: a sanity check on the shape of the
allocation, not a measurement.

## One environment dependency worth knowing

The step is numerically correct across a range of Transformers versions, but
the *observed operator* is not. Whether SDPA is reached with `enable_gqa=True`
and eight key/value heads, or with an explicit mask and `repeat_kv` having
materialised thirty-two, is the installed library's choice. The harvest refuses
the second case by name; see `../../harvest/`. The canonical harvest on record
was produced under transformers 5.16.1.
