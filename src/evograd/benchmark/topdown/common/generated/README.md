# Generated-input DiT-S/2 benchmark

This workload executes a full DiT-S/2 forward, diffusion loss, backward, and
AdamW step at Level 3. The benchmark spec (`spec.canonical`) uses
deterministic generated batches and seeded random weights. The harvest and the
verification use pinned Imagenette images encoded with the pinned
`stabilityai/sd-vae-ft-mse` VAE, and start from a frozen warm start of the
seeded initialization (`evograd.verification.integrations.warm_start`): the
reference project does not provide an official S/2 checkpoint, and at the
initialization adaLN-Zero leaves every block without gradient. This is
real-image-derived training input, not pretrained DiT fine-tuning or the full
ImageNet distribution. The implementation imports nothing from `references/`.

The generated latents and diffusion noise are independent N(0,1) tensors.
Timesteps and class labels are sampled uniformly from their declared ranges;
class-drop masks are sampled with probability 0.1. A seed fixes each batch.

The canonical configuration has 12 blocks, hidden width 384, six attention
heads, 4×32×32 latents, batch 256, float32, and 32,865,056 trainable
parameters. Its two patch sites are `dit_modulation` (adaLN shift, scale, and
gate projection) and `dit_mlp` (linear, tanh-approximate GELU, linear). Each
site is invoked 12 times per model step. Level 1 binds observed `linear` and
`spatial_conv2d` calls. Attention, affine-free layer normalization, and the
diffusion loss remain outside the standalone operator pool.

Measured Level-1/2 cases are derived from the whole-model harvest of the
verification run. A reduced configuration can verify execution but cannot
publish a snapshot.

## Harvest

The harvest runs the step the verification capture records first: the
verification workload (`evograd.verification.integrations.dit_s_2`: real
input, the streamed latent pool, the warm start), a fresh native model, the
first training batch of data order 101, under the verification's deterministic
settings (`integrations.harvest_run.first_step`). Its snapshot (schema `/2`)
names that run under `run`: `weights` (`warm_start:sha256:<file digest>`; the
workload hash does not cover the warm start), `data` (the data identity the
capture records, without its per-order batches), `order`, `batch_digest`
(the batch, as the capture names it) and `parameter_state_sha256` (the
warm-started parameters).

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.benchmark.topdown.dit_s_2.harvest.harvest \
  --out results/dit_s_2/harvest.json --write-snapshot
```

`python -m evograd.benchmark.topdown.dit_s_2.harvest.snapshot --harvest
HARVEST.json [--write]` derives the snapshot from a saved harvest without a
GPU; it, too, accepts only a harvest of the verification run. Reduced
configurations (changed `batch_size`, `device`, `arch_overrides`) run from
Python with `collect(spec)` and never write a tracked snapshot.

The latent pool is `data/dit_s_2/batches/latent_pool.pt` (override with
`EVOGRAD_DIT_LATENT_POOL`; build it once with `data.build_latent_pool()` on a
GPU) and the warm start `data/dit_s_2/batches/warm_start.pt` (override with
`EVOGRAD_DIT_S_2_WARM_START`; written by `integrations.warm_start`).

## Pinned real-image input

Install the data/VAE dependencies and cache the exact source files:

```bash
pip install -e ".[dit]"
python scripts/prepare_benchmark_inputs.py --benchmark dit_s_2
```

Preparation uses a 256-pixel center crop, random horizontal flip, `[-1, 1]`
normalization, and VAE latent scale 0.18215. A fixed seed controls flips,
latent sampling, diffusion noise, timesteps, and class dropout. Prepared
batches (a workload's `data_path`) are cached at `data/dit_s_2/batches/`
(override with `EVOGRAD_DIT_DATA_CACHE`). Imagenette labels are mapped to
their ImageNet-1k IDs. The latent pool encodes every training image the same
way, and the stream draws fresh noise, timesteps and class dropout per data
order and step.
