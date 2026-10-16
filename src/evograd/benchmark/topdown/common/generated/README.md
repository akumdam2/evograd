# Generated-input DiT-S/2 benchmark

This workload executes a full DiT-S/2 forward, diffusion loss, backward, and
AdamW step at Level 3. It uses deterministic generated batches by default.
`--real-input` uses pinned Imagenette images encoded with the pinned
`stabilityai/sd-vae-ft-mse` VAE. DiT-S/2 weights remain seeded random weights:
the reference project does not provide an official S/2 checkpoint. This is
real-image-derived training input, not pretrained DiT fine-tuning or the full
ImageNet distribution. The implementation imports nothing from `references/`.

The generated latents and diffusion noise are independent N(0,1) tensors.
Timesteps and class labels are sampled uniformly from their declared ranges;
class-drop masks are sampled with probability 0.1. A seed fixes each batch.
The model gate calibrates on five batches and evaluates on five disjoint
batches, checking outputs, the latent input gradient, every parameter
gradient, and the updated loss trajectory.

The canonical configuration has 12 blocks, hidden width 384, six attention
heads, 4×32×32 latents, batch 256, float32, and 32,865,056 trainable
parameters. Its two patch sites are `dit_modulation` (adaLN shift, scale, and
gate projection) and `dit_mlp` (linear, tanh-approximate GELU, linear). Each
site is invoked 12 times per model step. Level 1 binds observed `linear` and
`spatial_conv2d` calls. Attention, affine-free layer normalization, and the
diffusion loss remain outside the standalone operator pool.

Measured Level-1/2 cases are derived from a canonical whole-model harvest;
they remain empty until its snapshot is saved. A reduced configuration can
verify execution but cannot publish a canonical timing snapshot.

## Canonical generated-input identity check

```bash
python -m evograd.benchmark.topdown.dit_s_2.harvest.harvest \
  --out results/dit_s_2/harvest.json --write-snapshot

python -m evograd.evaluation.tier3.workloads.common.generated.gate \
  --model dit_s_2 \
  --config src/evograd/benchmark/topdown/dit_s_2/benchmark.json \
  --sites dit_modulation,dit_mlp --steps 5 \
  --out results/dit_s_2/calibration.json

python -m evograd.evaluation.tier3.cli --model dit_s_2 \
  --calibration results/dit_s_2/calibration.json \
  --structural-identity --providers eager,structural_identity \
  --baseline none --out results/dit_s_2/identity.json
```

For a reduced run, pass a JSON file with changed `batch_size`, `device`, and
`arch_overrides` to harvest and gate as `--config`, and to Tier 3 as
`--model-config`. Its calibration is tied to that exact configuration. Omit
`--write-snapshot` for a reduced run.

## Pinned real-image identity check

Install the data/VAE dependencies and cache the exact source files:

```bash
pip install -e ".[dit]"
hf download ShaomuTan/imagenette 320px/train-00000-of-00001.parquet \
  --repo-type dataset --revision 061f0ab5ca70523e13982a7ba463c076af657386
hf download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.safetensors \
  --revision 31f26fdeee1355a5c34592e401dd41e45d25a493
```

Preparation uses a 256-pixel center crop, random horizontal flip, `[-1, 1]`
normalization, and VAE latent scale 0.18215. A fixed seed controls flips,
latent sampling, diffusion noise, timesteps, and class dropout. Prepared
batches are cached at `~/.cache/evograd/dit/` (override with
`EVOGRAD_DIT_DATA_CACHE`). The default gate needs ten batches: five for
calibration and five disjoint batches for evaluation. Imagenette labels are
mapped to their ImageNet-1k IDs.

```bash
python -m evograd.evaluation.tier3.workloads.common.generated.gate \
  --model dit_s_2 --config src/evograd/benchmark/topdown/dit_s_2/benchmark.json \
  --real-input --sites dit_modulation,dit_mlp --steps 5 \
  --out results/dit_s_2/real_calibration.json

python -m evograd.evaluation.tier3.cli --model dit_s_2 --real-input \
  --calibration results/dit_s_2/real_calibration.json \
  --structural-identity --providers eager,structural_identity \
  --baseline none --out results/dit_s_2/real_identity.json
```

Add `--real-input` to the harvest command to observe Level-1/2 shapes from
the first prepared image batch. The tracked canonical snapshot remains the
generated-input identity; real-input harvests use `--out` without
`--write-snapshot`. Shapes are the same under both modes for the fixed
configuration.
