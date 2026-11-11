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
parameters. Its Level-2 sites are attributes of every `DiTBlock` the model
calls through:

| Task (`dit_s_2_...`) | Attribute | Boundary | Calls per step |
| --- | --- | --- | --- |
| `dit_modulation` | `mod_fn` | SiLU, then the biased projection to the six adaLN-Zero shift, scale and gate vectors | 12 |
| `dit_attention` | `attn_fn` | biased qkv projection 384 -> 1152, q/k/v of 6 heads x 64, non-causal SDPA, heads merged, biased output projection | 12 |
| `dit_adaln_residual` | `residual_fn` | `summed = x + gate * branch`, then the next sub-layer's input `LayerNorm(summed) * (1 + scale) + shift` (no affine, eps 1e-6); returns `(modulated, summed)` | 24 |
| `dit_mlp` | `mlp_fn` | linear, tanh-approximate GELU, linear | 12 |

`residual_fn` runs twice per block: after the attention (its input is the
block's MLP input) and after the MLP (its input is the next block's attention
input, or for the last block the final layer's). That is 23 calls of one
configuration (`blocks.residual`) and one of another (`final_residual`): the
final layer's shift and scale are halves of a `[B, 2H]` projection, row stride
2H instead of 6H. The first block's attention input, the LayerNorm and
modulation of the patch embedding, has no residual add before it and is not
part of any case. Level 1 binds observed `linear` and `spatial_conv2d` calls;
the diffusion loss remains outside the operator pool.

Measured Level-1/2 cases are derived from the whole-model harvest of the
verification run. A reduced configuration can verify execution but cannot
publish a snapshot.

## Harvest

The harvest runs the step the verification capture records first: the
verification workload (`evograd.verification.integrations.dit_s_2`: real
input, the streamed latent pool, the warm start), a fresh native model, the
first training batch of data order 101, under the verification's deterministic
settings (`integrations.base.real_run`, the step every harvest but MegaFold's
runs). Its snapshot (schema `/2`) records that run's identity under `run`, in
the one format every model shares (`base.run_identity`; see "Regenerating the
snapshots" in `src/evograd/verification/README.md`). Its `workload` names the
warm start by the SHA-256 of its file (`init_checkpoint_sha256`), not by its
environment-dependent path; the workload hash does not cover the warm start,
so a warm start regenerated after the harvest makes the snapshot command, and
every capture, refuse that harvest.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.benchmark.topdown.dit_s_2.harvest.harvest \
  --out results/dit_s_2/harvest.json --write-snapshot
```

`python -m evograd.benchmark.topdown.dit_s_2.harvest.snapshot --harvest
HARVEST.json [--write]` derives the snapshot from a saved harvest without a
GPU; it, too, accepts only a harvest of the verification run
(`base.require_real_run`). Reduced
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
