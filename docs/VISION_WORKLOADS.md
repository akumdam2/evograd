# ConvNeXt V2 Tiny and Qwen2-VL-2B

These are **whole-model Level-3 training workloads**, with harvested Level-1
primitives and selected Level-2 fusion sites.
Neither model imports from `references/`. Removing that directory is safe.

## Architecture sources and execution contracts

| | ConvNeXt V2 Tiny | Qwen2-VL-2B |
|---|---|---|
| Published model | `facebook/convnextv2-tiny-1k-224` | `Qwen/Qwen2-VL-2B-Instruct` |
| Unique parameters | 28,635,496 | 2,208,985,600 |
| PyTorch implementation | `transformers.ConvNextV2ForImageClassification` | `transformers.Qwen2VLForConditionalGeneration` |
| Structure | depths 3/3/9/3, widths 96/192/384/768, 1000 labels | 28 text layers, hidden 1536, intermediate 8960, 12 query/2 KV heads; 32 vision layers, width 1280, 16 heads |
| Canonical input | B32, RGB 224×224 | B1, T512 total positions, one RGB 224×224 image |
| Image packing | NCHW | 256 patches × 1176 values; grid `[1,16,16]`; 64 merged image tokens |
| Precision / attention | BF16 | BF16 / SDPA, both vision and text |
| Loss | image classification cross entropy | image-conditioned next-token cross entropy; image markers and padding are ignored targets |
| Trainable parameters | all | all, including vision encoder and merger |
| Default initialization | seeded random | seeded random, tied embedding/output weights retained |
| `--real-input` checkpoint | pinned published safetensors | pinned published safetensors and tokenizer |
| `--real-input` dataset | Imagenette 320px train shard, labels mapped to ImageNet-1K | TextVQA train shard, image/question/answer teacher forcing |

Both use CUDA, training mode, no cache, no gradient checkpointing, and the
existing Tier-3 AdamW step (`lr=1e-4`, betas 0.9/0.999, eps 1e-8, weight decay
0.01). Existing timing code and Tier-2 timing fields have not changed.

Architecture JSON and default benchmark JSON are shipped under each model's
`benchmark/topdown/<model>/` package. Model construction is offline. The default
uses random weights; `--real-input` loads the pinned checkpoint and refuses to
run until its weights, processor and dataset shard are cached. Runtime model internals are supported on
`transformers>=4.57,<4.58`, tested on **4.57.6**. Use a separate environment
from workloads requiring Transformers 5.x.

Pinned configuration sources:

- [ConvNeXt config](https://huggingface.co/facebook/convnextv2-tiny-1k-224/blob/f4db009e63145e02b3c075aa64d90ce41bcca4b1/config.json).
  Label-name dictionaries are omitted locally; `num_labels=1000` is explicit.
- [Qwen2-VL config](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/895c3a49bc3fa70a340399125c650a463535e71c/config.json).
- [ConvNeXt PyTorch implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/convnextv2/modeling_convnextv2.py).
- [Qwen2-VL PyTorch implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py).

### Source training recipes versus benchmark settings

The ConvNeXt reference's `TRAINING.md` gives Tiny ImageNet-1K fine-tuning:
300 epochs, warmup 40, per-device batch 32, base LR 8e-4, layer decay 0.9,
weight decay 0.05, drop path 0.2, random erasing 0.25, mixup 0.8, cutmix 1.0,
smoothing 0.1, EMA, AMP, and an FCMAE checkpoint. The benchmark follows the
published classifier architecture and shared timing protocol. It does **not**
reproduce that complete accuracy-training experiment: no augmentation, EMA,
mixup, pretraining or distributed training is timed. The checkpoint config uses
`drop_path_rate=0`; its final LayerNorm epsilon is 1e-12, while the original
reference final norm uses 1e-6. The checkpoint's HF implementation/config is
our explicit authority for those choices.

The supplied `Qwen3-VL` checkout is not Qwen2-VL. Its scripts target Qwen2.5-VL
and Qwen3-VL, often freezing vision. Those scripts are not used as a Qwen2-VL
recipe. Our fixed image/text lengths and full-parameter training are benchmark
choices. This workload covers one image per sample, not video or arbitrary
image resolutions. Changing batch size, sequence length or image grid defines
a different specialist contract and requires a new harvest/calibration.

## Coverage

| Model | Observed L1 contracts | Patchable L2 sites |
|---|---|---|
| ConvNeXt | `spatial_conv2d`, `linear`, `layernorm`, `gelu`, `grn` | `convnext_mlp_stage0` … `stage3` → `convnextv2_tiny_convnext_mlp` |
| Qwen2-VL | `linear`, `linear_no_bias`, `layernorm`, `rmsnorm`, `quick_gelu`, `gelu` | `text_mlp` → `qwen2_vl_2b_text_mlp`; `vision_mlp` → `qwen2_vl_2b_vision_mlp` |

`spatial_conv2d` is a new biased, square-kernel Conv2D contract supporting
stride, padding and groups (dilation 1). The older `conv2d` contract is unchanged.
GRN preserves its spatial reduction axes. Qwen2's vision activation is
**QuickGELU**, not GELU or SwiGLU. Epsilons and tensor strides are recorded in
observed cases. Row-based L1 contracts flatten leading axes; a copy is explicit
when the native layout cannot be represented as a matrix view. The manifest
retains original tensor metadata and records `flatten_leading_copy` versus
`flatten_leading_view`; isolated primitive timing excludes that adaptation.
L2 contracts keep their native ranks. Shape/frequency records are produced by an executed forward
and backward, then deduplicated by configuration; they are never invented
from parameter counts.

These are selected boundaries, not exhaustive primitive coverage. Attention,
multimodal RoPE, the Conv3D patch embedding, loss, residual addition and pooling
still execute in the full model but are not all separate L1/L2 tasks. Each
manifest lists its unbound boundaries. L1/L2 configuration counts remain
unreported until a canonical CUDA harvest exists.

## Install and harvest

```bash
pip install -e '.[vision_workloads,gpu,dev]'
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m evograd.benchmark.topdown.convnextv2_tiny.harvest.harvest \
  --out results/convnextv2_tiny/harvest.json --write-snapshot
python -m evograd.benchmark.topdown.qwen2_vl_2b.harvest.harvest \
  --out results/qwen2_vl_2b/harvest.json --write-snapshot
```

Only a canonical run can publish a tracked snapshot. Before that, the model
adapters and reduced correctness grids are available, but observed benchmark
suites are empty. Site verification still uses explicitly labelled
configuration-derived shapes; changing a workload cannot silently reuse another
configuration’s verification cases. A fresh process loads the new snapshots and binds those
cases to each task under `<model>_observed`. Harvest fails on missing/nonfinite
parameter gradients or missing declared primitive boundaries.

`--config file.json` accepts explicit reduced architecture/device overrides for
smoke harvests. Such runs are marked noncanonical and cannot write a tracked
snapshot. Tests contain small configurations for both models.

## Generated-input verification

By default, both workloads use seeded Gaussian pixels/patches, random tokens
and random initial weights. The whole-model gate now calibrates and evaluates
on disjoint generated batches when `--data` is omitted. This verifies the
declared execution contract on those batches, including loss, logits, input
and parameter gradients, and a short training trajectory. It does not establish
real-data training validity.

```bash
python -m evograd.evaluation.tier3.workloads.common.vision.gate \
  --config src/evograd/benchmark/topdown/convnextv2_tiny/benchmark.json \
  --sites convnext_mlp_stage0,convnext_mlp_stage1,convnext_mlp_stage2,convnext_mlp_stage3 \
  --steps 5 --out results/convnextv2_tiny/synthetic_calibration.json
python -m evograd.evaluation.tier3.cli --model convnextv2_tiny \
  --calibration results/convnextv2_tiny/synthetic_calibration.json \
  --structural-identity --providers eager,structural_identity \
  --out results/convnextv2_tiny/synthetic_identity.json
```

For Qwen2-VL, use its benchmark JSON, `--sites text_mlp,vision_mlp`, and
`--model qwen2_vl_2b`. The calibration and evaluation require substantial GPU
memory and temporary disk space at the canonical Qwen2-VL size.

## Pinned real input

`--real-input` changes **both** the weights and inputs. ConvNeXt uses the
published 1,000-class checkpoint, official image processor, and the first 320
rows of the revision-pinned Imagenette 320px training shard for the default
five calibration plus five holdout steps (B32). Its ten dataset labels are
mapped to the checkpoint's ImageNet-1K indices and checked against its label
names. Qwen2-VL uses the published checkpoint and processor, with the first
ten rows of the pinned TextVQA training shard (B1). Each image is resized to
224×224; the chat template includes the real question, and only tokens of a
deterministically selected reference answer and `<|im_end|>` contribute to
loss. Both run the existing differentiable full-model training step.

The model and dataset commits, selected row indices, processor revision and
prepared tensor artifact are recorded. The artifact is validated for the fixed
specialist shapes before calibration or evaluation. Calibration and evaluation
use disjoint batches from that one artifact. This checks the declared inputs;
it does not prove correctness for other resolutions, questions or shapes.

Prepare the pinned assets once from the repository root. Each benchmark keeps
its Hugging Face cache in `data/<model>/hf_cache`, which the CLI reads by
default:

```bash
python scripts/prepare_benchmark_inputs.py --benchmark convnextv2_tiny
python scripts/prepare_benchmark_inputs.py --benchmark qwen2_vl_2b
```

Then calibrate and run ConvNeXt:

```bash
python -m evograd.evaluation.tier3.workloads.common.vision.gate \
  --config src/evograd/benchmark/topdown/convnextv2_tiny/benchmark.json \
  --real-input --sites convnext_mlp_stage0,convnext_mlp_stage1,convnext_mlp_stage2,convnext_mlp_stage3 \
  --steps 5 --out results/convnextv2_tiny/real_calibration.json
python -m evograd.evaluation.tier3.cli --model convnextv2_tiny --real-input \
  --calibration results/convnextv2_tiny/real_calibration.json \
  --structural-identity --providers eager,structural_identity \
  --out results/convnextv2_tiny/real_identity.json
```

For Qwen2-VL, substitute its benchmark JSON, `--sites text_mlp,vision_mlp`,
`--model qwen2_vl_2b`, and its own output paths. The flag alone prepares and
caches the pinned batches at `data/<model>/batches/`. Use
`--data-path` for the Tier-3 CLI or `--data` for the calibration CLI to select
an already prepared pinned artifact explicitly. Keep the same artifact for
calibration and evaluation.

## Optional prepared-data path

The existing `--data-path` path without `--real-input` accepts caller-supplied
image examples while keeping random weights. It is separate from the pinned
real-input mode.
Prepare local examples as JSON:

```json
[{"image": "images/cat.jpg", "label": 281}]
```

For Qwen2-VL use `{"image": "images/cat.jpg", "text": "A cat sitting on a chair."}`.
Image paths are relative to the JSON. Classification labels must use the
checkpoint's ImageNet class ordering. Use at least **10 complete batches** for
the default calibration (5 calibration updates + 5 disjoint holdout updates).
The legacy preparation command may download the official processor/tokenizer.
It does not download a dataset or model weights. These caller-supplied batches
are not accepted under `--real-input`.

Example for ConvNeXt (substitute the Qwen2 model/config and sites as needed):

```bash
python -m evograd.benchmark.topdown.common.vision.data \
  --model convnextv2_tiny --examples data/imagenet_examples.json \
  --source 'ImageNet-1K train; record revision and sample selection here' \
  --out results/convnextv2_tiny/batches.pt
python -m evograd.evaluation.tier3.workloads.common.vision.gate \
  --config src/evograd/benchmark/topdown/convnextv2_tiny/benchmark.json \
  --data results/convnextv2_tiny/batches.pt --sites convnext_mlp_stage0,convnext_mlp_stage1,convnext_mlp_stage2,convnext_mlp_stage3 --steps 5 \
  --out results/convnextv2_tiny/calibration.json
python -m evograd.evaluation.tier3.cli --model convnextv2_tiny \
  --data-path results/convnextv2_tiny/batches.pt \
  --calibration results/convnextv2_tiny/calibration.json \
  --structural-identity --providers eager,structural_identity \
  --out results/convnextv2_tiny/identity.json
```

ConvNeXt has four independent stage sites so a specialist can target one
spatial/channel configuration; repeated layers within that stage share it.
For Qwen2 calibrate `--sites text_mlp,vision_mlp` for a provider replacing both.
A single-site provider needs calibration for that exact site set. Candidates
use the existing `--candidate SITE=PATH` / `--patch-set` mechanisms; pair
contracts are bound through the existing autograd deployment wrapper.

Calibration independently rebuilds native and same-site `torch.compile`
controls twice and requires bitwise repeatability of each. Budgets are frozen
per loss/logit/input-gradient/parameter-gradient tensor from native–compiled
relative L2 and max-absolute/reference-RMS discrepancies. The fixed policy is
`2 * max(control discrepancy, 1e-5)`. Every held-out tensor must meet both
limits. Independent AdamW trajectories use all prescribed steps; loss-difference
RMS and maximum absolute difference must both meet separately frozen budgets.
Calibration is tied to configuration, hardware/software, data hash and sites.
A missing calibration, resource error, nonfinite tensor or missing gradient
fails the gate; no candidate is used to choose budgets.

The calibration process uses disk-backed tensor references and can require
substantial temporary storage for Qwen2. Use a suitable `TMPDIR`. Dataset
provenance is supplied by the caller and must be reviewed before publication;
checking an artifact's hash cannot establish that its examples are real.

## Validation status

Reduced CPU tests cover both full models, visual/merger gradients, tied weights,
structural identity, pair binding, stride replay, harvest coverage, calibration
with the diagnostic `aot_eager` backend, and rejection of a corrupted candidate.
Canonical CUDA/BF16 harvests, Inductor calibration, real-data runs and timing
measurements have not been performed on the development Mac. The default
operator tolerances are provisional until GPU qualification. Do not present
CPU fixtures or uncalibrated timing as paper results.
