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
| Benchmark spec (`canonical`) initialization | seeded random | seeded random, tied embedding/output weights retained |
| Harvest and verification checkpoint | pinned published safetensors | pinned published safetensors and tokenizer |
| Harvest and verification data | frozen pool of the Imagenette 320px train shard, labels mapped to ImageNet-1K | frozen pool of TextVQA train shard 0, image/question/answer teacher forcing, right-padded to T512 |

Both use CUDA, training mode, no cache, no gradient checkpointing, and the
existing Tier-3 AdamW step (`lr=1e-4`, betas 0.9/0.999, eps 1e-8, weight decay
0.01). Existing timing code and Tier-2 timing fields have not changed.

Architecture JSON and default benchmark JSON are shipped under each model's
`benchmark/topdown/<model>/` package. Model construction is offline. The
benchmark spec (`spec.canonical`) uses random weights; the harvest and the
verification workload load the pinned checkpoint and refuse to run until its
weights, processor and real-data pool are cached. Runtime model internals
require `transformers>=4.57` (tested on 4.57.6 and 5.16.1).

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

| Model | Observed L1 contracts | Patchable L2 sites (calls per step, canonical) |
|---|---|---|
| ConvNeXt | `spatial_conv2d`, `linear`, `layernorm`, `gelu`, `grn` | `dwconv_layernorm_stage0` … `stage3` → `convnextv2_tiny_dwconv_layernorm` (18); `convnext_mlp_stage0` … `stage3` → `convnextv2_tiny_convnext_mlp` (18) |
| Qwen2-VL | `linear`, `linear_no_bias`, `layernorm`, `rmsnorm`, `quick_gelu`, `gelu` | vision: `vision_attention` (32), `vision_mlp` (32), `vision_residual_layernorm` (64); text: `text_attention_block` (28), `text_mlp` (28), `text_residual_rmsnorm` (56); each → `qwen2_vl_2b_<site>` |

The Level-2 boundaries are the HF modules' own computation, recorded and
replayed bit for bit (`common/vision/operators.py`):

* `dwconv_layernorm`: `ConvNextV2Layer`'s 7×7 depthwise conv (bias) on the
  NCHW input, the permute to NHWC and its LayerNorm (eps 1e-6); output NHWC.
  Each `ConvNextV2Layer` holds this site and `convnext_mlp`; Tier 3 runs
  whichever of the two is not patched natively.
* `vision_attention`: `VisionAttention` (SDPA): biased qkv, the 2-D RoPE in
  float32 cast back (cos/sin are float32 inputs), non-causal SDPA over the one
  image's tokens (a single `cu_seqlens` segment; more images per call are
  refused), biased proj.
* `text_attention_block`: `Qwen2VLAttention` (SDPA): biased q/k/v, M-RoPE (the
  cos/sin tables after their section interleave), GQA SDPA with the boolean
  causal + padding mask the right-padded rows produce (KV heads repeated, no
  `is_causal`), biasless o_proj. An unpadded row (SDPA's `is_causal`, another
  computation) is refused; the synthetic fixture pads like the real pool.
* `vision_residual_layernorm` / `text_residual_rmsnorm`: each add of an
  attention or MLP branch to the residual stream, fused with the norm that
  follows it (`norm2`, the next block's `norm1`, `merger.ln_q`; Qwen2's
  `post_attention_layernorm`, the next `input_layernorm`, `model.norm`), two
  per block. They return `(normed, summed)`; `summed`'s cotangent is the next
  sum's gradient, zero (recorded as absent) for the last. Tier 3 rebinds the
  blocks so that each hands its second normalized sum to the next block or the
  final norm, with the sum itself flowing between blocks as before.

`spatial_conv2d` is a biased, square-kernel Conv2D contract supporting
stride, padding and groups (dilation 1).
GRN preserves its spatial reduction axes. Qwen2's vision activation is
**QuickGELU**, not GELU or SwiGLU. Epsilons and tensor strides are recorded in
observed cases. Row-based L1 contracts flatten leading axes; a copy is explicit
when the native layout cannot be represented as a matrix view. The manifest
retains original tensor metadata and records `flatten_leading_copy` versus
`flatten_leading_view`; isolated primitive timing excludes that adaptation.
L2 contracts keep their native ranks. Shape/frequency records are produced by an executed forward
and backward, then deduplicated by configuration; they are never invented
from parameter counts.

These are selected boundaries, not exhaustive primitive coverage. SDPA and
both RoPEs (inside the Qwen2-VL attention boundaries only), the Conv3D patch
embedding, loss, ConvNeXt's residual addition and pooling still execute in the
full model but are not separate L1 tasks. Each manifest lists its unbound
boundaries. L1/L2 configuration counts come from the CUDA harvest of the
verification run; the tracked snapshots predate the four Qwen2-VL boundaries
and `dwconv_layernorm`, which become cases at the next harvest.

## Install and harvest

The harvest runs the step the verification capture records first: the
verification workload (`evograd.verification.integrations.<model>`: the pinned
checkpoint and the frozen real-data pool), a fresh native model, the first
training batch of data order 101, under the verification's deterministic
settings (`integrations.base.real_run`, the step every harvest but MegaFold's
runs). Its snapshot (schema `/2`) records that run's identity under `run`, in
the format every model shares (`base.run_identity`: `model`, `workload` (the
weights and data pins), `data` (the data identity the capture records,
without its per-order batches), `order`, `batch` (the batch with its digest)
and `parameter_state` (the initial parameters with their digest), plus the
settings as `execution`); a capture refuses a snapshot whose `run` is not its
own. The checkpoint revision is pinned by the workload id and the parameter
digest. Qwen2-VL's pool rows
are right-padded, so its text attention gets an explicit mask rather than
SDPA's `is_causal`; `text_attention_masks` records the mask each text
attention call received.

```bash
pip install -e '.[vision_workloads,gpu,dev]'
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m evograd.benchmark.topdown.convnextv2_tiny.harvest.harvest \
  --out results/convnextv2_tiny/harvest.json --write-snapshot
python -m evograd.benchmark.topdown.qwen2_vl_2b.harvest.harvest \
  --out results/qwen2_vl_2b/harvest.json --write-snapshot
```

`python -m evograd.benchmark.topdown.<model>.harvest.snapshot --harvest
HARVEST.json [--write]` derives the snapshot from a saved harvest without a
GPU. Only a harvest of the verification run publishes a tracked snapshot
(`base.require_real_run`); the exact GPU commands and environment for every
model are in "Regenerating the snapshots" in `src/evograd/verification/README.md`. A
fresh process loads the new snapshots and binds those cases to each task under
`<model>_observed`; a workload whose `workload_id` differs from the snapshot's
uses explicitly labelled configuration-derived shapes for site verification,
so changing a workload cannot silently reuse another configuration's cases.
Harvest fails on missing/nonfinite parameter gradients or missing declared
primitive boundaries.

Reduced configurations (explicit architecture/device overrides, random
weights, synthetic inputs) run from Python with `collect(spec)` and never
write a tracked snapshot. Tests contain small configurations for both models.

## Pinned real data

Both forms use the published checkpoint and its processor. ConvNeXt's ten
Imagenette labels are mapped to the checkpoint's ImageNet-1K indices and
checked against its label names. Each Qwen2-VL image is resized to 224×224;
the chat template includes the real question, and only tokens of a
deterministically selected reference answer and `<|im_end|>` contribute to
loss.

* The frozen pool (`common/vision/stream.py`, harvest and verification): every
  row of the pinned shard (Qwen2-VL rows longer than T512 skipped and
  counted), processed once (`build_pool`, CPU), with a seeded permutation per
  data order and epoch.
* The prepared artifact (`common/vision/data.py`, a workload's `data_path`
  with pretrained weights): the first 320 Imagenette rows as ten B32 batches,
  or the first ten TextVQA rows (B1), with the model and dataset commits,
  selected row indices and processor revision recorded and validated for the
  fixed specialist shapes before use.

Neither proves correctness for other resolutions, questions or shapes.

Prepare the pinned assets once from the repository root. Each benchmark keeps
its Hugging Face cache in `data/<model>/hf_cache`, which the code reads by
default:

```bash
python scripts/prepare_benchmark_inputs.py --benchmark convnextv2_tiny
python scripts/prepare_benchmark_inputs.py --benchmark qwen2_vl_2b
```

## Optional prepared-data path

A workload's `data_path`, with random weights, accepts caller-supplied image
examples. It is separate from the pinned real data above.
Prepare local examples as JSON:

```json
[{"image": "images/cat.jpg", "label": 281}]
```

For Qwen2-VL use `{"image": "images/cat.jpg", "text": "A cat sitting on a chair."}`.
Image paths are relative to the JSON. Classification labels must use the
checkpoint's ImageNet class ordering.
The legacy preparation command may download the official processor/tokenizer.
It does not download a dataset or model weights. These caller-supplied batches
are not accepted with pretrained weights.

Example for ConvNeXt (substitute the Qwen2 model/config and sites as needed):

```bash
python -m evograd.benchmark.topdown.common.vision.data \
  --model convnextv2_tiny --examples data/imagenet_examples.json \
  --source 'ImageNet-1K train; record revision and sample selection here' \
  --out results/convnextv2_tiny/batches.pt
```

ConvNeXt has four independent stage sites so a specialist can target one
spatial/channel configuration; repeated layers within that stage share it.

## Validation status

Reduced CPU tests cover both full models, visual/merger gradients, tied weights,
structural identity, pair binding, stride replay and harvest coverage.
Canonical CUDA/BF16 harvests, real-data runs and timing
measurements have not been performed on the development Mac. Do not present
CPU fixtures or uncalibrated timing as paper results.
