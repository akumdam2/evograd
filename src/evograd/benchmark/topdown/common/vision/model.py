"""Published Transformers architectures with explicit training inputs.

Random weights and synthetic inputs are deterministic benchmark fixtures, not
real-data training validation. Nothing reads the reference checkouts.
"""

import json
import types
import weakref

import torch


def pretrained_snapshot(spec):
    """Resolve the exact published checkpoint from the local Hugging Face cache."""
    from huggingface_hub import snapshot_download
    from .spec import REVISIONS
    from evograd.benchmark.topdown.common.data_paths import hf_cache_dir

    try:
        location = snapshot_download(
            spec.model_name, revision=REVISIONS[spec.model],
            cache_dir=hf_cache_dir(spec.model),
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Pinned checkpoint {spec.model_name}@{REVISIONS[spec.model]} is not cached; "
            "download it before --real-input"
        ) from exc
    from pathlib import Path

    root = Path(location)
    weights = (
        ["model.safetensors"] if spec.model == "convnextv2_tiny" else
        ["model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    )
    required = ["config.json", "preprocessor_config.json", *weights]
    if spec.model == "qwen2_vl_2b":
        required += [
            "tokenizer.json", "tokenizer_config.json", "chat_template.json",
            "merges.txt", "vocab.json",
        ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Pinned checkpoint {spec.model_name}@{REVISIONS[spec.model]} is incomplete "
            f"in cache: {missing}. Fetch the files with `hf download {spec.model_name} "
            + " ".join(missing)
            + f" --revision {REVISIONS[spec.model]}` using the same cache."
        )
    return location


def _check_checkpoint_config(spec, location):
    from pathlib import Path

    published = json.loads((Path(location) / "config.json").read_text())
    declared = spec.arch
    ignored = {"architectures", "transformers_version", "torch_dtype", "use_cache", "id2label", "label2id"}
    mismatches = {
        key: (published.get(key), value)
        for key, value in declared.items()
        if key not in ignored and key in published and published[key] != value
    }
    if spec.model == "convnextv2_tiny":
        labels = len(published.get("id2label", {})) or published.get("num_labels")
        if labels != declared["num_labels"]:
            mismatches["num_labels"] = (labels, declared["num_labels"])
    if mismatches:
        raise RuntimeError(f"pinned checkpoint differs from workload architecture: {mismatches}")


def build_model(spec):
    import transformers
    from packaging.version import Version

    version = Version(transformers.__version__)
    if version < Version("4.57.0"):
        raise RuntimeError("Vision workloads require transformers>=4.57 (tested: 4.57.6, 5.16.1)")
    if spec.model == "convnextv2_tiny":
        config_cls = transformers.ConvNextV2Config
        model_cls = transformers.ConvNextV2ForImageClassification
    else:
        config_cls = transformers.Qwen2VLConfig
        model_cls = transformers.Qwen2VLForConditionalGeneration
    config = config_cls.from_dict(spec.arch)
    config.use_cache = False
    if hasattr(config, "text_config"):
        config.text_config.use_cache = False
    kwargs = (
        {}
        if spec.model == "convnextv2_tiny"
        else {"attn_implementation": spec.attn_implementation}
    )
    if spec.weights == "pretrained":
        location = pretrained_snapshot(spec)
        _check_checkpoint_config(spec, location)
        model = model_cls.from_pretrained(
            location, local_files_only=True, use_safetensors=True,
            dtype=getattr(torch, spec.dtype), **kwargs
        )
        if spec.model == "qwen2_vl_2b" and (
            model.lm_head.weight.data_ptr()
            != model.model.language_model.embed_tokens.weight.data_ptr()
        ):
            raise RuntimeError("checkpoint lost tied Qwen2-VL embeddings")
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            model = model_cls._from_config(config, dtype=getattr(torch, spec.dtype), **kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.to(spec.device).train()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if spec.model == "qwen2_vl_2b":
        _scatter_images_by_index(model.model)
    return model


def make_inputs(spec, seed=0):
    g = torch.Generator(device="cpu").manual_seed(spec.data_seed + seed)
    dtype = getattr(torch, spec.dtype)
    if spec.model == "convnextv2_tiny":
        batch = {
            "pixel_values": torch.randn(
                spec.batch_size, 3, spec.image_size, spec.image_size, generator=g, dtype=dtype
            ),
            "labels": torch.randint(spec.arch["num_labels"], (spec.batch_size,), generator=g),
        }
    else:
        a = spec.arch
        v = a["vision_config"]
        grid = spec.image_size // v["patch_size"]
        n_image = (grid // v["spatial_merge_size"]) ** 2
        # Reserve special token IDs. Real image/text datasets must be processed
        # with AutoProcessor; this fixture exercises the same packed-patch API.
        limit = min(
            a["vocab_size"],
            a["vision_start_token_id"],
            a["image_token_id"],
            a["vision_end_token_id"],
        )
        ids = torch.randint(3, limit, (spec.batch_size, spec.seq_len), generator=g)
        ids[:, 0] = a["vision_start_token_id"]
        ids[:, 1 : 1 + n_image] = a["image_token_id"]
        ids[:, 1 + n_image] = a["vision_end_token_id"]
        labels = ids.clone()
        labels[:, : 2 + n_image] = -100
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
            "pixel_values": torch.randn(
                spec.batch_size * grid * grid,
                v["in_chans"] * v["temporal_patch_size"] * v["patch_size"] ** 2,
                generator=g,
                dtype=dtype,
            ),
            "image_grid_thw": torch.tensor([[1, grid, grid]] * spec.batch_size),
            # Transformers 5 computes M-RoPE positions from the token modalities
            # (what its processor returns): 1 at image tokens, 0 elsewhere.
            "mm_token_type_ids": (ids == a["image_token_id"]).long(),
        }
    return {key: value.to(spec.device) for key, value in batch.items()}


def forward(model, batch, model_key):
    if model_key != "qwen2_vl_2b":
        return model(**batch)
    return model(**{**batch, **_host_inputs(model, batch)}, use_cache=False)


# ── Qwen2-VL without host reads ──────────────────────────────────────────────
#
# Qwen2-VL's forward reads tensor values back to the host: M-RoPE positions
# (``get_rope_index``: ``.tolist()``, ``.item()``), the vision encoder's patch
# positions and attention segments (``repeat_interleave``, ``.tolist()``), the
# image split sizes, the SDPA mask shortcut (``padding_mask.all()``), the
# placeholder count check, and ``masked_scatter``'s backward (``masked_select``).
# A CUDA graph cannot capture a step that does (T3-G), and an eager step stalls
# on each. All of them depend on the batch only, so they are computed once per
# batch, outside the step, with Transformers' own functions, and passed in the
# way Transformers accepts them precomputed: ``position_ids``, a prepared mask
# mapping, ``image_position_ids``/``image_cu_seqlens``, and ``image_grid_thw``
# on the host. ``masked_scatter`` becomes ``index_copy`` at the image rows: the
# same copy forward and backward, bitwise. Every execution takes this path.

_HOST_INPUTS = weakref.WeakKeyDictionary()
_BATCH_KEYS = ("input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids")


def _host_inputs(model, batch):
    """The batch's host-computed inputs, cached while the batch tensors are unchanged."""
    tensors = {key: batch[key] for key in _BATCH_KEYS if batch.get(key) is not None}
    cached = _HOST_INPUTS.get(model)
    if cached is not None and cached[0].keys() == tensors.keys() and all(
            ref() is tensors[key] and version == tensors[key]._version
            for key, (ref, version) in cached[0].items()):
        return cached[1]
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

    inner = model.model
    text = inner.language_model
    input_ids, grid = batch["input_ids"], batch["image_grid_thw"]
    attention_mask = batch.get("attention_mask")
    with torch.no_grad():
        position_ids, _ = inner.get_rope_index(
            input_ids, batch["mm_token_type_ids"], image_grid_thw=grid,
            attention_mask=attention_mask)
        # What Qwen2VLTextModel prepares for 3D positions (no packed text positions).
        mask_kwargs = {"config": text.config, "inputs_embeds": text.embed_tokens(input_ids),
                       "attention_mask": attention_mask, "past_key_values": None,
                       "position_ids": None}
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if text.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    values = {
        "position_ids": position_ids,
        "attention_mask": masks,
        "image_grid_thw": grid.cpu(),
        "image_position_ids": get_vision_position_ids(grid, inner.visual.spatial_merge_size),
        # On the host: the SDPA vision path splits by its lengths (``.tolist()``).
        "image_cu_seqlens": get_vision_cu_seqlens(grid).cpu(),
        "image_rows": (input_ids.reshape(-1) == inner.config.image_token_id).nonzero().squeeze(1),
    }
    _HOST_INPUTS[model] = ({key: (weakref.ref(t), t._version) for key, t in tensors.items()}, values)
    return values


def _scatter_images_by_index(inner):
    """Qwen2VLModel.forward with the image embeddings placed by ``index_copy``.

    Used when ``image_rows`` is given (:func:`forward`); any other call is the original.
    """
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModelOutputWithPast

    original = inner.forward

    def scatter_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        pixel_values=None, pixel_values_videos=None, image_grid_thw=None,
                        video_grid_thw=None, mm_token_type_ids=None, image_rows=None, **kwargs):
        if image_rows is None:
            return original(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache,
                            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                            mm_token_type_ids=mm_token_type_ids, **kwargs)
        if inputs_embeds is not None or pixel_values_videos is not None or position_ids is None:
            raise ValueError("the image-row path takes input ids, images and positions only")
        vision = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith("image_")}
        inputs_embeds = self.get_input_embeddings()(input_ids)
        images = self.get_image_features(pixel_values, image_grid_thw, **vision).pooler_output
        images = torch.cat(images, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        width = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(-1, width).index_copy(0, image_rows, images)
        outputs = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds.view(*input_ids.shape, width),
            use_cache=use_cache, **kwargs)
        return Qwen2VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            rope_deltas=self.rope_deltas)

    inner.forward = types.MethodType(scatter_forward, inner)
