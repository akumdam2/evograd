"""Published Transformers architectures with explicit training inputs.

Random weights and synthetic inputs are deterministic benchmark fixtures, not
real-data training validation. Nothing reads the reference checkouts.
"""

import json
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
    kwargs = {"use_cache": False} if model_key == "qwen2_vl_2b" else {}
    return model(**batch, **kwargs)
