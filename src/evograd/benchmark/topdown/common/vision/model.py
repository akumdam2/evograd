"""Published Transformers architectures with explicit training inputs.

Random weights and synthetic inputs are deterministic benchmark fixtures, not
real-data training validation. Nothing reads the reference checkouts.
"""

import torch


def build_model(spec):
    import transformers
    from packaging.version import Version

    version = Version(transformers.__version__)
    if not (Version("4.57.0") <= version < Version("4.58.0")):
        raise RuntimeError("Vision workloads require transformers>=4.57,<4.58 (tested: 4.57.6)")
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        model = model_cls._from_config(config, dtype=getattr(torch, spec.dtype), **kwargs)
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
        }
    return {key: value.to(spec.device) for key, value in batch.items()}


def forward(model, batch, model_key):
    kwargs = {"use_cache": False} if model_key == "qwen2_vl_2b" else {}
    return model(**batch, **kwargs)
