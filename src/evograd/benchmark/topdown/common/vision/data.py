"""Prepare fixed-shape batches from user-supplied image datasets.

Input JSON is a list of {image, label} (classification) or {image, text}
(image-conditioned next-token prediction). Image paths are relative to that
JSON. Dataset provenance is required. No training dataset is downloaded here.
"""

import hashlib
import argparse
import json
from pathlib import Path
import torch
from .spec import canonical, VisionSpec, REVISIONS


def prepare(spec, examples_path, source):
    from PIL import Image
    from transformers import AutoImageProcessor, AutoProcessor

    if spec.arch_overrides:
        raise ValueError("real-data preparation requires the published architecture")
    examples_path = Path(examples_path)
    examples = json.loads(examples_path.read_text())
    if not source or not examples or len(examples) % spec.batch_size:
        raise ValueError("source and a nonempty whole number of batches are required")
    if spec.model == "convnextv2_tiny":
        processor = AutoImageProcessor.from_pretrained(
            spec.model_name, revision=REVISIONS[spec.model], use_fast=False
        )
    else:
        pixels = spec.image_size**2
        processor = AutoProcessor.from_pretrained(
            spec.model_name,
            revision=REVISIONS[spec.model],
            use_fast=False,
            min_pixels=pixels,
            max_pixels=pixels,
        )
        processor.tokenizer.padding_side = "right"
    batches = []
    for start in range(0, len(examples), spec.batch_size):
        rows = examples[start : start + spec.batch_size]
        images = []
        for row in rows:
            with Image.open(examples_path.parent / row["image"]) as image:
                images.append(image.convert("RGB"))
        if spec.model == "convnextv2_tiny":
            values = processor(
                images=images, return_tensors="pt", size={"shortest_edge": spec.image_size}
            )
            batch = {
                "pixel_values": values["pixel_values"],
                "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
            }
        else:
            # A fixed image prefix plus caption text, with image/padding targets
            # ignored. This is caption next-token training, not an SFT recipe.
            texts = [
                "<|vision_start|><|image_pad|><|vision_end|>"
                + row["text"]
                + processor.tokenizer.eos_token
                for row in rows
            ]
            values = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding="max_length",
                max_length=spec.seq_len,
                truncation=True,
            )
            batch = {
                k: values[k]
                for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            }
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            for token in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
                labels[labels == spec.arch[token]] = -100
            batch["labels"] = labels
        batch["pixel_values"] = batch["pixel_values"].to(getattr(torch, spec.dtype))
        batches.append(batch)
    return {
        "schema_version": "evograd-vision-data/1",
        "model": spec.model,
        "source": source,
        "processor": spec.model_name,
        "examples": len(examples),
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("convnextv2_tiny", "qwen2_vl_2b"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--source", required=True, help="Dataset name, split, revision and sample selection"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = canonical(args.model).to_config()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    spec = VisionSpec(**config)
    data = prepare(spec, args.examples, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    # Validate the artifact against the same contract the model gate uses.
    from types import SimpleNamespace

    load_data(SimpleNamespace(spec=spec, data_path=str(args.out)))
    print(f'Prepared {len(data["batches"])} batches at {args.out}')


def load_data(workload):
    if not workload.data_path:
        raise ValueError("real-data batch artifact required; synthetic fixtures cannot pass Gate 3")
    path = Path(workload.data_path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (
        data.get("schema_version") != "evograd-vision-data/1"
        or data.get("model") != workload.spec.model
        or not data.get("source")
    ):
        raise ValueError("data artifact requires schema, matching model and dataset source")
    from .model import make_inputs

    expected = make_inputs(workload.spec.replace(device="cpu"))
    shapes = {k: tuple(v.shape) for k, v in expected.items()}
    if not data["batches"]:
        raise ValueError("empty data artifact")
    for batch in data["batches"]:
        if set(batch) != set(shapes) or any(tuple(batch[k].shape) != s for k, s in shapes.items()):
            raise ValueError("real-data batch shapes differ from the specialist execution contract")
        if any(batch[k].dtype != expected[k].dtype for k in batch):
            raise ValueError("batch tensor dtypes differ from the execution contract")
        if workload.spec.model == "qwen2_vl_2b":
            a = workload.spec.arch
            n_image = (
                workload.spec.image_size
                // (a["vision_config"]["patch_size"] * a["vision_config"]["spatial_merge_size"])
            ) ** 2
            if not (batch["input_ids"].eq(a["image_token_id"]).sum(-1) == n_image).all():
                raise ValueError(
                    "each example must have exactly one image with the declared token count"
                )
            if not torch.equal(batch["image_grid_thw"], expected["image_grid_thw"]):
                raise ValueError("image grid differs from the declared contract")
            if (batch["labels"][:, 1:] == -100).all():
                raise ValueError("no supervised shifted text labels")
        if not all(torch.isfinite(t).all() for t in batch.values()):
            raise ValueError("nonfinite data")
        if (batch["labels"] == -100).all():
            raise ValueError("batch has no supervised labels")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return data, h.hexdigest()


if __name__ == "__main__":
    main()
