"""Prepare fixed-shape vision batches from cached pinned or caller-supplied data.

Pinned mode reads local cached public dataset shards and the exact model
processor revision. Legacy JSON mode accepts {image, label} or {image, text}
rows with image paths relative to that JSON. Preparation never downloads a
dataset implicitly.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path
import torch
from .spec import canonical, VisionSpec, REVISIONS, DATASETS

# The dataset's ClassLabel order, mapped to the pinned classifier's ImageNet-1K order.
IMAGENETTE_CLASS_IDS = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
IMAGENETTE_CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)


def _local_dataset_file(spec):
    from huggingface_hub import hf_hub_download

    repo, revision, filename = DATASETS[spec.model]
    try:
        return hf_hub_download(
            repo, filename=filename, repo_type="dataset", revision=revision,
            cache_dir=os.environ.get("EVOGRAD_HF_CACHE") or None,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Pinned dataset shard {repo}@{revision}/{filename} is not cached; "
            "download it before --real-input"
        ) from exc


def _processor(spec):
    from transformers import AutoImageProcessor, AutoProcessor
    from .model import pretrained_snapshot

    location = pretrained_snapshot(spec)
    if spec.model == "convnextv2_tiny":
        return AutoImageProcessor.from_pretrained(location, local_files_only=True, use_fast=False)
    pixels = spec.image_size**2
    processor = AutoProcessor.from_pretrained(
        location, local_files_only=True, use_fast=False,
        min_pixels=pixels, max_pixels=pixels,
    )
    processor.tokenizer.padding_side = "right"
    return processor


def _answer(answers):
    from collections import Counter

    cleaned = [str(x).strip() for x in answers if str(x).strip()]
    if not cleaned:
        raise ValueError("TextVQA row has no reference answer")
    counts = Counter(cleaned)
    return min(counts, key=lambda x: (-counts[x], cleaned.index(x)))


def prepare_pinned(spec, *, batch_count=10):
    """Build deterministic training batches from cached, revision-pinned public data."""
    from datasets import load_dataset
    from PIL import Image

    if spec.weights != "pretrained" or spec.arch_overrides:
        raise ValueError("pinned data requires the published architecture and checkpoint")
    if spec.image_size != 224:
        raise ValueError("pinned real-input policy uses 224x224 images")
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    repo, revision, filename = DATASETS[spec.model]
    count = batch_count * spec.batch_size
    shard = _local_dataset_file(spec)
    dataset = load_dataset("parquet", data_files={"train": shard}, split=f"train[:{count}]")
    if len(dataset) != count:
        raise ValueError(f"dataset shard has {len(dataset)} rows, need {count}")
    processor = _processor(spec)
    if spec.model == "convnextv2_tiny":
        names = dataset.features["label"].names
        if tuple(names) != IMAGENETTE_CLASS_NAMES:
            raise ValueError(f"Imagenette ClassLabel order changed: {names}")
        from .model import pretrained_snapshot
        labels = json.loads((Path(pretrained_snapshot(spec)) / "config.json").read_text())["id2label"]
        for name, class_id in zip(IMAGENETTE_CLASS_NAMES, IMAGENETTE_CLASS_IDS):
            if not labels[str(class_id)].lower().startswith(name.lower()):
                raise ValueError(f"ImageNet class mapping changed at {class_id}")
    batches = []
    selected = []
    for start in range(0, count, spec.batch_size):
        rows = list(dataset.select(range(start, start + spec.batch_size)))
        images = [row["image"].convert("RGB") for row in rows]
        image_hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        if spec.model == "convnextv2_tiny":
            values = processor(images=images, return_tensors="pt")
            batch = {
                "pixel_values": values["pixel_values"],
                "labels": torch.tensor(
                    [IMAGENETTE_CLASS_IDS[int(row["label"])] for row in rows], dtype=torch.long
                ),
            }
            selected.extend(
                {"row": start + i, "label": int(row["label"]), "image_sha256": image_hashes[i]}
                for i, row in enumerate(rows)
            )
        else:
            if spec.batch_size != 1:
                raise ValueError("pinned TextVQA preparation uses the canonical batch size 1")
            row = rows[0]
            images = [images[0].resize((spec.image_size, spec.image_size), Image.Resampling.BICUBIC)]
            question = str(row["question"]).strip()
            answer = _answer(row["answers"])
            if not question:
                raise ValueError("TextVQA row has an empty question")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            target = prompt + answer + "<|im_end|>"
            prefix = processor(text=[prompt], images=images, return_tensors="pt")
            full = processor(text=[target], images=images, return_tensors="pt")
            prefix_len = prefix["input_ids"].shape[1]
            ids = full["input_ids"]
            if ids.shape[1] > spec.seq_len or not torch.equal(
                ids[:, :prefix_len], prefix["input_ids"]
            ):
                raise ValueError(f"TextVQA row {start} exceeds or changes the fixed prompt contract")
            labels = torch.full((1, spec.seq_len), -100, dtype=torch.long)
            labels[:, prefix_len : ids.shape[1]] = ids[:, prefix_len:]
            batch = {
                "input_ids": torch.nn.functional.pad(
                    ids, (0, spec.seq_len - ids.shape[1]), value=processor.tokenizer.pad_token_id
                ),
                "attention_mask": torch.nn.functional.pad(
                    full["attention_mask"], (0, spec.seq_len - ids.shape[1])
                ),
                "labels": labels,
                "pixel_values": full["pixel_values"],
                "image_grid_thw": full["image_grid_thw"],
            }
            selected.append({
                "row": start, "question_id": int(row["question_id"]),
                "answer": answer, "answer_start": prefix_len,
                "answer_end": ids.shape[1], "image_sha256": image_hashes[0],
            })
        batch["pixel_values"] = batch["pixel_values"].to(getattr(torch, spec.dtype))
        batches.append(batch)
    return {
        "schema_version": "evograd-vision-data/2", "model": spec.model,
        "source": f"{repo}@{revision}/{filename}:train[:{count}]",
        "dataset_repo": repo, "dataset_revision": revision, "dataset_file": filename,
        "processor": spec.model_name, "processor_revision": REVISIONS[spec.model],
        "selection": selected, "examples": count, "batches": batches,
    }


def default_data_path(spec):
    root = Path(os.environ.get("EVOGRAD_VISION_DATA_CACHE", Path.home() / ".cache/evograd/vision"))
    return root / spec.model / f"{spec.workload_hash}-pinned.pt"


def ensure_pinned_data(spec, path=None, *, batch_count=10):
    path = Path(path) if path else default_data_path(spec)
    if not path.exists():
        data = prepare_pinned(spec, batch_count=batch_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(data, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    from types import SimpleNamespace
    load_data(SimpleNamespace(spec=spec, data_path=str(path)))
    return str(path)


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
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--pinned", action="store_true", help="use cached pinned public dataset")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument(
        "--source", help="Dataset name, split, revision and sample selection"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = canonical(args.model).to_config()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    config["weights"] = "random"
    spec = VisionSpec(**config)
    if args.pinned:
        if args.examples or args.source:
            parser.error("--pinned cannot be combined with --examples or --source")
        spec = spec.replace(weights="pretrained")
        data = prepare_pinned(spec, batch_count=args.batches)
    else:
        if not args.examples or not args.source:
            parser.error("--examples and --source are required without --pinned")
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
    real = workload.spec.weights == "pretrained"
    if (
        data.get("schema_version") != ("evograd-vision-data/2" if real else "evograd-vision-data/1")
        or data.get("model") != workload.spec.model
        or not data.get("source")
    ):
        raise ValueError("data artifact requires schema, matching model and dataset source")
    if real:
        repo, revision, filename = DATASETS[workload.spec.model]
        if (
            data.get("dataset_repo") != repo
            or data.get("dataset_revision") != revision
            or data.get("dataset_file") != filename
            or data.get("processor") != workload.spec.model_name
            or data.get("processor_revision") != REVISIONS[workload.spec.model]
            or data.get("examples") != len(data["batches"]) * workload.spec.batch_size
            or len(data.get("selection", [])) != data["examples"]
        ):
            raise ValueError("real-input artifact does not match pinned dataset and processor")
    from .model import make_inputs

    expected = make_inputs(workload.spec.replace(device="cpu"))
    shapes = {k: tuple(v.shape) for k, v in expected.items()}
    if not data["batches"]:
        raise ValueError("empty data artifact")
    for batch_index, batch in enumerate(data["batches"]):
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
            if real:
                selection = data["selection"][batch_index]
                start, end = selection["answer_start"], selection["answer_end"]
                if not (0 < start < end <= workload.spec.seq_len):
                    raise ValueError("invalid TextVQA answer span")
                if not batch["labels"][:, :start].eq(-100).all() or not batch["labels"][:, end:].eq(-100).all():
                    raise ValueError("TextVQA prompt or padding is supervised")
                if not torch.equal(batch["labels"][:, start:end], batch["input_ids"][:, start:end]):
                    raise ValueError("TextVQA answer labels differ from input tokens")
        elif real:
            selection = data["selection"][batch_index * workload.spec.batch_size : (batch_index + 1) * workload.spec.batch_size]
            expected_labels = torch.tensor([IMAGENETTE_CLASS_IDS[row["label"]] for row in selection])
            if not torch.equal(batch["labels"], expected_labels):
                raise ValueError("Imagenette labels differ from ImageNet-1K mapping")
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
