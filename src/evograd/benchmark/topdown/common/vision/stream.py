"""Real training data per data order, for verification (no torchvision, no ``datasets``).

Each model's pinned shard is processed once into a frozen pool:

* ConvNeXt V2: every Imagenette 320px training image through the pinned
  checkpoint's image processor (PIL backend), ``pixel_values`` in the model
  dtype, labels mapped to ImageNet-1K ids;
* Qwen2-VL: every TextVQA shard-0 row as the pinned processor would make it:
  the image at 224x224 through the PIL image processor, the chat template's
  ``<|image_pad|>`` expanded to ``grid / merge**2`` tokens, the answer plus
  ``<|im_end|>`` as the only labelled tokens, padded to ``seq_len``, and
  ``mm_token_type_ids`` marking the image tokens (1) -- the processor class
  itself needs torchvision for its video half, so its text step is spelled
  out here. Rows longer than ``seq_len`` are skipped and counted.

A data order permutes the pool per epoch under a seed named by
``(model, order, epoch)``; batch ``i`` is the ``i``-th slice of that epoch.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import torch

POOL_SCHEMA = "evograd-vision-pool/1"
STREAM_RULE = ("data order o, batch i: epoch e = i // (n // B) is torch.randperm(n) under the seed "
               "of '<model>/order<o>/epoch<e>'; the batch is its rows [(i mod (n // B)) * B, +B)")
POOL_RULES = {
    "convnextv2_tiny": "every Imagenette 320px train row: RGB, ConvNextImageProcessorPil of the "
                       "pinned checkpoint, pixel_values cast to the model dtype; label -> ImageNet id",
    "qwen2_vl_2b": "every TextVQA shard-0 row: image RGB resized to 224x224 (bicubic), "
                   "Qwen2VLImageProcessorPil (224*224 pixels); chat template with one image and the "
                   "question, <|image_pad|> expanded to grid.prod() // merge**2; target = prompt + "
                   "most common answer + <|im_end|>; labels on the answer and <|im_end|> only; right "
                   "padded to seq_len; mm_token_type_ids = 1 at image tokens; longer rows skipped",
}


def seed_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & (2**63 - 1)


def default_pool_path(spec) -> Path:
    from evograd.benchmark.topdown.common.data_paths import prepared_dir

    override = os.environ.get(f"EVOGRAD_VISION_POOL_{spec.model.upper()}")
    return Path(override) if override else Path(prepared_dir(spec.model)) / f"{spec.model}_pool.pt"


def _table(spec):
    import pyarrow.parquet as pq

    from .data import _local_dataset_file

    return pq.read_table(_local_dataset_file(spec))


def _convnext_pool(spec):
    import numpy as np
    from PIL import Image
    from transformers.models.convnext.image_processing_pil_convnext import ConvNextImageProcessorPil

    from .data import IMAGENETTE_CLASS_IDS, IMAGENETTE_CLASS_NAMES
    from .model import pretrained_snapshot

    table = _table(spec)
    features = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
    names = tuple(features.get("info", {}).get("features", {}).get("label", {}).get("names", ()))
    if names != IMAGENETTE_CLASS_NAMES:
        raise ValueError(f"Imagenette ClassLabel order changed: {names}")
    location = pretrained_snapshot(spec)
    id2label = json.loads((Path(location) / "config.json").read_text())["id2label"]
    for name, class_id in zip(IMAGENETTE_CLASS_NAMES, IMAGENETTE_CLASS_IDS):
        if not id2label[str(class_id)].lower().startswith(name.lower()):
            raise ValueError(f"ImageNet class mapping changed at {class_id}")
    processor = ConvNextImageProcessorPil.from_pretrained(location, local_files_only=True)
    images, labels = table.column("image").to_pylist(), table.column("label").to_pylist()
    pixels, selection = [], []
    for start in range(0, len(images), 64):
        chunk = [Image.open(io.BytesIO(item["bytes"])).convert("RGB") for item in images[start:start + 64]]
        values = processor(images=chunk, return_tensors="pt")["pixel_values"]
        pixels.append(values.to(getattr(torch, spec.dtype)))
        selection += [{"row": start + i, "label": int(labels[start + i]),
                       "image_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}
                      for i, image in enumerate(chunk)]
    return {"pixel_values": torch.cat(pixels),
            "labels": torch.tensor([IMAGENETTE_CLASS_IDS[s["label"]] for s in selection])}, selection


def _qwen2_vl_pool(spec):
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    from .data import _answer
    from .model import pretrained_snapshot

    location = pretrained_snapshot(spec)
    pixels = spec.image_size ** 2
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(location, local_files_only=True,
                                                               min_pixels=pixels, max_pixels=pixels)
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    template = json.loads((Path(location) / "chat_template.json").read_text())["chat_template"]
    a = spec.arch
    image_token, image_id = "<|image_pad|>", a["image_token_id"]
    merge = image_processor.merge_size ** 2
    table = _table(spec)
    columns = {name: table.column(name).to_pylist() for name in ("image", "question", "answers",
                                                                  "question_id")}
    fields = {k: [] for k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids",
                              "pixel_values", "image_grid_thw")}
    selection, skipped = [], []
    for row in range(len(columns["question"])):
        question = str(columns["question"][row]).strip()
        if not question:
            skipped.append({"row": row, "reason": "empty question"})
            continue
        answer = _answer(columns["answers"][row])
        image = Image.open(io.BytesIO(columns["image"][row]["bytes"])).convert("RGB")
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        image = image.resize((spec.image_size, spec.image_size), Image.Resampling.BICUBIC)
        vision = image_processor(images=[image], return_tensors="pt")
        tokens = int(vision["image_grid_thw"][0].prod()) // merge
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False,
                                               add_generation_prompt=True)
        prompt = prompt.replace(image_token, image_token * tokens, 1)
        prefix = tokenizer(prompt, return_tensors="pt")["input_ids"]
        ids = tokenizer(prompt + answer + "<|im_end|>", return_tensors="pt")["input_ids"]
        if ids.shape[1] > spec.seq_len or not torch.equal(ids[:, :prefix.shape[1]], prefix):
            skipped.append({"row": row, "reason": "longer than seq_len or prompt changed"})
            continue
        pad = spec.seq_len - ids.shape[1]
        labels = torch.full((1, spec.seq_len), -100, dtype=torch.long)
        labels[:, prefix.shape[1]:ids.shape[1]] = ids[:, prefix.shape[1]:]
        padded = torch.nn.functional.pad(ids, (0, pad), value=tokenizer.pad_token_id)
        fields["input_ids"].append(padded)
        fields["attention_mask"].append(torch.nn.functional.pad(torch.ones_like(ids), (0, pad)))
        fields["labels"].append(labels)
        fields["mm_token_type_ids"].append((padded == image_id).long())
        fields["pixel_values"].append(vision["pixel_values"].to(getattr(torch, spec.dtype)))
        fields["image_grid_thw"].append(vision["image_grid_thw"])
        selection.append({"row": row, "question_id": int(columns["question_id"][row]), "answer": answer,
                          "answer_start": int(prefix.shape[1]), "answer_end": int(ids.shape[1]),
                          "image_sha256": image_hash})
    if len({tuple(p.shape) for p in fields["pixel_values"]}) != 1:
        raise ValueError("TextVQA rows produced differently shaped patch grids")
    # ``pixel_values`` is each row's packed patch matrix (no batch dimension);
    # every other field carries the processor's batch dimension of one.
    return {k: torch.stack([v if k == "pixel_values" else v[0] for v in values])
            for k, values in fields.items()}, {"selection": selection, "skipped": skipped}


def build_pool(spec, path=None) -> str:
    """Process the pinned shard once into the pool file (atomic). CPU only."""
    from .spec import DATASETS, REVISIONS

    if spec.weights != "pretrained" or spec.arch_overrides:
        raise ValueError("a real-data pool needs the published architecture and checkpoint")
    path = Path(path) if path is not None else default_pool_path(spec)
    if spec.model == "convnextv2_tiny":
        tensors, selection = _convnext_pool(spec)
        extra = {"selection": selection}
    else:
        tensors, extra = _qwen2_vl_pool(spec)
    pool = {"schema": POOL_SCHEMA, "model": spec.model, "dataset": list(DATASETS[spec.model]),
            "checkpoint_revision": REVISIONS[spec.model], "dtype": spec.dtype,
            "image_size": spec.image_size, "seq_len": getattr(spec, "seq_len", None),
            "rule": POOL_RULES[spec.model], "tensors": tensors, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(pool, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


_POOLS: dict[str, tuple[dict, str]] = {}


def load_pool(spec, path=None):
    from .spec import DATASETS, REVISIONS

    path = Path(path) if path is not None else default_pool_path(spec)
    key = str(path.resolve())
    if key not in _POOLS:
        if not path.is_file():
            raise FileNotFoundError(f"no {spec.model} real-data pool at {path}; build it with "
                                    "evograd.benchmark.topdown.common.vision.stream.build_pool")
        pool = torch.load(path, map_location="cpu", weights_only=True)
        if (pool.get("schema") != POOL_SCHEMA or pool.get("model") != spec.model
                or tuple(pool.get("dataset", ())) != DATASETS[spec.model]
                or pool.get("checkpoint_revision") != REVISIONS[spec.model]
                or pool.get("dtype") != spec.dtype or pool.get("rule") != POOL_RULES[spec.model]):
            raise ValueError(f"{path} is not a {spec.model} pool of the pinned contract")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        _POOLS[key] = (pool, digest.hexdigest())
    return _POOLS[key]


class PoolStream:
    """The batch sequence of one data order over a model's pool."""

    def __init__(self, spec, order, path=None):
        self.spec, self.order = spec, int(order)
        self.pool, self.pool_sha256 = load_pool(spec, path)
        self.tensors = self.pool["tensors"]
        self.rows = int(next(iter(self.tensors.values())).shape[0])
        self.per_epoch = self.rows // spec.batch_size
        if self.per_epoch < 1:
            raise ValueError(f"pool of {self.rows} rows is smaller than one batch")
        self._epoch = None

    def indices(self, index):
        epoch, offset = divmod(int(index), self.per_epoch)
        if self._epoch is None or self._epoch[0] != epoch:
            g = torch.Generator("cpu").manual_seed(
                seed_of(f"{self.spec.model}/order{self.order}/epoch{epoch}"))
            self._epoch = (epoch, torch.randperm(self.rows, generator=g))
        b = self.spec.batch_size
        return self._epoch[1][offset * b:(offset + 1) * b]

    def batch(self, index):
        idx = self.indices(index)
        device = self.spec.device
        if self.spec.model == "convnextv2_tiny":
            return {"pixel_values": self.tensors["pixel_values"][idx].to(device),
                    "labels": self.tensors["labels"][idx].to(device)}
        batch = {k: self.tensors[k][idx].to(device)
                 for k in ("input_ids", "attention_mask", "labels", "mm_token_type_ids", "image_grid_thw")}
        # The packed patches of every image in the batch, in batch order.
        batch["pixel_values"] = self.tensors["pixel_values"][idx].reshape(
            -1, self.tensors["pixel_values"].shape[-1]).to(device)
        return batch

    def describe(self):
        return {"source": f"{self.spec.model} pool", "order": self.order,
                "batch_size": self.spec.batch_size, "rule": STREAM_RULE}


def pool_identity(spec, path=None):
    pool, sha = load_pool(spec, path)
    rows = int(next(iter(pool["tensors"].values())).shape[0])
    return {"dataset": pool["dataset"], "checkpoint_revision": pool["checkpoint_revision"],
            "pool_sha256": sha, "rows": rows, "skipped": len(pool.get("skipped", ())),
            "pool_rule": pool["rule"], "rule": STREAM_RULE}
