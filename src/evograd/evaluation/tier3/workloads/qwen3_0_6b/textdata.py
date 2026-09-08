"""Pinned real text for the Qwen3 training-behaviour check: WikiText-2 raw.

Everything that decides which tokens the model sees is fixed and recorded:
the dataset repository and revision, the tokenizer revision, the join
convention between documents, the block length, the rule for the last
partial block, and the batch order for a given data seed. Two runs with the
same seed see byte-identical batches; the training and validation splits come
from different parquet files and are disjoint by construction.

The parquet files are read directly with ``pyarrow`` -- the ``datasets``
library is not a dependency of this environment and is not needed for two
columns of text.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

IGNORE_INDEX = -100


def hf_cache_dir() -> str | None:
    """Where pinned snapshots live. ``EVOGRAD_HF_CACHE`` wins; else HF's default."""
    return os.environ.get("EVOGRAD_HF_CACHE") or None


def dataset_file(split: str, *, cache_dir: str | None = None) -> str:
    """The local path of one split's parquet at the pinned revision.

    Local-only: a benchmark must not silently reach the network. Download once
    with ``huggingface_hub.hf_hub_download`` at ``DATASET_REVISION`` first.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        DATASET_REPO, f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet",
        repo_type="dataset", revision=DATASET_REVISION,
        cache_dir=cache_dir or hf_cache_dir(), local_files_only=True,
    )


def read_split(split: str, *, cache_dir: str | None = None) -> list[str]:
    """Non-empty lines of one split, in file order."""
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_file(split, cache_dir=cache_dir), columns=["text"])
    return [line for line in table.column("text").to_pylist() if line and line.strip()]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedSplit:
    """One split tokenised and packed into fixed-length blocks."""

    split: str
    blocks: torch.Tensor          # [N, block_len] int64, on CPU
    tokens_total: int             # tokens before packing
    dropped_tail: int             # tokens in the discarded partial block
    document_count: int
    parquet_sha256: str

    @property
    def block_count(self) -> int:
        return int(self.blocks.shape[0])

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split, "blocks": self.block_count,
            "block_len": int(self.blocks.shape[1]),
            "tokens_total": self.tokens_total, "dropped_tail": self.dropped_tail,
            "documents": self.document_count, "parquet_sha256": self.parquet_sha256,
        }


def pack_split(
    split: str,
    tokenizer,
    *,
    block_len: int,
    cache_dir: str | None = None,
) -> PackedSplit:
    """Tokenise a split and pack it into contiguous ``block_len`` blocks.

    Documents (non-empty lines) are joined with the tokenizer's EOS token, then
    the whole stream is cut into blocks of ``block_len``. The trailing partial
    block is dropped rather than padded, so every block is full and every
    position except the last predicts a real token; ``labels = input_ids``
    therefore needs no ``ignore_index`` anywhere. That choice is recorded, not
    silent: ``dropped_tail`` says how many tokens it cost.
    """
    lines = read_split(split, cache_dir=cache_dir)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("the tokenizer has no EOS token to join documents with")
    stream: list[int] = []
    for line in lines:
        stream.extend(tokenizer(line, add_special_tokens=False)["input_ids"])
        stream.append(eos)
    total = len(stream)
    usable = (total // block_len) * block_len
    blocks = torch.tensor(stream[:usable], dtype=torch.long).view(-1, block_len)
    return PackedSplit(
        split=split, blocks=blocks, tokens_total=total,
        dropped_tail=total - usable, document_count=len(lines),
        parquet_sha256=file_sha256(dataset_file(split, cache_dir=cache_dir)),
    )


@dataclass
class TextBatches:
    """Deterministic batches over a packed split, for one data-order seed.

    The order is a seeded permutation of block indices, consumed
    ``batch_size`` at a time and cycled when exhausted -- so a 1,000-step run
    over ~550 batches sees the same blocks again in the same order rather than
    a fresh shuffle. Every provider given the same seed sees the same batches
    at the same steps.
    """

    packed: PackedSplit
    batch_size: int
    seed: int
    device: str = "cuda"
    _order: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        generator = torch.Generator().manual_seed(self.seed)
        self._order = torch.randperm(self.packed.block_count, generator=generator)

    @property
    def batches_per_epoch(self) -> int:
        return self.packed.block_count // self.batch_size

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(input_ids, labels)`` for ``step``; ``labels = input_ids``."""
        if self.batches_per_epoch == 0:
            raise ValueError("split has fewer blocks than one batch")
        index = step % self.batches_per_epoch
        chosen = self._order[index * self.batch_size:(index + 1) * self.batch_size]
        ids = self.packed.blocks[chosen].to(self.device)
        return ids, ids.clone()

    def epoch_of(self, step: int) -> int:
        return step // self.batches_per_epoch

    def describe(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size, "seed": self.seed,
                "batches_per_epoch": self.batches_per_epoch,
                "order_sha256": hashlib.sha256(
                    self._order.numpy().tobytes()).hexdigest()[:16]}


def data_identity(train: PackedSplit, validation: PackedSplit, *, block_len: int,
                  tokenizer_sha256: str) -> dict[str, Any]:
    """Everything that fixes which tokens are seen, for the workload record."""
    return {
        "dataset": DATASET_REPO, "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "join": "documents joined with EOS; trailing partial block dropped",
        "block_len": block_len, "ignore_index": IGNORE_INDEX,
        "tokenizer_sha256": tokenizer_sha256,
        "train": train.describe(), "validation": validation.describe(),
        "disjoint": "train and validation are separate parquet files of the "
                    "published split",
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]
