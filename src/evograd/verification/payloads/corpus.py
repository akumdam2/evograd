"""A versioned, immutable corpus of real operator-call payloads.

A *payload* is one recorded invocation of one optimization case during a
native training step: the exact argument tensors the model passed (values,
dtypes, shapes, strides, storage offsets, aliases), its scalar arguments, and
the output cotangents the training objective's backward delivered to it. The
corpus is model-independent; a workload's capture code decides what a call and
a case are (``evograd.verification.integrations.qwen3_0_6b`` for Qwen3-0.6B).

Storage: every tensor's whole untyped storage is written once, keyed by the
SHA-256 of its bytes, into raw shard files; a tensor record names its storage
plus ``dtype/shape/stride/storage_offset``. Loading rebuilds the exact view on
the target device (``Tensor.set_``), so ``.cpu()``/``.contiguous()``/reshape
never alter the interface, and two arguments that shared a storage at capture
share one again after loading. Each storage's bytes are re-hashed when loaded;
a mismatch is an error, never a silent repair. Internally overlapping layouts
(e.g. ``expand``) are refused at capture. A view of a storage much larger than
itself is stored compacted: same shape and strides at offset 0, recorded as
``compacted`` (the call's stride interface is unchanged; only unrelated bytes of
its storage are dropped).

The manifest (``manifest.json``) holds identity, the storage index, every
payload record, the frozen splits and the coverage, and its own hash. Once
written it is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "evograd-real-payload-corpus/1"
MANIFEST = "manifest.json"
#: A storage at least this many times larger than its view, and larger by at
#: least ``_COMPACT_MIN_EXTRA`` bytes, is stored compacted.
_COMPACT_RATIO = 4
_COMPACT_MIN_EXTRA = 64 << 20


class CorpusError(RuntimeError):
    """The corpus is missing, incompatible, corrupt, or cannot serve a request."""


class UnsupportedLayout(CorpusError):
    """A tensor whose layout the corpus cannot reproduce exactly (e.g. internal overlap)."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dtype(name: str):
    import torch

    return getattr(torch, name.split(".", 1)[1] if name.startswith("torch.") else name)


def cotangent_stats(t) -> dict[str, Any]:
    """Zero/non-zero coverage of one cotangent (a zero cotangent exercises no backward)."""
    import torch

    if t is None:
        return {"present": False}
    nonzero = int((t != 0).sum()) if t.numel() else 0
    finite = bool(torch.isfinite(t).all()) if t.is_floating_point() else True
    return {"present": True, "numel": t.numel(), "nonzero": nonzero, "all_zero": nonzero == 0,
            "finite": finite}


# ── writing ──────────────────────────────────────────────────────────────────


class CorpusWriter:
    """Streams storages to sharded files with content deduplication.

    Use ``new_step()`` between independent executions: the live-storage
    shortcut (same device pointer => same storage) is only valid while the
    tensors of one execution are alive.
    """

    def __init__(self, root: str | Path, *, shard_bytes: int = 4 << 30):
        self.root = Path(root)
        if (self.root / MANIFEST).exists():
            raise CorpusError(f"{self.root} already holds a corpus; corpora are immutable")
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_bytes)
        self.storages: dict[str, dict[str, Any]] = {}
        self.payloads: list[dict[str, Any]] = []
        self._live: dict[tuple[str, int, int], str] = {}
        self._shard_index = -1
        self._shard_handle = None
        self._shard_size = 0
        self.bytes_written = 0
        self.bytes_deduplicated = 0

    def new_step(self) -> None:
        self._live.clear()

    def _open_shard(self) -> None:
        if self._shard_handle is not None:
            self._shard_handle.close()
        self._shard_index += 1
        self._shard_size = 0
        self._shard_handle = open(self.root / "shards" / f"shard_{self._shard_index:05d}.bin", "wb")

    def _write(self, data) -> tuple[str, int]:
        if self._shard_handle is None or self._shard_size + len(data) > self.shard_bytes:
            self._open_shard()
        offset = self._shard_size
        self._shard_handle.write(data)
        self._shard_size += len(data)
        self.bytes_written += len(data)
        return f"shards/shard_{self._shard_index:05d}.bin", offset

    def _store(self, storage, device: str) -> str:
        import torch

        nbytes = storage.nbytes()
        live = (device, storage.data_ptr(), nbytes)
        if nbytes and live in self._live:
            return self._live[live]
        flat = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (nbytes,), (1,))
        host = flat.detach().cpu().numpy()
        key = hashlib.sha256(memoryview(host)).hexdigest()
        if key in self.storages:
            self.bytes_deduplicated += nbytes
        else:
            shard, offset = self._write(memoryview(host))
            self.storages[key] = {"shard": shard, "offset": offset, "nbytes": nbytes}
        if nbytes:
            self._live[live] = key
        return key

    def put(self, t) -> dict[str, Any]:
        """Store one tensor; returns its record. Refuses internally overlapping layouts."""
        import torch

        from evograd.verification.common.tensors import tensor_digest

        if t.layout != torch.strided:
            raise UnsupportedLayout(f"layout {t.layout} is not supported")
        t = t.detach()
        if t.numel() and torch._debug_has_internal_overlap(t) != 0:
            raise UnsupportedLayout(f"internally overlapping layout {tuple(t.shape)} / {t.stride()}")
        logical = t.numel() * t.element_size()
        storage = t.untyped_storage()
        compacted = False
        if storage.nbytes() > _COMPACT_RATIO * max(logical, 1) and \
                storage.nbytes() - logical > _COMPACT_MIN_EXTRA:
            dense = torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device=t.device)
            dense.copy_(t)
            t, storage, compacted = dense, dense.untyped_storage(), True
        key = self._store(storage, str(t.device))
        return {"storage": key, "dtype": str(t.dtype), "shape": list(t.shape),
                "stride": list(t.stride()), "storage_offset": int(t.storage_offset()),
                "digest": tensor_digest(t), "compacted": compacted}

    def add_payload(self, record: Mapping[str, Any]) -> None:
        if any(p["payload_id"] == record["payload_id"] for p in self.payloads[-64:]):
            raise CorpusError(f"duplicate payload id {record['payload_id']}")
        self.payloads.append(dict(record))

    def finish(self, *, identity: Mapping[str, Any], splits: Mapping[str, list[str]],
               coverage: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> Path:
        if self._shard_handle is not None:
            self._shard_handle.close()
            self._shard_handle = None
        ids = [p["payload_id"] for p in self.payloads]
        if len(ids) != len(set(ids)):
            raise CorpusError("payload ids are not unique")
        known = set(ids)
        for name, members in splits.items():
            missing = [m for m in members if m not in known]
            if missing:
                raise CorpusError(f"split {name!r} names unknown payloads: {missing[:3]}")
        shards = []
        for path in sorted((self.root / "shards").glob("shard_*.bin")):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(64 << 20), b""):
                    digest.update(block)
            shards.append({"file": f"shards/{path.name}", "bytes": path.stat().st_size,
                           "sha256": digest.hexdigest()})
        body = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "identity": dict(identity), "shards": shards, "storages": self.storages,
                "payloads": self.payloads, "splits": {k: list(v) for k, v in splits.items()},
                "coverage": dict(coverage),
                "storage_stats": {"bytes_written": self.bytes_written,
                                  "bytes_deduplicated": self.bytes_deduplicated,
                                  "storages": len(self.storages)},
                **dict(extra or {})}
        body["corpus_sha256"] = sha256_text(canonical_json(body))
        body["corpus_id"] = f"{identity.get('name', 'corpus')}@{body['corpus_sha256'][:16]}"
        path = self.root / MANIFEST
        path.write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
        os.chmod(path, 0o440)
        return path


# ── reading ──────────────────────────────────────────────────────────────────


@dataclass
class Corpus:
    """A loaded manifest. Payload values are read on demand, never all at once."""

    manifest_path: Path
    data: dict[str, Any] = field(repr=False)

    @classmethod
    def open(cls, manifest: str | Path, *, check_shards: bool = True) -> "Corpus":
        path = Path(manifest)
        if path.is_dir():
            path = path / MANIFEST
        if not path.is_file():
            raise CorpusError(f"no corpus manifest at {path}")
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise CorpusError(f"{path}: schema {data.get('schema')!r} is not {SCHEMA!r}")
        stored = data.pop("corpus_sha256", None)
        corpus_id = data.pop("corpus_id", None)
        if stored != sha256_text(canonical_json(data)):
            raise CorpusError(f"{path}: manifest content does not match its recorded hash")
        data["corpus_sha256"], data["corpus_id"] = stored, corpus_id
        corpus = cls(path, data)
        if check_shards:
            for shard in data["shards"]:
                file = corpus.root / shard["file"]
                if not file.is_file() or file.stat().st_size != shard["bytes"]:
                    raise CorpusError(f"shard {file} is missing or has the wrong size")
        return corpus

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    @property
    def corpus_id(self) -> str:
        return self.data["corpus_id"]

    @property
    def sha256(self) -> str:
        return self.data["corpus_sha256"]

    @property
    def identity(self) -> dict[str, Any]:
        return self.data["identity"]

    def binding(self, split: str | None = None) -> dict[str, Any]:
        """What a policy or a cached result binds to: corpus content, and the split's members."""
        out = {"corpus_id": self.corpus_id, "corpus_sha256": self.sha256, "schema": SCHEMA,
               "reference": self.identity.get("reference")}
        if split is not None:
            members = self.split(split)
            out["split"] = {"name": split, "payloads": len(members),
                            "sha256": sha256_text(canonical_json(sorted(members)))}
        return out

    def split(self, name: str) -> list[str]:
        try:
            return list(self.data["splits"][name])
        except KeyError:
            raise CorpusError(f"corpus {self.corpus_id} has no split {name!r}; "
                              f"splits: {sorted(self.data['splits'])}") from None

    def payloads(self, *, case_id: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        selected = self.data["payloads"]
        if split is not None:
            members = set(self.split(split))
            selected = [p for p in selected if p["payload_id"] in members]
        if case_id is not None:
            selected = [p for p in selected if p["case_id"] == case_id]
        return list(selected)

    def payload(self, payload_id: str) -> dict[str, Any]:
        for p in self.data["payloads"]:
            if p["payload_id"] == payload_id:
                return p
        raise CorpusError(f"no payload {payload_id!r} in {self.corpus_id}")

    def _read_storage(self, key: str, device: str):
        import numpy as np
        import torch

        entry = self.data["storages"].get(key)
        if entry is None:
            raise CorpusError(f"storage {key[:16]} is not in the corpus index")
        host = np.fromfile(self.root / entry["shard"], dtype=np.uint8, count=entry["nbytes"],
                           offset=entry["offset"])
        if host.size != entry["nbytes"]:
            raise CorpusError(f"storage {key[:16]}: short read ({host.size} of {entry['nbytes']} bytes)")
        if hashlib.sha256(memoryview(host)).hexdigest() != key:
            raise CorpusError(f"storage {key[:16]}: content does not match its hash")
        flat = torch.from_numpy(host)
        if device != "cpu":
            flat = flat.to(device)
        else:
            flat = flat.clone()
        return flat.untyped_storage()

    def load(self, payload: str | Mapping[str, Any], *, device: str = "cuda",
             names: Iterable[str] | None = None) -> dict[str, Any]:
        """The payload's arguments and cotangents by name, with their exact layouts.

        Tensors that shared a storage at capture share one storage here.
        """
        import torch

        record = self.payload(payload) if isinstance(payload, str) else payload
        entries = {**record["args"], **record["cotangents"]}
        if names is not None:
            entries = {n: entries[n] for n in names}
        storages: dict[str, Any] = {}
        out: dict[str, Any] = {}
        for name, entry in entries.items():
            if "scalar" in entry:
                out[name] = entry["scalar"]
                continue
            key = entry["storage"]
            if key not in storages:
                storages[key] = self._read_storage(key, device)
            dtype = _dtype(entry["dtype"])
            t = torch.empty(0, dtype=dtype, device=device).set_(
                storages[key], int(entry["storage_offset"]), tuple(entry["shape"]), tuple(entry["stride"]))
            if list(t.stride()) != list(entry["stride"]) or t.storage_offset() != entry["storage_offset"]:
                raise CorpusError(f"{record['payload_id']}:{name}: layout not reproduced")
            out[name] = t
        return out
