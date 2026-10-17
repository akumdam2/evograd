"""Fixed Criteo Kaggle batches in TorchRec's preprocessed NumPy format.

Only the labeled first half of ``train`` is sampled. TorchRec-compatible preprocessing has
transformed the dense features; sparse IDs are reduced modulo each local table
size, matching its in-memory Criteo loader. No reference code is imported.
"""

import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from evograd.benchmark.topdown.common.data_paths import benchmark_dir, prepared_dir

DATASET = ("criteo_kaggle", "torchrec_preprocessed_numpy", "train_first_half")
SCHEMA = "evograd-dlrm-real-data/1"
MAPPING = "per_feature_modulo_table_size"
FILES = ("train_dense.npy", "train_sparse.npy", "train_labels.npy")
DEFAULT_SOURCE_DIR = benchmark_dir("torchrec_dlrm") / "processed"


def _source_dir(source_dir):
    value = (source_dir if source_dir is not None
             else os.environ.get("EVOGRAD_DLRM_CRITEO_DIR") or DEFAULT_SOURCE_DIR)
    directory = Path(value).expanduser().resolve()
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        hint = (" --criteo-dir resolved to the current directory; if you passed "
                '"$PROCESSED_DIR", that variable may be unset.'
                if directory == Path.cwd().resolve() else "")
        raise FileNotFoundError(
            f"Criteo preprocessing files missing in {directory}: {missing}. "
            "Expected TorchRec's --output_dir containing train_dense.npy, "
            "train_sparse.npy, and train_labels.npy. By default, run from the "
            f"repository root and preprocess into {DEFAULT_SOURCE_DIR}.{hint}"
        )
    return directory


def _arrays(directory):
    dense, sparse, labels = (np.load(directory / name, mmap_mode="r", allow_pickle=False)
                             for name in FILES)
    if (dense.ndim != 2 or dense.shape[1] != 13 or sparse.shape != (len(dense), 26)
            or labels.shape != (len(dense), 1)):
        raise ValueError("TorchRec Criteo arrays must have shapes [N,13], [N,26], [N,1]")
    if (dense.dtype != np.float32 or sparse.dtype.kind not in "iu"
            or labels.dtype.kind not in "iu"):
        raise ValueError("TorchRec Criteo arrays need float32 dense and integer sparse/labels")
    return dense, sparse, labels


def _source_stats(directory):
    return {name: {"bytes": (directory / name).stat().st_size,
                   "mtime_ns": (directory / name).stat().st_mtime_ns}
            for name in FILES}


def _selected_source_hash(directory, rows, batch_size):
    dense, sparse, labels = _arrays(directory)
    h = hashlib.sha256()
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        for value in (np.array(dense[selected], dtype=np.float32, copy=True),
                      np.array(sparse[selected], dtype=np.int64, copy=True),
                      np.array(labels[selected, 0], dtype=np.int64, copy=True)):
            h.update(value.tobytes())
    return h.hexdigest()


def prepare_pinned(spec, *, source_dir=None, batch_count=10):
    if spec.model != "torchrec_dlrm" or spec.input_mode != "real" or batch_count < 1:
        raise ValueError("DLRM prepared inputs require real mode and positive batch_count")
    directory = _source_dir(source_dir)
    dense, sparse, labels = _arrays(directory)
    train_rows = (len(dense) + 1) // 2  # Exclude the half TorchRec uses for its test stage.
    count = spec.batch_size * batch_count
    if count > train_rows:
        raise ValueError(f"Criteo training half has {train_rows} rows, need {count}")
    rows = random.Random(spec.data_seed).sample(range(train_rows), count)
    a = spec.arch
    sizes = np.asarray(a["num_embeddings_per_feature"], dtype=np.int64)
    source_hash = hashlib.sha256()
    batches = []
    for start in range(0, count, spec.batch_size):
        selected = rows[start:start + spec.batch_size]
        raw_dense = np.array(dense[selected], dtype=np.float32, copy=True)
        raw_sparse = np.array(sparse[selected], dtype=np.int64, copy=True)
        raw_labels = np.array(labels[selected, 0], dtype=np.int64, copy=True)
        if (not np.isfinite(raw_dense).all() or (raw_sparse < 0).any()
                or not np.isin(raw_labels, [0, 1]).all()):
            raise ValueError("selected Criteo rows contain invalid dense, sparse, or label values")
        for value in (raw_dense, raw_sparse, raw_labels):
            source_hash.update(value.tobytes())
        mapped = np.remainder(raw_sparse, sizes).T[:, :, None].copy()
        batches.append({
            "dense": torch.from_numpy(raw_dense).to(getattr(torch, spec.dtype)),
            "sparse_ids": torch.from_numpy(mapped),
            "labels": torch.from_numpy(raw_labels.astype(np.float32)).to(getattr(torch, spec.dtype)),
        })
    return {"schema_version": SCHEMA, "model": spec.model, "workload_id": spec.workload_id,
            "dataset": DATASET, "mapping": MAPPING, "table_sizes": a["num_embeddings_per_feature"],
            "source_dir": str(directory), "source_files": _source_stats(directory),
            "source_rows_sha256": source_hash.hexdigest(), "source_train_rows": train_rows,
            "selection": rows, "examples": count, "batches": batches}


def default_data_path(spec, source_dir=None):
    directory = _source_dir(source_dir)
    root = Path(os.environ.get("EVOGRAD_DLRM_DATA_CACHE", prepared_dir("torchrec_dlrm")))
    source_key = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    return root / f"{spec.workload_hash}-{source_key}-pinned.pt"


def ensure_pinned_data(spec, path=None, *, source_dir=None, batch_count=10):
    if batch_count < 1:
        raise ValueError("batch_count must be positive")
    supplied = path is not None
    path = Path(path) if supplied else default_data_path(spec, source_dir)
    if path.exists():
        from types import SimpleNamespace
        existing, _ = load_data(SimpleNamespace(spec=spec, data_path=str(path)))
        if source_dir:
            directory = _source_dir(source_dir)
            if (existing["source_dir"] != str(directory)
                    or existing["source_files"] != _source_stats(directory)
                    or existing["source_rows_sha256"] != _selected_source_hash(
                        directory, existing["selection"], spec.batch_size)):
                raise ValueError("prepared DLRM artifact differs from the supplied Criteo source")
        if len(existing["batches"]) >= batch_count:
            return str(path)
        if supplied:
            raise ValueError(f"prepared artifact has {len(existing['batches'])} batches, need {batch_count}")
    data = prepare_pinned(spec, source_dir=source_dir, batch_count=batch_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(data, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def load_data(workload):
    if workload.spec.model != "torchrec_dlrm" or workload.spec.input_mode != "real" or not workload.data_path:
        raise ValueError("real DLRM inputs require a prepared Criteo artifact")
    path = Path(workload.data_path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    a, b = workload.spec.arch, workload.spec.batch_size
    if (data.get("schema_version") != SCHEMA or data.get("model") != workload.spec.model
            or data.get("workload_id") != workload.spec.workload_id
            or tuple(data.get("dataset", ())) != DATASET or data.get("mapping") != MAPPING
            or data.get("table_sizes") != a["num_embeddings_per_feature"]):
        raise ValueError("DLRM artifact does not match the Criteo real-input contract")
    batches, rows = data.get("batches"), data.get("selection")
    count = data.get("examples")
    if (not isinstance(batches, list) or not batches or not isinstance(rows, list)
            or count != len(batches) * b or len(rows) != count
            or len(set(rows)) != count or any(type(row) is not int or row < 0
                                              or row >= data.get("source_train_rows", 0) for row in rows)
            or not isinstance(data.get("source_rows_sha256"), str)
            or len(data["source_rows_sha256"]) != 64):
        raise ValueError("DLRM artifact has invalid source selection or batch count")
    sizes = torch.tensor(a["num_embeddings_per_feature"])[:, None, None]
    for batch in batches:
        if (set(batch) != {"dense", "sparse_ids", "labels"}
                or tuple(batch["dense"].shape) != (b, 13)
                or tuple(batch["sparse_ids"].shape) != (26, b, 1)
                or tuple(batch["labels"].shape) != (b,)
                or batch["dense"].dtype != getattr(torch, workload.spec.dtype)
                or batch["labels"].dtype != getattr(torch, workload.spec.dtype)
                or batch["sparse_ids"].dtype != torch.int64
                or not torch.isfinite(batch["dense"]).all()
                or not torch.isfinite(batch["labels"]).all()
                or not ((batch["labels"] == 0) | (batch["labels"] == 1)).all()
                or (batch["sparse_ids"] < 0).any() or (batch["sparse_ids"] >= sizes).any()):
            raise ValueError("DLRM artifact batch differs from the Criteo specialist contract")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return data, h.hexdigest()
