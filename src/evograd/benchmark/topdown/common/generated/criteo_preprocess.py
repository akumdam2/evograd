"""Convert Criteo Kaggle train.txt to the NumPy layout used by TorchRec.

This implements the data conversion directly so preprocessing does not import
TorchRec's distributed stack or FBGEMM. It streams rows to disk-backed arrays
instead of holding the full dataset in Python lists.
"""

from pathlib import Path

import numpy as np


def _row_count(path: Path) -> int:
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            count += block.count(b"\n")
        if stream.tell():
            stream.seek(-1, 2)
            count += stream.read(1) != b"\n"
    if count == 0:
        raise ValueError(f"Criteo training file is empty: {path}")
    return count


def _write_chunk(dense, sparse, labels, start, dense_rows, sparse_rows, label_rows):
    size = len(label_rows)
    values = np.asarray(dense_rows, dtype=np.int32)
    dense[start:start + size] = values
    sparse[start:start + size] = np.asarray(sparse_rows, dtype=np.int64)
    labels[start:start + size, 0] = np.asarray(label_rows, dtype=np.int32)
    return int(values.min())


def preprocess_train(input_path: Path, output_dir: Path, *, chunk_rows: int = 16_384) -> None:
    """Write train_{dense,sparse,labels}.npy with TorchRec's Kaggle transforms."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    count = _row_count(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "train_dense_raw.npy.partial"
    partial = {
        name: output_dir / f"train_{name}.npy.partial"
        for name in ("dense", "sparse", "labels")
    }
    print(f"Converting {count:,} Criteo training rows", flush=True)
    try:
        dense = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.int32,
                                          shape=(count, 13))
        sparse = np.lib.format.open_memmap(partial["sparse"], mode="w+",
                                           dtype=np.int64, shape=(count, 26))
        labels = np.lib.format.open_memmap(partial["labels"], mode="w+",
                                           dtype=np.int32, shape=(count, 1))
        dense_rows, sparse_rows, label_rows = [], [], []
        start, minimum = 0, None
        with input_path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                columns = line.rstrip(b"\r\n").split(b"\t")
                if len(columns) != 40:
                    raise ValueError(f"Criteo row {line_number} has {len(columns)} fields, expected 40")
                try:
                    label_rows.append(int(columns[0] or b"0"))
                    dense_rows.append([int(value or b"0") for value in columns[1:14]])
                    sparse_rows.append([int(value or b"0", 16) for value in columns[14:]])
                except ValueError as exc:
                    raise ValueError(f"Criteo row {line_number} contains an invalid value") from exc
                if len(label_rows) == chunk_rows:
                    block_min = _write_chunk(dense, sparse, labels, start,
                                             dense_rows, sparse_rows, label_rows)
                    minimum = block_min if minimum is None else min(minimum, block_min)
                    start += len(label_rows)
                    dense_rows, sparse_rows, label_rows = [], [], []
                    if start % (chunk_rows * 64) == 0:
                        print(f"  converted {start:,}/{count:,} rows", flush=True)
        if label_rows:
            block_min = _write_chunk(dense, sparse, labels, start,
                                     dense_rows, sparse_rows, label_rows)
            minimum = block_min if minimum is None else min(minimum, block_min)
            start += len(label_rows)
        if start != count:
            raise ValueError(f"Criteo row count changed while reading: {start} != {count}")
        dense.flush()
        sparse.flush()
        labels.flush()
        del dense, sparse, labels

        raw_dense = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        dense_out = np.lib.format.open_memmap(partial["dense"], mode="w+",
                                               dtype=np.float32, shape=(count, 13))
        offset = np.int32(minimum - 2)
        for start in range(0, count, chunk_rows):
            values = np.array(raw_dense[start:start + chunk_rows], dtype=np.int32, copy=True)
            values -= offset
            dense_out[start:start + len(values)] = np.log(values, dtype=np.float32)
        dense_out.flush()
        del dense_out, raw_dense
        for name, path in partial.items():
            path.replace(output_dir / f"train_{name}.npy")
        print(f"Criteo NumPy arrays ready in {output_dir}", flush=True)
    finally:
        raw_path.unlink(missing_ok=True)
        for path in partial.values():
            path.unlink(missing_ok=True)
