from pathlib import Path

import numpy as np
import pytest

from evograd.benchmark.topdown.common.generated.criteo_preprocess import preprocess_train


def test_criteo_conversion_matches_torchrec_kaggle_transform(tmp_path: Path):
    rows = [
        [b"1", *[str(value).encode() for value in range(13)], b"", *([b"a"] * 25)],
        [b"0", *([b"4"] * 13), *([b"ff"] * 26)],
        [b"1", *([b"2"] * 13), *([b"FFFFFFFF"] * 26)],
    ]
    source = tmp_path / "train.txt"
    source.write_bytes(b"\n".join(b"\t".join(row) for row in rows))
    output = tmp_path / "processed"

    preprocess_train(source, output, chunk_rows=2)

    dense = np.load(output / "train_dense.npy")
    sparse = np.load(output / "train_sparse.npy")
    labels = np.load(output / "train_labels.npy")
    raw_dense = np.array([list(range(13)), [4] * 13, [2] * 13], dtype=np.int32)
    np.testing.assert_array_equal(dense, np.log(raw_dense + 2, dtype=np.float32))
    assert dense.dtype == np.float32
    assert sparse.dtype == np.int64
    assert sparse.shape == (3, 26)
    assert (sparse[0, 0], sparse[0, 1], sparse[1, 0], sparse[2, 0]) == (
        0, 10, 255, 2**32 - 1
    )
    assert labels.dtype == np.int32
    assert labels.tolist() == [[1], [0], [1]]


def test_criteo_conversion_rejects_malformed_rows_without_output(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_bytes(b"1\t2\n")
    output = tmp_path / "processed"

    with pytest.raises(ValueError, match="expected 40"):
        preprocess_train(source, output, chunk_rows=2)

    assert not list(output.glob("train_*.npy"))
