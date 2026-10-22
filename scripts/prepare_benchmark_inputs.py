#!/usr/bin/env python3
"""Download the pinned real-input sources needed by a model benchmark.

Run with an editable evograd installation. Sources are kept under
``data/<benchmark>/`` in the repository. The gate/harvest/Tier-3 CLI creates
the fixed batch artifact there when it runs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from shlex import quote
import shutil
import sys
import tarfile
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CRITEO_URL = (
    "https://go.criteo.net/"
    "criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"
)
CRITEO_HF_REPO = "Recommenders/criteo"
CRITEO_HF_REVISION = "1d462d3f6add11c4b4d5b7567da59a3311ee909e"
CRITEO_HF_FILE = "dac.tar.gz"
CRITEO_HF_SIZE = 4_576_820_670
CRITEO_HF_SHA256 = "ffb2ecbbc4ca478d2e87cc5168cca29f35a51ac5132c3a8fdea78c378c4e4c33"
BENCHMARKS = (
    "convnextv2_tiny", "qwen2_vl_2b", "dit_s_2", "torchrec_dlrm", "qwen3_0_6b",
    "llama_3_2_1b",
)


def _hf_cache(benchmark: str, cache_dir: Path | None) -> str:
    from evograd.benchmark.topdown.common.data_paths import hf_cache_dir

    cache = str(cache_dir.expanduser().resolve()) if cache_dir else str(
        Path(hf_cache_dir(benchmark)).resolve()
    )
    os.environ["EVOGRAD_HF_CACHE"] = cache
    return cache


def _hf_sources(benchmark: str, cache_dir: str | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    if benchmark in ("convnextv2_tiny", "qwen2_vl_2b"):
        from evograd.benchmark.topdown.common.vision.spec import DATASETS, MODELS, REVISIONS
        from evograd.benchmark.topdown.common.vision.model import pretrained_snapshot
        from evograd.benchmark.topdown.common.vision.data import _local_dataset_file
        from evograd.benchmark.topdown.common.vision.spec import canonical

        repo, revision, filename = DATASETS[benchmark]
        print(f"Downloading dataset {repo}@{revision}/{filename}", flush=True)
        hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                        cache_dir=cache_dir)
        model = MODELS[benchmark][0]
        print(f"Downloading checkpoint {model}@{REVISIONS[benchmark]}", flush=True)
        snapshot_download(model, revision=REVISIONS[benchmark], cache_dir=cache_dir,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
        spec = canonical(benchmark).replace(weights="pretrained")
        _local_dataset_file(spec)
        pretrained_snapshot(spec)
    elif benchmark == "dit_s_2":
        from evograd.benchmark.topdown.common.generated.data import DATASET, VAE
        from evograd.benchmark.topdown.common.generated.data import _dataset_file, _vae_path

        repo, revision, filename = DATASET
        print(f"Downloading dataset {repo}@{revision}/{filename}", flush=True)
        hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                        cache_dir=cache_dir)
        repo, revision = VAE
        print(f"Downloading VAE {repo}@{revision}", flush=True)
        snapshot_download(repo, revision=revision, cache_dir=cache_dir,
                          allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
        _dataset_file()
        _vae_path()
    elif benchmark == "qwen3_0_6b":
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import PRETRAINED
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.textdata import (
            DATASET_CONFIG, DATASET_REPO, DATASET_REVISION, dataset_file,
        )

        model, revision = PRETRAINED.removeprefix("pretrained:").split("@", 1)
        print(f"Downloading checkpoint {model}@{revision}", flush=True)
        snapshot_download(model, revision=revision, cache_dir=cache_dir,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
        for split in ("train", "validation"):
            filename = f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet"
            print(f"Downloading dataset {DATASET_REPO}@{DATASET_REVISION}/{filename}",
                  flush=True)
            hf_hub_download(DATASET_REPO, filename, repo_type="dataset",
                            revision=DATASET_REVISION, cache_dir=cache_dir)
            dataset_file(split, cache_dir=cache_dir)
    elif benchmark == "llama_3_2_1b":
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.model import pretrained_snapshot
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.spec import (
            CANONICAL, PRETRAINED, REAL_TEXT,
        )
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.textdata import (
            DATASET_CONFIG, DATASET_REPO, DATASET_REVISION, dataset_file,
        )

        if REAL_TEXT != f"wikitext-2-raw:{DATASET_REVISION}":
            raise RuntimeError("Llama's pinned WikiText revision differs from the shared loader's")
        model, revision = PRETRAINED.removeprefix("pretrained:").split("@", 1)
        print(f"Downloading checkpoint {model}@{revision} (gated)", flush=True)
        try:
            snapshot_download(model, revision=revision, cache_dir=cache_dir,
                              allow_patterns=["*.json", "*.safetensors"],
                              ignore_patterns=["original/*"])
        except Exception as exc:
            raise RuntimeError(
                f"cannot download {model}@{revision}: {type(exc).__name__}: {exc}\n"
                "The repository is gated: accept Meta's Llama 3.2 license on "
                "huggingface.co with your account, then `huggingface-cli login` "
                "(or set HF_TOKEN). On an offline node, run this on a machine with "
                "network access and copy the cache, passing --hf-cache."
            ) from exc
        for split in ("train", "validation"):
            filename = f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet"
            print(f"Downloading dataset {DATASET_REPO}@{DATASET_REVISION}/{filename}",
                  flush=True)
            hf_hub_download(DATASET_REPO, filename, repo_type="dataset",
                            revision=DATASET_REVISION, cache_dir=cache_dir)
            dataset_file(split, cache_dir=cache_dir)
        # The same local-only resolution the benchmark uses, so a cache that
        # prepares but would not load fails here rather than on a GPU node.
        pretrained_snapshot(CANONICAL.replace(weights=PRETRAINED, data=REAL_TEXT))


def _download_criteo(raw_dir: Path, url: str, archive_path: Path | None = None) -> None:
    train = raw_dir / "train.txt"
    if train.is_file():
        print(f"Using existing {train}", flush=True)
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_path if archive_path is not None else raw_dir / "criteo-kaggle.tar.gz"
    if archive_path is not None and not archive.is_file():
        raise FileNotFoundError(f"Criteo archive does not exist: {archive}")
    if not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".partial")
        print(f"Downloading Criteo Kaggle archive to {archive}", flush=True)
        try:
            with urlopen(url, timeout=60) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 << 20)
            partial.replace(archive)
        except URLError as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "Cannot reach the Criteo download host from this machine. "
                "Download the archive on a machine with internet access, copy it "
                "to the cluster, and rerun with --criteo-archive /path/to/archive.tar.gz. "
                "An existing raw/train.txt also works."
            ) from exc
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    print(f"Extracting train.txt from {archive}", flush=True)
    partial = train.with_suffix(".txt.partial")
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if member.isfile() and Path(member.name).name == "train.txt":
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError("Cannot read train.txt from Criteo archive")
                    with source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 << 20)
                    break
            else:
                raise RuntimeError("Criteo archive has no train.txt")
        partial.replace(train)
    finally:
        partial.unlink(missing_ok=True)


def _verify_hf_criteo_archive(path: Path) -> Path:
    if path.stat().st_size != CRITEO_HF_SIZE:
        raise RuntimeError(f"Criteo Hub archive has unexpected size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != CRITEO_HF_SHA256:
        raise RuntimeError(f"Criteo Hub archive failed SHA256 verification: {path}")
    return path


def _hf_criteo_archive() -> Path:
    """Fetch and verify the pinned Hub copy of the Criteo Kaggle archive."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading {CRITEO_HF_REPO}@{CRITEO_HF_REVISION}/{CRITEO_HF_FILE}",
          flush=True)
    path = Path(hf_hub_download(
        CRITEO_HF_REPO, CRITEO_HF_FILE, repo_type="dataset",
        revision=CRITEO_HF_REVISION,
        cache_dir=os.environ.get("EVOGRAD_HF_CACHE") or None,
    ))
    return _verify_hf_criteo_archive(path)


def _prepare_criteo(directory: Path, raw_dir: Path, url: str | None,
                    archive_path: Path | None = None) -> None:
    from evograd.benchmark.topdown.common.generated.dlrm_data import FILES, _arrays

    directory.mkdir(parents=True, exist_ok=True)
    if not all((directory / name).is_file() for name in FILES):
        local_hf_archive = raw_dir / CRITEO_HF_FILE
        if archive_path is None and url is None and local_hf_archive.is_file():
            archive_path = _verify_hf_criteo_archive(local_hf_archive)
        if (archive_path is None and url is None
                and not (raw_dir / "train.txt").is_file()
                and not (raw_dir / "criteo-kaggle.tar.gz").is_file()):
            archive_path = _hf_criteo_archive()
        _download_criteo(raw_dir, url or CRITEO_URL, archive_path)
        from evograd.benchmark.topdown.common.generated.criteo_preprocess import preprocess_train

        preprocess_train(raw_dir / "train.txt", directory)
    _arrays(directory)
    print(f"Criteo NumPy inputs ready: {directory}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--hf-cache", type=Path,
                        help="override data/<benchmark>/hf_cache")
    parser.add_argument("--criteo-dir", type=Path,
                        help="processed Criteo directory (default: data/torchrec_dlrm/processed)")
    parser.add_argument("--criteo-raw-dir", type=Path,
                        help="raw Criteo directory (default: data/torchrec_dlrm/raw)")
    parser.add_argument("--criteo-url",
                        help="override the default pinned Hugging Face Criteo archive")
    parser.add_argument("--criteo-archive", type=Path,
                        help="existing local Criteo .tar.gz archive for offline clusters")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    cache_dir = _hf_cache(args.benchmark, args.hf_cache)

    if args.benchmark == "torchrec_dlrm":
        from evograd.benchmark.topdown.common.generated.dlrm_data import DEFAULT_SOURCE_DIR
        from evograd.benchmark.topdown.common.data_paths import benchmark_dir

        directory = (args.criteo_dir or DEFAULT_SOURCE_DIR).expanduser().resolve()
        raw_dir = (args.criteo_raw_dir or benchmark_dir("torchrec_dlrm") / "raw").expanduser().resolve()
        archive = args.criteo_archive.expanduser().resolve() if args.criteo_archive else None
        _prepare_criteo(directory, raw_dir, args.criteo_url, archive)
        if directory != DEFAULT_SOURCE_DIR.resolve():
            print(f"Pass --criteo-dir {directory} to the gate and Tier-3 CLI.")
    else:
        _hf_sources(args.benchmark, cache_dir)
    if args.hf_cache:
        print(f"Before running the CLI: export EVOGRAD_HF_CACHE={quote(cache_dir)}")
    print(f"Sources ready for {args.benchmark}; run the gate/harvest/Tier-3 CLI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
