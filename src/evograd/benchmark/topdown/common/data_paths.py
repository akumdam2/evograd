"""Default local source and prepared-input paths for model benchmarks.

Benchmark commands run from the repository root, so their data stays under
``data/<model>/``. Explicit cache environment variables remain supported.
"""

import os
from pathlib import Path


def benchmark_dir(model: str) -> Path:
    return Path(os.environ.get("EVOGRAD_BENCHMARK_DATA_ROOT", "data")) / model


def hf_cache_dir(model: str) -> str:
    return os.environ.get("EVOGRAD_HF_CACHE") or str(benchmark_dir(model) / "hf_cache")


def prepared_dir(model: str) -> Path:
    return benchmark_dir(model) / "batches"
