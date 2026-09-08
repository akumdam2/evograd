"""evograd: evolved backward kernels — seed via pipelines, optimize via OpenEvolve.

``evograd`` and ``EvogradResult`` are resolved on first use rather than at
import. Eagerly importing :mod:`evograd.api` here pulled the benchmark task
registry -- and through it every model manifest and frozen snapshot -- into any
process that merely said ``import evograd.ops``, which made the layer boundary
true on paper and false at runtime. The public names are unchanged; only when
they are loaded is.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["EvogradResult", "evograd"]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from evograd.api import EvogradResult, evograd


def __getattr__(name: str):
    if name in __all__:
        import importlib

        value = getattr(importlib.import_module("evograd.api"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
