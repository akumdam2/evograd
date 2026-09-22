"""The facts an evaluation module needs in order to judge one model's cases.

Every module under :mod:`evograd.evaluation.workloads` was, until now, written
once per model and copied. The copies were 81% byte-identical, and what differed
was never logic -- it was a handful of names. This gathers those names in one
object so the logic can be written once.

The benchmark side already works this way: a model declares a
:class:`~evograd.benchmark.topdown.common.level4.Level4Workload` and the shared
harvest machinery reads it. This is the same idea one layer up, and it
deliberately *wraps* that declaration rather than restating it -- the model's
name, label, classes and package have exactly one home.

What is added here is evaluation's own vocabulary: where reports are written,
which benchmark suite carries this model's observed shapes, which layer the
capture describes, and which Level-2 tasks it presents.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalWorkload:
    """One model, as the evaluation modules need to refer to it."""

    #: Registry key, e.g. ``"llama_3_2_1b"``. Matches ``Level4Workload.name``.
    name: str
    #: Dotted path of the benchmark package that owns this model's cases.
    topdown_package: str
    #: Where this model's run artifacts and reports live, e.g.
    #: ``results/llama3-level4``. A local directory, never tracked.
    results_dir: Path
    #: Report schema prefix, e.g. ``"evograd-llama3"``. Each report appends its
    #: own suffix, so a Llama report cannot be mistaken for another model's.
    schema_prefix: str
    #: The decoder layer the Level-3 capture describes. Mid-stack by
    #: convention: the first and last layers see distributions the rest do not.
    representative_layer: int
    #: The Level-2 tasks this model presents, by their registry names.
    level2_tasks: tuple[str, ...]
    #: This model's canonical Level-4 loss, as its harvest recorded it.
    #: Printed for reference by the Level-1 loss check -- it is not a gate,
    #: and it is environment-sensitive: the same step under a Transformers
    #: release that takes a different SDPA path produces a different value.
    canonical_loss: float
    #: Directory name of this model's q/k/v boundary under
    #: ``levels/level2/``. The only Level-2 boundary whose identity differs
    #: between architectures: Qwen3 normalizes q and k per head before the
    #: rotation and Llama does not, so they are different operators.
    qkv_module: str
    #: The full-model classes a standalone layer replay must NOT hold. The
    #: decoder-layer and attention classes are expected to be live; these
    #: two existing would falsify the standalone claim.
    full_model_classes: tuple[str, ...]

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def observed_suite(self) -> str:
        """The benchmark suite carrying this model's harvested shapes."""
        return f"{self.name}_observed"

    @property
    def artifact_path(self) -> Path:
        """Default Level-3 capture, e.g. ``results/llama3-level4/layer8.pt``."""
        return self.results_dir / f"layer{self.representative_layer}.pt"

    def schema(self, suffix: str) -> str:
        """A report schema string, e.g. ``evograd-llama3-layer-replay-report/2``."""
        return f"{self.schema_prefix}-{suffix}"

    def capture(self, boundary: str) -> Any:
        """One Level-2 boundary's capture module, e.g. ``attention``.

        Always through ``.capture``: every boundary has one, and going via
        the package would depend on each model re-exporting the same names.
        """
        return self.module(f"levels.level2.{boundary}.capture")

    def module(self, relative: str) -> Any:
        """Import a module from this model's benchmark package.

        Resolved on demand rather than at construction: these modules reach
        torch and Transformers, and a descriptor must stay importable on a
        machine that has neither.
        """
        return importlib.import_module(f"{self.topdown_package}.{relative}")

    @cached_property
    def declaration(self):
        """This model's :class:`Level4Workload` -- the benchmark's own record."""
        return self.module("declaration").WORKLOAD

    @property
    def label(self) -> str:
        return self.declaration.label
