"""GEPA direct-code backend for shape-aware Evograd evolution."""

from .candidate import EvolveBlockTemplate
from .evaluator import ShapeBatchEvaluator, ShapeExample, examples_for_suite

__all__ = [
    "EvolveBlockTemplate",
    "ShapeBatchEvaluator",
    "ShapeExample",
    "examples_for_suite",
]
