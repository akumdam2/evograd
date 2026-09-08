"""Level 1 — the reusable primitives this workload executes.

    >>> from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import COMPOSES_INTO

The contracts live in :mod:`evograd.ops.level1`; :mod:`manifest` records which
of them Qwen3-0.6B runs, at which observed configurations, and which Level-2
site each composes into.
"""
