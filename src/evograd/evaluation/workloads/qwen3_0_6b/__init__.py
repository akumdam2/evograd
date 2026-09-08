"""Judging implementations against the Qwen3-0.6B benchmark's cases.

The cases themselves -- the captured boundaries, the shapes, the frequencies,
the provenance -- belong to
:mod:`evograd.benchmark.topdown.qwen3_0_6b`. What is here is everything that
decides whether an implementation of one of them is acceptable: reference
comparison, tolerance calibration, repeated-noise measurement, negative
controls, and the reports and CLIs that carry a verdict.

The split is what keeps a threshold from being derived by the same module that
defines the case it gates.
"""
