"""Level-2 fused tasks that belong to no single model.

These are compositions any transformer-shaped training run performs, so their
cases are declared from generic shape sweeps and from frozen public model
configurations rather than from one harvested workload. A task whose shapes,
frequencies and provenance come from one model's captured step belongs to that
model's package under :mod:`evograd.benchmark.topdown` instead.

A grouping package: it declares no task itself. :mod:`evograd.benchmark.core`
recurses through it when it builds the task registry.
"""
