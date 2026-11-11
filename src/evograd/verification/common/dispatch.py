"""One site's calls told apart by configuration.

A site observed at several configurations (MegaFold's LayerNorm+Linear at 23)
holds a candidate for one of them; that candidate replaces its configuration's
calls and nothing else. A call's configuration is what ``base.case_for`` binds
a call to its case by: each shaped argument's shape, dtype and strides (the
case's recorded strides, else contiguous).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Collection, Mapping

import torch


def shaped(op) -> list:
    """The declared arguments with a shape."""
    return [a for a in op.args if getattr(a, "shape", None) not in (None, "unspecified")]


def configuration(op, workload) -> tuple:
    """The configuration of ``op``'s calls at ``workload``."""
    from evograd.opdecl.activity import bind_shape
    from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

    strides = dict(getattr(getattr(workload, "provenance", None), "strides", None) or ())
    key = []
    for arg in shaped(op):
        shape = tuple(bind_shape(arg.shape, workload.dims))
        key.append((shape, resolve_dtype(arg_dtype_name(arg, workload)),
                    tuple(strides.get(arg.name) or torch.empty(shape, device="meta").stride())))
    return tuple(key)


def _strided(observed: Collection[tuple]) -> frozenset[int]:
    """The shaped arguments (by position) whose strides tell apart two observed
    configurations that agree in every shape and dtype."""
    return frozenset(i for a, b in combinations(observed, 2)
                     if all(x[:2] == y[:2] for x, y in zip(a, b))
                     for i, (x, y) in enumerate(zip(a, b)) if x[2] != y[2])


def by_configuration(op, observed: Collection[tuple], routes: Mapping[tuple, Callable],
                     other: Callable) -> Callable:
    """A site callable sending each call of a configuration in ``routes`` to its
    callable, and every other call to ``other``.

    ``observed`` holds every configuration the site was observed at. A call is
    routed by each shaped argument's shape and dtype, and by the strides of only
    those arguments whose strides tell observed configurations apart (DiT's last
    residual add differs from the others only in its shift and scale, halves of
    the final layer's projection). Any other stride is a candidate's to choose:
    DiT's ``summed`` is the next residual call's ``x``, so routing on it would
    send a candidate's later calls to ``other`` whenever it returns another layout.
    """
    names, keyed = [a.name for a in op.args], [a.name for a in shaped(op)]
    strided = _strided(observed)

    def key(entries) -> tuple:
        return tuple((shape, dtype, stride if i in strided else None)
                     for i, (shape, dtype, stride) in enumerate(entries))

    routes = {key(k): call for k, call in routes.items()}

    def dispatch(*args: Any, **kwargs: Any):
        values = {**dict(zip(names, args)), **kwargs}
        return routes.get(key((tuple(v.shape), v.dtype, tuple(v.stride()))
                              for v in map(values.get, keyed)), other)(*args, **kwargs)

    return dispatch
