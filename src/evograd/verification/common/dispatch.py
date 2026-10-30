"""One site's calls told apart by configuration.

A site observed at several configurations (MegaFold's LayerNorm+Linear at 23)
holds a candidate for one of them; that candidate replaces its configuration's
calls and nothing else. A call's configuration is what ``base.case_for`` binds
a call to its case by: each shaped argument's shape, dtype and strides (the
case's recorded strides, else contiguous).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

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


def by_configuration(op, routes: Mapping[tuple, Callable], other: Callable) -> Callable:
    """A site callable sending each call of a configuration in ``routes`` to its
    callable, and every other call to ``other``."""
    names, keyed = [a.name for a in op.args], [a.name for a in shaped(op)]
    routes = dict(routes)

    def dispatch(*args: Any, **kwargs: Any):
        values = {**dict(zip(names, args)), **kwargs}
        key = tuple((tuple(v.shape), v.dtype, tuple(v.stride())) for v in map(values.get, keyed))
        return routes.get(key, other)(*args, **kwargs)

    return dispatch
