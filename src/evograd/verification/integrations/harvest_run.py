"""The run a harvest observes: the step a capture records first.

A model's snapshot is derived from one real training step; its corpus is
recorded from the same model's first step on each data order
(:data:`.base.ORDERS`). :func:`first_step` sets up the first of those exactly
as :func:`.base.capture` does -- the verification settings, the model's pinned
workload, a fresh native model, the first training batch of data order 101 --
and names it, so a snapshot can say which run it came from.
"""

from __future__ import annotations

from typing import Any

from .base import CALIBRATION_ORDERS, Model, build_model, configure, objective_of
from .objectives import to_device

#: The data order whose first batch a harvest runs (the capture's first order).
HARVEST_ORDER = CALIBRATION_ORDERS[0]


def first_step(model: Model, *, order: int = HARVEST_ORDER,
               device: str = "cuda") -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    """``(workload, native model, batch, run identity, determinism record)``.

    The run identity has ``data`` (the model's data identity without its
    per-order batches), ``order``, ``batch_digest`` (the batch's digest under
    the objective's ``batch_key``, as the capture names it) and
    ``parameter_state_sha256`` (the initial parameter state, as the capture
    records it). The caller adds ``weights``, which only its workload family
    can name.
    """
    settings = configure()
    workload = model.load_workload()
    objective = objective_of(model)
    batches, identity = model.train_batches(workload, (order,))
    net = build_model(workload)
    run = {"data": {k: v for k, v in identity.items() if k != "batches"}, "order": order,
           "batch_digest": identity["batches"][str(order)][objective.batch_key],
           "parameter_state_sha256": objective.parameter_state(net)["state_sha256"]}
    return workload, net, to_device(batches[order], device), run, settings
