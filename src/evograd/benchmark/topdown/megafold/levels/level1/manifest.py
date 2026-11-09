"""MegaFold's Level-1 observed bindings: every primitive its training step runs.

Each binds a shared generic task, because the mathematics matches exactly; the
generic tasks are in :mod:`evograd.ops.level1`, and a primitive gets a new one only
where none fits. ``linear`` and ``linear_no_bias`` are library-only: their cases are
counted, recorded and replayed as every model's are, and the evaluation plan leaves
them out of search. Operators the step runs that no task binds are in
``harvest.UNBOUND``, with why (``docs/AF3.md`` section 5).
"""

from __future__ import annotations

import dataclasses

from evograd.benchmark.cases import ObservedBinding

WORKLOAD = "megafold"
SUITE = "megafold_observed"

#: The primitives a training step runs, by the task each one binds.
OPERATORS = ("layernorm", "layernorm_no_affine", "layernorm_no_bias", "swiglu", "glu", "sigmoid",
             "softmax", "relu", "linear", "linear_no_bias")


@dataclasses.dataclass(frozen=True)
class HarvestedBinding(ObservedBinding):
    """A binding that contributes nothing until the snapshot holds its task.

    The bound tasks outgrew the tracked snapshot's two until its next harvest, and
    a binding the snapshot does not know must not stop the registry importing.
    """

    def workloads(self, primitive):
        from evograd.benchmark.topdown import has_snapshot, load_snapshot

        if not has_snapshot(WORKLOAD) or self.task not in load_snapshot(WORKLOAD)["level1"]:
            return ()
        return super().workloads(primitive)


OBSERVED_BINDINGS = tuple(HarvestedBinding(WORKLOAD, op, SUITE) for op in OPERATORS)
