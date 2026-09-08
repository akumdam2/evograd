"""Registry for benchmark tasks that are not operator declarations."""

from __future__ import annotations

from evograd.opdecl.workloads import WorkloadDecl

from evograd.benchmark.topdown.alphafold3 import workload as _alphafold3

WORKLOADS: dict[str, WorkloadDecl] = {_alphafold3.name: _alphafold3}


def get_workload(name: str) -> WorkloadDecl:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise KeyError(
            f"Unknown workload {name!r}; available: {sorted(WORKLOADS)}"
        ) from None


__all__ = ["WORKLOADS", "get_workload"]
