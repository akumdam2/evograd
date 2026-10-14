"""Registry for benchmark tasks that are not operator declarations."""

from __future__ import annotations

from evograd.opdecl.workloads import WorkloadDecl

from evograd.benchmark.topdown.megafold import workload as _megafold

#: The level-3 whole-model tasks. MegaFold replaced the alphafold3-pytorch
#: declaration here: it is the implementation every AlphaFold3 shape in
#: ``opdecl.models`` was transcribed from, and it ships its own fused kernels
#: for four of the five patch sites, so the level-3 comparison has a real
#: baseline rather than only eager. The alphafold3 declaration and its tier-3
#: adapters are still in the tree and still importable; they are no longer part
#: of the suite.
WORKLOADS: dict[str, WorkloadDecl] = {_megafold.name: _megafold}


def get_workload(name: str) -> WorkloadDecl:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise KeyError(
            f"Unknown workload {name!r}; available: {sorted(WORKLOADS)}"
        ) from None


__all__ = ["WORKLOADS", "get_workload"]
