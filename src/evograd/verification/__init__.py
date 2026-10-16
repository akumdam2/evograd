"""Candidate verification, rebuilt: one execution core per gate.

    common/   result contracts, tensor comparators, reports
    gate1/    local pair vs the declared PyTorch oracle (elementwise atol/rtol)
    gate2/    framework integration: one autograd forward/backward per scope
              (operator, model) vs native eager, calibrated E2/Emax
    gate3/    continuous training, calibrated whole-trajectory RMS/max acceptance

Gate submodules load on access; model
dependencies (transformers, workload packages) load inside the adapters that
need them. See README.md in this directory.
"""

import importlib

_GATES = ("common", "gate1", "gate2", "gate3")


def __getattr__(name):
    if name in _GATES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_GATES)
