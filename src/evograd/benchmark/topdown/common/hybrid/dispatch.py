"""Which implementation an import-time-dispatched Transformers function resolved to.

``use_kernel_func_from_hub_with_fallback`` picks the implementation of
``torch_chunk_gated_delta_rule``, ``causal_conv1d_fn`` and their decode
counterparts when the modeling module is imported: an importable ``fla`` or
``causal_conv1d`` silently wins over the torch function. A workload reports
what its module-level names dispatch to rather than assuming it.
"""

from __future__ import annotations

import importlib.util
from typing import Any

#: The dispatched names of a hybrid GDN modeling module, by the setting they report as.
DISPATCHED = {
    "gated_delta_rule_dispatch": "torch_chunk_gated_delta_rule",
    "causal_conv1d_dispatch": "causal_conv1d_fn",
    "recurrent_gated_delta_rule_dispatch": "torch_recurrent_gated_delta_rule",
    "causal_conv1d_update_dispatch": "causal_conv1d_update",
}

#: The two a training step calls; a workload pins them to ``"torch"``.
TRAINING_DISPATCH = ("gated_delta_rule_dispatch", "causal_conv1d_dispatch")


def dispatch_of(function) -> str:
    """``"torch"`` when the wrapper calls the torch function it wraps, else the
    implementation's dotted name (``"unknown"`` if it cannot be read)."""
    wrapped = getattr(function, "__wrapped__", None)
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None) or ()
    if code is None:
        return "unknown"
    cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure)))
    implementation = cells.get("implementation")
    if wrapped is not None and implementation is wrapped:
        return "torch"
    if implementation is None:
        return "unknown"
    return (f"{getattr(implementation, '__module__', '?')}."
            f"{getattr(implementation, '__qualname__', '?')}")


def kernel_dispatch(modeling) -> dict[str, Any]:
    """Every dispatched name of ``modeling``, plus whether the hub-kernels
    package (which can replace whole modules) is importable at all."""
    out: dict[str, Any] = {setting: dispatch_of(getattr(modeling, name))
                           for setting, name in DISPATCHED.items() if hasattr(modeling, name)}
    out["kernels_hub_installed"] = importlib.util.find_spec("kernels") is not None
    return out


__all__ = ["DISPATCHED", "TRAINING_DISPATCH", "dispatch_of", "kernel_dispatch"]
