"""Gate 1 tolerances: fixed dtype defaults for model-reference tasks.

The table is frozen from torch.testing.assert_close in PyTorch 2.11, rather
than imported from a private PyTorch helper whose defaults could change.
Values are (atol, rtol), unlike PyTorch's internal (rtol, atol) helper.
Model tasks ignore declaration overrides, result multipliers and shape hooks.
Generic mathematical tasks retain their existing declaration policy.
"""

from __future__ import annotations

import torch

MODEL_POLICY = "evograd-gate1-model-dtype-tolerances/1"
DECLARED_POLICY = "evograd-gate1-declared-tolerances/1"
SOURCE = "torch.testing.assert_close defaults, PyTorch 2.11"

_DEFAULTS = {
    torch.float16: (1e-5, 1e-3),
    torch.bfloat16: (1e-5, 1.6e-2),
    torch.float32: (1e-5, 1.3e-6),
    torch.float64: (1e-7, 1e-7),
}


def model_policy_identity() -> dict:
    """Serializable policy content, included in reports and workflow bindings."""
    return {
        "schema": MODEL_POLICY,
        "source": SOURCE,
        "dtype_source": "reference result tensor (candidate dtype must match)",
        "defaults": {
            str(dtype).removeprefix("torch."): {"atol": atol, "rtol": rtol}
            for dtype, (atol, rtol) in _DEFAULTS.items()
        },
        "declaration_scaling": False,
    }


def policy_identity(op) -> dict:
    if op.reference_contract:
        return model_policy_identity()
    return {"schema": DECLARED_POLICY, "source": "op.tolerance_for(workload, result)"}


def result_tolerances(op, workload, result_name: str, reference: torch.Tensor) -> tuple[float, float]:
    """Return (atol, rtol) for this result, not the workload's activation dtype."""
    if not op.reference_contract:
        return op.tolerance_for(workload, result_name)
    try:
        return _DEFAULTS[reference.dtype]
    except KeyError:
        raise ValueError(f"{MODEL_POLICY}: unsupported result dtype {reference.dtype}") from None
