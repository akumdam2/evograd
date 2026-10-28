"""Gate 1 tolerances: fixed dtype defaults for model-reference tasks.

The table is frozen from torch.testing.assert_close in PyTorch 2.11, rather
than imported from a private PyTorch helper whose defaults could change.
Values are (atol, rtol), unlike PyTorch's internal (rtol, atol) helper.
Model tasks ignore declaration overrides, result multipliers and shape hooks.
Generic mathematical tasks retain their existing declaration policy.

Under the model policy an element's allowance is the larger of that rule and
the *envelope*: :data:`ENVELOPE_GAMMA` times the largest elementwise deviation
the trusted ``torch.compile`` control showed on the same result over the
calibration payloads (operator Gate 2's calibration, orders 101/202; Gate 1
judges the acceptance payloads 303/404). ``assert_close``'s rule alone judges a near-zero
element against nothing but ``atol``, and a result that is the difference of
two larger terms (rotary embedding) then fails on one bf16 rounding of the
operands -- as the compiler's own implementation does.
"""

from __future__ import annotations

import torch

MODEL_POLICY = "evograd-gate1-model-dtype-tolerances/4"
#: Gate 1's own safety factor on the compile envelope, separate from operator Gate 2's
#: (2): Gate 1 only has to accept every trusted implementation, rejecting small faults is
#: Gate 2's job. The smallest integer at which every frontend seed passes on every
#: benchmark model (2026-10-27): Qwen2-VL-2B's vision-MLP seed needs 4 (6 of 6.5M
#: up-projection weight-gradient elements off by twice the operator factor's envelope; its
#: tensor-level error against a float64 reference is 1.8x native bf16's). Every other
#: seed needs at most 0.17.
ENVELOPE_GAMMA = 4.0
ENVELOPE = ("ENVELOPE_GAMMA x max elementwise |compile - native| of the result on the operator "
            "Gate 2 calibration payloads")
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
        "envelope": ENVELOPE,
        "envelope_gamma": ENVELOPE_GAMMA,
        "comparison": "|c-r| <= max(atol + rtol * |r|, envelope) per element, float64",
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
