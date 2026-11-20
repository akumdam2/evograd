"""Gate 1 tolerances: one fixed budget per dtype against a float32 truth, for
model-reference tasks.

The truth ``t`` is the model reference evaluated with every floating input below
float32 widened to float32 (TF32 off). A result passes on a call when

    max|candidate - t| <= bound * max|t|

computed in float64, where ``max|t|`` is the largest magnitude of that result on
that call: every element gets the same allowance. ``bound`` is the larger of the
dtype's ``budget`` and ``NATIVE_FACTOR`` times the largest deviation of the model's
own implementation (the reference at the call's real dtypes, "native") from the
truth on the judged calls, ``max|native - t| / max|t|`` taken over every call. One
rule for every task, the same form as operator Gate 2's
(:mod:`evograd.verification.gate2.operator`): ``err(candidate) <= max(floor,
factor * err(trusted))``. A result whose truth is exactly zero passes only when
the candidate is exactly zero too.

Where native is itself less accurate than the budget, the result is
ill-conditioned at that dtype, and a candidate is held to native's accuracy
instead. On the benchmark that is the attention tasks: each row of the softmax
gradient sums to zero, so ``dq_i = scale * sum_j dS_ij k_j`` is what is left after
a component common to all keys cancels, and the q/k-norm weight gradients sum
such residues over every token and head. Ordinary bfloat16 rounding (0.2-0.4% per
element) then moves these results by several percent, up to tens of percent where
the keys carry a large common component: native itself is up to 8% off the truth
on Qwen3's and Qwen3.5's attention blocks and up to 89% on Qwen2-VL's text
attention. Everywhere else ``NATIVE_FACTOR`` times native's error stays within the
budget on the benchmark's cases (2026-11-19), so the budget decides.

Why a float32 truth. Native carries its own rounding (a bfloat16 RMSNorm weight
gradient is 2-3% of its largest value off), so judging against it penalizes
implementations that are more accurate than the model. Float32 differs from a
float64 evaluation by at most 4e-4 of the result's largest value on the
benchmark's cases (4e-3 on one ill-conditioned attention block), at least 30
times below native's own error, and runs every kernel the model uses (float64
does not: grouped matmul has no float64 kernel).

Why ``max|t|`` and not ``|t_i|``. A small element is usually the difference of
larger terms (a reduction over tokens, a softmax backward); its rounding error is
set by those terms, not by its own size. There is no absolute term: a recorded
gradient can be 1e-13 as a whole (MegaFold), and any fixed absolute allowance
would accept a wrong kernel, even all zeros, on such results.

The budget follows the least precise floating dtype of the call (its tensor
inputs, upstream gradients and the result): a float32 gradient computed from
bfloat16 activations carries bfloat16 rounding, whatever its storage dtype.

Gate 1 is the search-time check that a kernel is written correctly; it needs no
calibration and never consults ``torch.compile``. Small faults are operator
Gate 2's job. A gradient scaled by 1.05 moves its largest element by 5% of
``max|t|``, so any bfloat16 budget below that rejects it where native is accurate.

Gate 1 judges only model-reference tasks (``op.reference_contract``): for any
other declaration every function here raises.
"""

from __future__ import annotations

import torch

from evograd.opdecl.activity import Active
from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

MODEL_POLICY = "evograd-gate1-model-dtype-budget/10"
SOURCE = "Gate 1's own fixed budgets (2026-11-19)"
COMPARISON = ("max|c-t| <= bound * max|t| per result and call, float64; t = float32 truth; "
              "bound = max(budget, NATIVE_FACTOR * native's own error)")

#: The bound is never below this many times native's own error (FlashAttention's
#: tests hold a kernel to twice the error of the PyTorch implementation).
NATIVE_FACTOR = 2.0
#: Per dtype (the least precise floating dtype of the call): fraction of max|t|.
#: float16 is bfloat16's at its eight times finer rounding and is not measured.
BUDGETS = {
    torch.float16: 5e-3,
    torch.bfloat16: 4e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-7,
}


def model_policy_identity() -> dict:
    """Serializable policy content, included in reports and workflow bindings."""
    return {
        "schema": MODEL_POLICY,
        "source": SOURCE,
        "dtype_source": "least precise floating dtype of the call's tensors and the result",
        "truth": "model reference with floating inputs below float32 widened to float32, TF32 off",
        "native_factor": NATIVE_FACTOR,
        "budgets": {str(dtype).removeprefix("torch."): budget for dtype, budget in BUDGETS.items()},
        "declaration_scaling": False,
        "comparison": COMPARISON,
    }


def _require_model_reference(op) -> None:
    if not op.reference_contract:
        raise ValueError(f"{op.name}: Gate 1 judges only a model-reference task "
                         "(op.reference_contract); this declaration has none")


def policy_identity(op) -> dict:
    _require_model_reference(op)
    return model_policy_identity()


def call_dtypes(op, workload) -> list[torch.dtype]:
    """The declared floating dtypes of the call's tensors: tensor arguments and upstream gradients."""
    entries = [a for a in op.args if isinstance(a, Active) or getattr(a, "is_tensor", False)]
    dtypes = [resolve_dtype(arg_dtype_name(a, workload)) for a in (*entries, *op.outputs)]
    return [d for d in dtypes if d.is_floating_point]


def rule_dtype(op, workload, dtype: torch.dtype) -> torch.dtype:
    """The least precise of the result's dtype and the call's floating dtypes."""
    _require_model_reference(op)
    if dtype not in BUDGETS:
        raise ValueError(f"{MODEL_POLICY}: unsupported result dtype {dtype}")
    candidates = [dtype, *(d for d in call_dtypes(op, workload) if d in BUDGETS)]
    return max(candidates, key=lambda d: torch.finfo(d).eps)


def result_budget(op, workload, dtype: torch.dtype) -> float:
    """The budget for a result of ``dtype`` (see :func:`rule_dtype`)."""
    return BUDGETS[rule_dtype(op, workload, dtype)]


def result_bound(budget: float, native_error: float) -> float:
    """A result's fraction of max|t|: the larger of the budget and ``NATIVE_FACTOR``
    times native's own error (never stricter than the model's own implementation
    allows, and continuous in native's error)."""
    return max(budget, NATIVE_FACTOR * native_error)


def allowance(truth: torch.Tensor, bound: float) -> float:
    """``bound * max|truth|``: every element's allowance (0 for an all-zero truth)."""
    scale = float(truth.detach().abs().max().double()) if truth.numel() else 0.0
    return float(bound) * scale
