"""Gate 1 tolerances: one fixed budget per dtype against a float32 truth, for
model-reference tasks.

The truth ``t`` is the model reference evaluated with every floating input below
float32 widened to float32 (TF32 off). An element of a result passes when

    |candidate - t| <= atol + bound * max|t|

computed in float64, where ``max|t|`` is the largest magnitude of that result on
that call. Every element is checked; the allowance is the same for all elements
of the result. ``bound`` is the dtype's ``budget``.

Attention tasks (:data:`NATIVE_BOUND_TASKS`) are the one exception: there
``bound`` is the larger of the budget and ``NATIVE_FACTOR`` times the largest
deviation of the model's own implementation (the reference at the call's real
dtypes, "native") from the truth on the judged calls. Their query and key
gradients are ill-conditioned at low precision: each row of the softmax gradient
sums to zero, so ``dq_i = scale * sum_j dS_ij k_j`` is what is left after a
component common to all keys cancels, and the q/k-norm weight gradients sum such
residues over every token and head. Ordinary bfloat16 rounding (0.2-0.4% per
element) then moves these results by several percent, up to tens of percent where
the keys carry a large common component (a k_proj bias, a large k-norm weight on a
slowly rotating RoPE dimension): native itself is up to 8% off the truth on
Qwen3's and Qwen3.5's attention blocks and up to 89% on Qwen2-VL's text attention.
A candidate there is held to native's accuracy instead of the budget. No other
benchmark task needs it: native is within the budget everywhere else.

Why a float32 truth. Native carries its own rounding (a bfloat16 RMSNorm weight
gradient is 2-3% of its largest value off), so judging against it penalizes
implementations that are more accurate than the model. Float32 differs from a
float64 evaluation by at most 4e-4 of the result's largest value on the
benchmark's cases (4e-3 on one ill-conditioned attention block), at least 30
times below native's own error, and runs every kernel the model uses (float64
does not: grouped matmul has no float64 kernel).

Why ``max|t|`` and not ``|t_i|``. A small element is usually the difference of
larger terms (a reduction over tokens, a softmax backward); its rounding error is
set by those terms, not by its own size.

``atol`` is absolute and tiny on purpose: it only absorbs results that are
numerically zero (rounding residue near 1e-14). Recorded gradients themselves
reach down to 1e-13, so any larger absolute term would accept a wrong kernel on
whole results.

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

MODEL_POLICY = "evograd-gate1-model-dtype-budget/9"
SOURCE = "Gate 1's own fixed budgets (2026-11-19)"
COMPARISON = ("|c-t| <= atol + bound * max|t| per element, float64; t = float32 truth; "
              "bound = budget, or on attention tasks max(budget, NATIVE_FACTOR * native's own error)")

#: Absolute term, every dtype.
ATOL = 1e-12
#: On an attention task the bound is never below this many times native's own error.
NATIVE_FACTOR = 1.5
#: Attention tasks of the benchmark (op names): the softmax over keys makes their
#: query/key gradients ill-conditioned at low precision (module docstring).
NATIVE_BOUND_TASKS = frozenset({
    "qwen3_attention_block", "qwen3_5_attention_block", "llama3_attention_block",
    "olmoe_attention_block", "qwen2_vl_2b_text_attention_block", "qwen2_vl_2b_vision_attention",
    "dit_s_2_dit_attention", "megafold_evoattention", "megafold_triangle_attention_starting",
    "megafold_triangle_attention_ending", "softmax",
})
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
        "atol": ATOL,
        "native_factor": NATIVE_FACTOR,
        "native_bound_tasks": sorted(NATIVE_BOUND_TASKS),
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


def native_bound(op) -> bool:
    """Whether ``op`` is an attention task, whose bound follows native's own error."""
    _require_model_reference(op)
    return op.name in NATIVE_BOUND_TASKS


def result_bound(budget: float, native_error: float) -> float:
    """An attention task's fraction of max|t|: the larger of the budget and
    ``NATIVE_FACTOR`` times native's own error (never stricter than the model's own
    implementation allows, and continuous in native's error)."""
    return max(budget, NATIVE_FACTOR * native_error)


def allowance(truth: torch.Tensor, bound: float) -> float:
    """``atol + bound * max|truth|``: every element's allowance."""
    scale = float(truth.detach().abs().max().double()) if truth.numel() else 0.0
    return ATOL + float(bound) * scale
