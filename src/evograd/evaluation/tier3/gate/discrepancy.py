"""What a failed tensor comparison has to say for itself.

A verdict that reports only "out: 0.5 > atol 0.0195" cannot be acted on: it
does not say how many elements disagree, whether they are the tensor's largest
values or its cancellations, or what the reference had there. This module turns
one failing comparison into a record that does, in a single streaming pass:

    shape, dtype, element count, finiteness, bitwise equality
    max |actual - reference|, relative L2, reference RMS and peak
    the elementwise rule that was applied and its atol/rtol
    violations and violating fraction
    a few representative violating elements -- the worst by error-to-allowance
      ratio -- each with its coordinate, the actual and reference values, the
      allowance at that element, and the ratio

and, because the scale-normalized metrics of the Tier-3 budget experiment are
the same two numbers divided by the reference RMS, ``e_rms`` and ``e_max``
alongside, recorded as diagnostics with no threshold attached.

Chunked deliberately: these run on live model activations (a [2, 16, 2048, 128]
gradient is 8M elements) inside a step that is already holding the model, so the
pass allocates a bounded temporary rather than a float32 copy of the tensor.
"""

from __future__ import annotations

import math
from typing import Any

import torch

#: Elements per chunk. 4M float32 temporaries are ~16 MiB each.
CHUNK = 1 << 22
#: How many violating elements one record keeps.
REPRESENTATIVES = 6
#: float32's smallest normal, the floor under an elementwise allowance.
TINY = torch.finfo(torch.float32).tiny


def _coordinate(flat: int, shape: tuple[int, ...]) -> list[int]:
    out: list[int] = []
    for size in reversed(shape):
        out.append(flat % size)
        flat //= size
    return list(reversed(out))


def elementwise_record(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float,
                       name: str = "", rule: str = "allclose(atol, rtol)",
                       reference: str | None = None, role: str | None = None,
                       representatives: int = REPRESENTATIVES, chunk: int = CHUNK,
                       **context: Any) -> dict[str, Any]:
    """One streaming pass over ``actual`` against ``expected``. Retains no tensor."""
    record: dict[str, Any] = {
        "tensor": name, "role": role, "reference": reference,
        "rule": {"kind": "elementwise", "expression": "|actual - reference| <= atol + rtol * |reference|",
                 "atol": float(atol), "rtol": float(rtol), "detail": rule},
        "shape": list(expected.shape), "dtype": str(expected.dtype).removeprefix("torch."),
        "elements": int(expected.numel()), **context,
    }
    if not torch.is_tensor(actual):
        record.update(ok=False, present=False, reason=f"not a tensor: {type(actual).__name__}")
        return record
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        record.update(ok=False, present=True,
                      actual_shape=list(actual.shape), actual_dtype=str(actual.dtype).removeprefix("torch."),
                      reason="shape or dtype differs")
        return record

    flat_actual = actual.detach().reshape(-1)
    flat_expected = expected.detach().reshape(-1)
    total = int(flat_expected.numel())
    err_sq = ref_sq = 0.0
    max_abs = 0.0
    violations = 0
    finite = True
    bitwise = True
    ref_absmax = 0.0
    kept: list[tuple[float, int]] = []
    for start in range(0, total, max(1, chunk)):
        a = flat_actual[start:start + chunk]
        b = flat_expected[start:start + chunk]
        bitwise = bitwise and bool(torch.equal(a, b))
        a32 = a.to(torch.float32)
        b32 = b.to(torch.float32)
        if not bool(torch.isfinite(a32).all()):
            finite = False
        diff = (a32 - b32).abs()
        allowance = (atol + rtol * b32.abs()).clamp_min(TINY)
        ratio = diff / allowance
        over = ratio > 1.0
        count = int(over.sum())
        violations += count
        err_sq += float(diff.to(torch.float64).pow(2).sum())
        ref_sq += float(b32.to(torch.float64).pow(2).sum())
        if diff.numel():
            max_abs = max(max_abs, float(diff.max()))
            ref_absmax = max(ref_absmax, float(b32.abs().max()))
        if count and representatives:
            values, indices = torch.topk(ratio, min(representatives, ratio.numel()))
            kept.extend((float(v), start + int(i)) for v, i in zip(values, indices) if float(v) > 1.0)
        del a, b, a32, b32, diff, allowance, ratio, over

    ref_rms = math.sqrt(ref_sq / total) if total else 0.0
    rel_l2 = math.sqrt(err_sq) / max(math.sqrt(ref_sq), 1e-30)
    kept.sort(reverse=True)
    shape = tuple(expected.shape)
    examples = []
    for ratio_value, flat in kept[:representatives]:
        want = float(flat_expected[flat].to(torch.float32))
        got = float(flat_actual[flat].to(torch.float32))
        allowance = max(atol + rtol * abs(want), float(TINY))
        examples.append({"coordinate": _coordinate(flat, shape), "flat_index": flat,
                         "actual": got, "reference": want, "abs_error": abs(got - want),
                         "allowed": allowance, "error_over_allowance": ratio_value})
    record.update({
        "ok": bool(finite and violations == 0),
        "present": True, "finite": finite, "bitwise": bitwise,
        "max_abs_error": max_abs, "rel_l2": rel_l2,
        "reference_rms": ref_rms, "reference_absmax": ref_absmax,
        "violations": violations,
        "violation_fraction": (violations / total) if total else 0.0,
        "violating_examples": examples,
        # The scale-normalized metrics of the budget experiment, reported only.
        "scale_normalized": {"e_rms": rel_l2,
                             "e_max": (max_abs / ref_rms) if ref_rms else None,
                             "note": "reported diagnostic; no threshold applied here"},
    })
    if not record["ok"]:
        record["reason"] = ("non-finite values" if not finite else
                            f"{violations} of {total} elements exceed atol {atol:g} + rtol {rtol:g} * |reference| "
                            f"(worst {kept[0][0]:.4g}x the allowance)" if kept else
                            f"{violations} of {total} elements exceed the allowance")
    return record


def aggregate_record(metric: str, value: float, threshold: float | None, *,
                     rule: str, reference: str | None = None,
                     localization: Any = None, **context: Any) -> dict[str, Any]:
    """One whole-tensor / whole-model metric against its frozen threshold."""
    record = {
        "metric": metric, "reference": reference,
        "rule": {"kind": "aggregate", "expression": rule, "threshold": threshold},
        "value": float(value),
        "threshold": (float(threshold) if threshold is not None else None),
        "ratio": (float(value) / float(threshold)) if threshold else None,
        "ok": (None if threshold is None else bool(float(value) <= float(threshold))),
        **context,
    }
    if localization is not None:
        record["localization"] = localization
    return record


__all__ = ["CHUNK", "REPRESENTATIVES", "aggregate_record", "elementwise_record"]
