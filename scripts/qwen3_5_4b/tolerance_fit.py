"""Fit Level-2 / Level-1 tolerances from calibration inventories.

Used by ``scripts/qwen3_5_4b/calibration.py fit``.

Reads ``calibration/inventory/<task>.json`` (oracle vs runtime_forward only:
grid, observed, derived, skewed) and proposes, per task, the declaration the
dense models use:

* ``tolerances`` = the calibrated base ``{"float32": (2e-5, 2e-5),
  "bfloat16": (2e-2, 2e-2)}``;
* ``tolerance_multipliers`` = per result, the measured minimum atol multiplier
  at that base on the correctness grid, times 1.5, rounded up to one decimal
  (none below 1.0);
* ``tolerance_hook`` = ``ReductionScaledAtol`` anchored at the largest
  bfloat16 grid case: parameter gradients reduce over the dims of the
  activations they are summed over, every result has its element count. The
  gain is the smallest of a fixed ladder for which the grid multipliers cover
  every real-size case with the same 1.5 margin; if none does, the gain stays
  2.0 and the multiplier is raised to cover the real-size case (flagged).

Nothing from a library baseline or a candidate is read. Writes
``calibration/fit.json`` and ``calibration/fit.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BASE = {"float32": (2e-5, 2e-5), "bfloat16": (2e-2, 2e-2)}
#: bfloat16 base atols tried, largest first. As for the dense models, the base
#: is set by the forward outputs: the smallest base their grid measurement fits
#: under with the margin. A task whose results are small (the delta rule's
#: gradients are ~0.1) would otherwise be gated by an atol the size of its signal.
BF16_BASE_ATOLS = (2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
                   5e-6, 2e-6, 1e-6)
MARGIN = 1.5
#: One bfloat16 ulp, relative: 2**-8.
BF16_ULP = 2.0 ** -8
GAINS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
#: A (result, case) pair counts toward the over-grant objective only if it
#: needs at least this fraction of the base atol; below it the base floor, not
#: the fit, sets the tolerance.
MEANINGFUL_NEED = 0.1


def ceil1(x: float) -> float:
    return math.ceil(x * 10.0 - 1e-9) / 10.0


def shape_dims(shape) -> tuple[str, ...]:
    if not shape or shape == "unspecified":
        return ()
    inner = shape.strip()[1:-1].strip()
    return tuple(p.strip() for p in inner.split(",") if p.strip() and not p.strip().isdigit())


#: Where the generic rule below misnames what a parameter gradient sums over:
#: a head axis folded into a parameter's width, or per-head norm weights that
#: also sum over heads.
REDUCTION_OVERRIDES = {
    "qwen3_5_gdn_norm_out": {"dw_out": ("B", "T")},
    "qwen3_5_gated_attention": {"do_weight": ("B", "T")},
    "qwen3_5_qkv_norm_partial_rope": {"dq_norm_weight": ("B", "HQ", "T"),
                                      "dk_norm_weight": ("B", "HK", "T")},
}


def hook_layout(op):
    """reduction_dims and result_dims for every result of ``op``.

    A parameter gradient sums over the activation axes no parameter has --
    the token axes (B, T / N / rows / M), and the routed slot K for the
    experts. Activations and outputs carry only their element count.
    """
    params = set(op.parameter_args)
    active = list(op.active_args())
    activation_dims, parameter_dims = set(), set()
    for arg in active:
        dims = set(shape_dims(getattr(arg, "shape", None)))
        (parameter_dims if arg.name in params else activation_dims).update(dims)
    tokens = activation_dims - parameter_dims
    overrides = REDUCTION_OVERRIDES.get(op.name, {})
    result_dims, reduction = {}, {}
    for out in op.outputs:
        result_dims[out.name] = shape_dims(getattr(out, "shape", None))
    for arg in active:
        dims = shape_dims(getattr(arg, "shape", None))
        result_dims[arg.grad_name] = dims
        if arg.name in params:
            reduced = tuple(sorted(tokens - set(dims)))
            if reduced:
                reduction[arg.grad_name] = reduced
    for name, reduced in overrides.items():
        reduction[name] = tuple(reduced)
    return reduction, result_dims


def need(record, dtype, base=None) -> float:
    """atol needed at the base rtol, as a multiple of the base atol."""
    atol, rtol = (base or BASE)[dtype]
    table = record.get("required_atol_by_rtol") or {}
    key = f"{rtol:g}"
    if key not in table:
        raise KeyError(f"inventory lacks required_atol_by_rtol[{key}] (re-run calibration)")
    return table[key] / atol


def fit_task(path: Path) -> dict:
    from evograd.benchmark import TASKS
    from evograd.opdecl.tolerance import ReductionScaledAtol

    report = json.loads(path.read_text())
    task_report = report["tasks"][0]
    op = TASKS[task_report["task"]]
    cases = [c for c in task_report["cases"] if c.get("results") and not c.get("error")]
    # The grid is also measured on the CPU (``--grid-only --device cpu``): the
    # declarations are checked there by the unit tests, and CPU bfloat16
    # rounds differently, so the grid multipliers must hold on both devices.
    cpu = path.parent.parent / "inventory_cpu" / path.name
    if cpu.exists():
        for c in json.loads(cpu.read_text())["tasks"][0]["cases"]:
            if c.get("results") and not c.get("error"):
                cases.append({**c, "label": c["label"].replace("correctness",
                                                               "correctness[cpu]", 1)})
    grid = [c for c in cases if c["label"].startswith("correctness")]
    real = [c for c in cases if not c["label"].startswith("correctness")]
    bf16_grid = [c for c in grid if c["dtype"] == "bfloat16"]
    anchor_case = max(bf16_grid, key=lambda c: math.prod(c["dims"].values()))
    anchor = dict(anchor_case["dims"])
    reduction, result_dims = hook_layout(op)
    names = list(cases[0]["results"])

    # bfloat16 base atol: the smallest step at or above both 1.5x what the
    # forward outputs need on the grid and one bfloat16 ulp of the outputs'
    # magnitude -- so a result that production computes exactly still admits
    # an implementation that rounds differently near zero.
    raw = {"float32": BASE["float32"], "bfloat16": BASE["bfloat16"]}
    output_need = max((need(c["results"][n], "bfloat16", raw) * raw["bfloat16"][0]
                       for c in bf16_grid for n in op.output_names), default=0.0)
    output_scale = max((c["results"][n]["ref_absmax"] for c in bf16_grid
                        for n in op.output_names), default=0.0)
    floor = max(MARGIN * output_need, BF16_ULP * output_scale)
    base_atol = BF16_BASE_ATOLS[0]
    for candidate in BF16_BASE_ATOLS:
        if floor <= candidate:
            base_atol = candidate
    base = {"float32": BASE["float32"], "bfloat16": (base_atol, BASE["bfloat16"][1])}

    def hook(gains, mult=None):
        if not isinstance(gains, dict):
            gains = {n: gains for n in names}
        return ReductionScaledAtol(anchor_dims=anchor, reduction_dims=reduction,
                                   result_dims=result_dims, gain=0.0, result_gain=dict(gains),
                                   dtype_multipliers=mult or {})

    def multiplier(name, population, gain, dtype):
        h = hook(gain)
        worst = max((need(c["results"][name], c["dtype"], base) / h.factor(name, c["dims"])
                     for c in population if c["dtype"] == dtype), default=0.0)
        return max(1.0, ceil1(MARGIN * worst))

    # Each dtype's multipliers come from that dtype's grid. Each result's gain
    # is then the smallest that carries its bfloat16 multiplier over the
    # real-size cases (all bfloat16) with the margin; if even the largest falls
    # short, the bfloat16 multiplier is raised to cover them and flagged.
    gains, mult, flagged = {}, {"float32": {}, "bfloat16": {}}, []
    for name in names:
        for g in GAINS:
            m = multiplier(name, grid, g, "bfloat16")
            h = hook({**{n: 0.0 for n in names}, name: g})
            short = [c for c in real
                     if MARGIN * need(c["results"][name], c["dtype"], base)
                     > m * h.factor(name, c["dims"]) * (1 + 1e-9)]
            if not short:
                gains[name], mult["bfloat16"][name] = g, m
                break
        else:
            g = GAINS[-1]
            gains[name] = g
            mult["bfloat16"][name] = multiplier(name, grid + real, g, "bfloat16")
            flagged.append(name)
        mult["float32"][name] = multiplier(name, grid, gains[name], "float32")
    h = hook(gains, mult)

    rows = []
    for c in cases:
        for name in names:
            have = mult[c["dtype"]][name] * h.factor(name, c["dims"])
            n = need(c["results"][name], c["dtype"], base)
            rows.append({"case": c["label"], "dtype": c["dtype"], "result": name,
                         "need": n, "have": have, "margin": (have / n) if n > 0 else None})
    assert all(r["margin"] is None or r["margin"] >= MARGIN - 1e-6 for r in rows), op.name
    # Over-grant on the generated real-size cases (what preflight runs). The
    # derived case's magnitudes are the model's own and may differ from the
    # generator's, so one absolute atol cannot be tight on both; its margins
    # are reported per case in ``coverage``.
    real_rows = [r for r in rows if not r["case"].startswith(("correctness", "derived"))
                 and r["margin"] is not None and r["need"] >= MEANINGFUL_NEED]
    worst_slack = max((r["margin"] for r in real_rows), default=1.0)

    relative = {}
    for c in real:
        for name in names:
            r = c["results"][name]
            atol = base[c["dtype"]][0] * mult[c["dtype"]][name] * h.factor(name, c["dims"])
            relative.setdefault(name, []).append({
                "case": c["label"], "atol": atol, "ref_absmax": r["ref_absmax"],
                "atol_over_ref_absmax": atol / r["ref_absmax"] if r["ref_absmax"] else None,
                "need_over_ref_absmax": (need(r, c["dtype"], base) * base[c["dtype"]][0]
                                         / r["ref_absmax"]) if r["ref_absmax"] else None,
                "noise_max_abs": c["production_noise_max_abs"].get(name),
                "max_abs_err": r["max_abs_err"]})
    noise_floor_violations = [
        (c["label"], n) for c in cases for n in names
        if (c["production_noise_max_abs"].get(n) or 0.0)
        > base[c["dtype"]][0] * mult[c["dtype"]][n] * h.factor(n, c["dims"])]
    return {
        "task": op.name, "anchor_dims": anchor, "result_gain": gains,
        "worst_over_grant": worst_slack,
        "tolerances": {k: list(v) for k, v in base.items()},
        "dtype_multipliers": {d: {n: m for n, m in ms.items() if m > 1.0}
                              for d, ms in mult.items()},
        "all_multipliers": mult,
        "reduction_dims": reduction, "result_dims": result_dims,
        "flagged_real_size_sets_multiplier": flagged,
        "noise_floor_violations": noise_floor_violations,
        "cases": [c["label"] for c in cases],
        "grid_need_max": {d: {n: max((need(c["results"][n], d, base) for c in grid
                                      if c["dtype"] == d), default=0.0) for n in names}
                          for d in ("float32", "bfloat16")},
        "real_need_max": {n: max((need(c["results"][n], c["dtype"], base) for c in real),
                                 default=None) for n in names},
        "coverage": rows,
        "relative": relative,
    }
