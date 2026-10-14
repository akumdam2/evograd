"""Write fitted tolerances into each declaration (``scripts/qwen3_5_4b/calibration.py apply``).

Reads ``calibration/fit.json`` (``fit_tolerances.py``) and, per task, replaces
the delimited calibrated-tolerances block in its declaration file -- or, the
first time, inserts it before ``op = declare_op(`` and points the declaration
at it. Re-runnable: a re-calibrated task is re-applied in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BEGIN = "# --- calibrated tolerances (scripts/qwen3_5_4b/calibration.py apply) ---"
END = "# --- end calibrated tolerances ---"


def declaration_path(task: str) -> Path:
    if task.startswith("qwen3_5_"):
        site = task.removeprefix("qwen3_5_")
        return (REPO / "src/evograd/benchmark/topdown/qwen3_5_4b/levels/level2"
                / site / "task.py")
    return REPO / "src/evograd/ops/level1" / task / "__init__.py"


def fmt(x: float) -> str:
    return f"{x:g}"


def block(fit: dict, run: str, cases: list[str]) -> str:
    rows = []
    names = sorted(fit["all_multipliers"]["bfloat16"])
    for n in names:
        real = fit["real_need_max"].get(n)
        margins = [r["margin"] for r in fit["coverage"]
                   if r["result"] == n and r["margin"] is not None
                   and not r["case"].startswith("correctness")]
        rows.append(f"#:   {n:<16} {fit['grid_need_max']['float32'][n]:>8.3g} "
                    f"{fit['grid_need_max']['bfloat16'][n]:>8.3g} "
                    f"{(real if real is not None else 0.0):>9.3g} "
                    f"{fit['all_multipliers']['float32'][n]:>5g} "
                    f"{fit['all_multipliers']['bfloat16'][n]:>5g} "
                    f"{fit['result_gain'].get(n, 0.0):>5g} "
                    f"{(min(margins) if margins else float('nan')):>8.3g}")
    base = fit["tolerances"]
    gains = {n: g for n, g in sorted(fit["result_gain"].items()) if g}
    dtype_mult = {d: {n: m for n, m in sorted(ms.items())}
                  for d, ms in sorted(fit["dtype_multipliers"].items()) if ms}
    populations = ", ".join(sorted({c.split(" ")[0] for c in cases}))
    lines = [
        BEGIN,
        "from evograd.opdecl.tolerance import ReductionScaledAtol  # noqa: E402",
        "",
        f"#: Calibrated for Qwen3.5-4B (run ``{run}``; GH200, grid also on CPU):",
        "#: ``op.forward`` against ``op.runtime_forward`` only -- never a library or",
        f"#: a candidate -- on {populations};",
        "#: four repeats each (production was deterministic). The bfloat16 base",
        "#: atol is the smallest 1-2-5 step at or above both 1.5x the forward",
        "#: outputs' grid need and one bfloat16 ulp of their magnitude; rtol 2e-2.",
        "#: Multipliers are per dtype (in the hook): 1.5x that dtype's grid need,",
        "#: rounded up (none below 1). Each gain is the smallest that carries the",
        "#: bfloat16 multiplier over the real-size cases with the same 1.5x.",
        "#: ``need`` is the atol required at the base rtol, in units of the base atol.",
        "#:",
        "#:   result           fp32 grid bf16 grid real need fp32x bf16x  gain margin@real",
        *rows,
        "CALIBRATED = True",
        f"CALIBRATED_TOLERANCES = {{\"float32\": ({fmt(base['float32'][0])}, "
        f"{fmt(base['float32'][1])}), \"bfloat16\": ({fmt(base['bfloat16'][0])}, "
        f"{fmt(base['bfloat16'][1])})}}",
        "#: Per-dtype multipliers live in the hook; none apply to every dtype.",
        "CALIBRATED_MULTIPLIERS = {}",
        "CALIBRATED_HOOK = ReductionScaledAtol(",
        f"    anchor_dims={json.dumps(fit['anchor_dims'], sort_keys=True)},",
        "    reduction_dims={" + ", ".join(
            f"\"{n}\": {tuple(v)!r}" for n, v in sorted(fit["reduction_dims"].items())) + "},",
        "    result_dims={" + ", ".join(
            f"\"{n}\": {tuple(v)!r}" for n, v in sorted(fit["result_dims"].items())) + "},",
        "    gain=0.0,",
        "    result_gain={" + ", ".join(f"\"{n}\": {fmt(g)}" for n, g in gains.items()) + "},",
        "    dtype_multipliers={" + ", ".join(
            f"\"{d}\": {{" + ", ".join(f"\"{n}\": {fmt(m)}" for n, m in ms.items()) + "}"
            for d, ms in dtype_mult.items()) + "},",
        ")",
        END,
    ]
    return "\n".join(lines) + "\n"


def apply(path: Path, text_block: str) -> None:
    s = path.read_text()
    if "# --- calibrated tolerances (" in s:
        s = re.sub(r"# --- calibrated tolerances \(.*?\) ---\n.*?" + re.escape(END) + r"\n",
                   lambda m: text_block, s, flags=re.S)
    else:
        anchor = "\nop = declare_op("
        assert s.count(anchor) == 1, path
        s = s.replace(anchor, "\n" + text_block + "\n\nop = declare_op(")
        # The declaration's own tolerance kwargs now point at the block.
        s, n = re.subn(r"\n    tolerances=PROVISIONAL,\n",
                       "\n    tolerances=CALIBRATED_TOLERANCES,\n"
                       "    tolerance_multipliers=CALIBRATED_MULTIPLIERS,\n"
                       "    tolerance_hook=CALIBRATED_HOOK,\n", s)
        if n == 0:
            s, n = re.subn(r"\n    tolerances=\{\"float32\": STANDARD_TOLERANCES\[\"float32\"\],\n"
                           r"                \"bfloat16\": STANDARD_TOLERANCES\[\"bfloat16\"\]\},\n"
                           r"(    tolerance_hook=_tolerance,\n)?",
                           "\n    tolerances=CALIBRATED_TOLERANCES,\n"
                           "    tolerance_multipliers=CALIBRATED_MULTIPLIERS,\n"
                           "    tolerance_hook=CALIBRATED_HOOK,\n", s)
        assert n == 1, f"{path}: tolerance kwargs not found"
        # The provisional hook it replaces.
        s = re.sub(r"\n\ndef _tolerance\(.*?\n(?=\n\n\S)", "\n", s, flags=re.S)
        s = re.sub(r"#: Not calibrated: the tolerances below are the standard defaults.\n",
                   "#: Calibrated: see the calibrated-tolerances block below.\n", s)
    s = re.sub(r"^PROVISIONAL_TOLERANCES = True$", "PROVISIONAL_TOLERANCES = False", s, flags=re.M)
    path.write_text(s)
