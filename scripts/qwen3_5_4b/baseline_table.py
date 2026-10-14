"""The baseline table (Markdown) from a run directory: numbers and status only.

    python scripts/qwen3_5_4b/baseline_table.py results/experiments/qwen3_5_4b_cases/<run_id>

Reads what the driver wrote (``l3_b1.json``, ``l3_b2.json``,
``t2/*.json``, ``l3/*.json``, ``l3_identity.json``) and lays it out. No bar is chosen
and no speedup is judged; a missing input is printed as missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _ms(value) -> str:
    return "-" if value is None else f"{value:.3f}" if value < 10 else f"{value:.1f}"


def _gib(value) -> str:
    return "-" if value is None else f"{value / 2**30:.1f}"


def l3_feasibility(run: Path) -> list[str]:
    lines = ["## L3 feasibility (eager, fwd + bwd + AdamW, 3 steps, step 1 = warmup)", "",
             "| batch | status | peak allocated GiB | peak reserved GiB | warm step ms | tokens/s |",
             "| --- | --- | --- | --- | --- | --- |"]
    for batch in (1, 2):
        data = _load(run / f"l3_b{batch}.json")
        if data is None:
            lines.append(f"| B{batch} x T2048 | missing | - | - | - | - |")
            continue
        status = "OOM" if data.get("oom") else "ok"
        peak = data.get("at_failure", {}).get("peak_allocated_bytes") if data.get("oom") \
            else data.get("peak_allocated_bytes")
        reserved = data.get("at_failure", {}).get("peak_reserved_bytes") if data.get("oom") \
            else data.get("peak_reserved_bytes")
        tps = data.get("tokens_per_second_warm")
        lines.append(f"| B{batch} x T2048 | {status} | {_gib(peak)} | {_gib(reserved)} | "
                     f"{_ms(data.get('warm_step_ms_mean'))} | "
                     f"{'-' if tps is None else f'{tps:.0f}'} |")
    return lines + [""]



def t2(run: Path) -> list[str]:
    lines = ["## T2 (Level-2 tasks, observed case, do_bench median ms of fwd + bwd)", "",
             "| task | provider | full step ms | forward ms | status |",
             "| --- | --- | --- | --- | --- |"]
    paths = sorted(p for p in (run / "t2").glob("*.json") if not p.name.endswith("nocheck.json"))
    if not paths:
        return lines + ["| missing | | | | |", ""]
    for path in paths:
        report = _load(path) or {}
        diagnostic = _load(path.with_name(path.stem + ".nocheck.json")) or {}
        diag_rows = {}
        for case in diagnostic.get("cases", []):
            diag_rows.update(case.get("providers", {}))
        for case in report.get("cases", []):
            for name, row in sorted(case.get("providers", {}).items()):
                if row.get("ok"):
                    lines.append(f"| {report.get('op')} | {name} | "
                                 f"{_ms(row.get('full_step', {}).get('median_ms'))} | "
                                 f"{_ms(row.get('forward', {}).get('median_ms'))} | ok |")
                    continue
                diag = diag_rows.get(name) or {}
                if diag.get("ok"):
                    lines.append(f"| {report.get('op')} | {name} | "
                                 f"{_ms(diag.get('full_step', {}).get('median_ms'))} | "
                                 f"{_ms(diag.get('forward', {}).get('median_ms'))} | "
                                 f"gate refused (provisional tol.); timed with --no-check, "
                                 f"diagnostic only |")
                else:
                    lines.append(f"| {report.get('op')} | {name} | - | - | "
                                 f"{str(row.get('error'))[:80]} |")
    return lines + [""]



def l3_baselines(run: Path) -> list[str]:
    lines = ["## L3 (tier-3 model scope, B1 x T2048, AdamW; median step ms)", "",
             "Each run is its own tier-3 invocation (own eager row); compare a provider "
             "with the eager row of the same run.", "",
             "| run | provider | step ms | tokens/s | peak GiB | evaluation_complete | first loss / last of 20 | status |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    found = False
    for name in ("main", "whole_model_compile"):
        report = _load(run / "l3" / f"{name}.json")
        if report is None:
            marker = _load(run / "l3" / f"{name}.DONE") or {}
            lines.append(f"| {name} | - | - | - | - | - | - | no report (rc={marker.get('rc')}) |")
            continue
        found = True
        for provider, row in (report.get("providers") or {}).items():
            losses = row.get("losses") or []
            loss_text = (f"{losses[0]:.4f} / {losses[-1]:.4f}" if losses else "-")
            status = "ok" if row.get("ok", True) and row.get("step_ms") is not None else \
                f"{row.get('failed_at') or ''} {str(row.get('error'))[:60]}"
            ups = row.get("units_per_second")
            lines.append(f"| {name} | {provider} | {_ms(row.get('step_ms'))} | "
                         f"{'-' if ups is None else f'{ups:.0f}'} | "
                         f"{_gib(row.get('peak_memory_bytes'))} | "
                         f"{row.get('evaluation_complete')} | {loss_text} | {status} |")
    if not found:
        lines.append("")
    return lines + [""]


def l3_identity(run: Path) -> list[str]:
    data = _load(run / "l3_identity.json")
    lines = ["## L3 structural identity (one training step, every parameter gradient compared)", "",
             "| SDPA mode | comparison | loss bitwise | params not bitwise | worst param | worst rel_l2 |",
             "| --- | --- | --- | --- | --- | --- |"]
    if data is None:
        return lines + ["| missing | | | | | |", ""]
    for mode in ("default", "deterministic_math_sdpa"):
        for pair in ("eager_vs_eager", "identity_vs_eager"):
            entry = (data.get(mode) or {}).get(pair) or {}
            worst = entry.get("worst_rel_l2")
            lines.append(f"| {mode} | {pair} | {entry.get('loss_bitwise')} | "
                         f"{entry.get('params_not_bitwise')}/{entry.get('params')} | "
                         f"{entry.get('worst_param') or '-'} | "
                         f"{'-' if worst is None else f'{worst:.2e}'} |")
    return lines + [""]


def build_table(run: Path) -> str:
    run = Path(run)
    head = [f"# Qwen3.5-4B baselines ({run.name})", "",
            "Numbers and completion status only; no bar is chosen. The L3 whole-model gate "
            "is not implemented, so every L3 row says `evaluation_complete: False`.", ""]
    body = l3_feasibility(run) + t2(run) + l3_baselines(run) + l3_identity(run)
    return "\n".join(head + body) + "\n"


if __name__ == "__main__":
    print(build_table(Path(sys.argv[1])))
