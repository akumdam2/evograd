"""Turn OpenEvolve program archives into one tidy per-iteration CSV.

    python scripts/collect_evolution.py --root ~/evograd_runs/ablation \
        --out curves.csv --summary summary.json

Reads ``<root>/<op>/<pipeline>/trial_<n>/programs/index.jsonl``, which the
evaluator appends to once per evaluation when ``evograd evolve --save-programs``
is used. Each line carries the full metrics dict, so the whole trajectory of a
run — not just its winner — is recoverable.

On the iteration axis: index.jsonl records no iteration number, but the rendered
OpenEvolve config sets ``parallel_evaluations: 1``, so evaluations are serial
and line order is iteration order. Line 0 is the seed's own evaluation, which is
why every trial of one pipeline should agree on the iteration-0 speedup — the
summary reports that spread as a check on the shared baseline cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterator

# The evaluator emits backward speedup under two names depending on the code
# path (scoring.score_from_aggregate vs evaluator._baseline_metrics).
BACKWARD_KEYS = ("backward_speedup", "speedup")
FIELDS = [
    "op",
    "pipeline",
    "trial",
    "iteration",
    "correct",
    "duplicate",
    "backward_speedup",
    "full_step_speedup",
    "combined_score",
    "saved_memory_ratio",
    "best_so_far",
    "sha1",
    "time",
]


def _float(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                return None
    return None


def _trials(root: Path, ops: list[str] | None) -> Iterator[tuple[str, str, int, Path]]:
    for op_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if ops and op_dir.name not in ops:
            continue
        for pipeline_dir in sorted(p for p in op_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(pipeline_dir.glob("trial_*")):
                index = trial_dir / "programs" / "index.jsonl"
                if index.is_file():
                    try:
                        number = int(trial_dir.name.split("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    yield op_dir.name, pipeline_dir.name, number, index


def _rows(op: str, pipeline: str, trial: int, index: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best: float | None = None
    for iteration, line in enumerate(index.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write can leave one torn line; the rest is good.
            continue
        metrics = record.get("metrics") or {}
        correct = _float(metrics, "correct") or 0.0
        full_step = _float(metrics, "full_step_speedup")
        # Best-so-far tracks only candidates that passed the correctness gate: a
        # fast wrong kernel is not an improvement.
        if correct >= 1.0 and full_step is not None and full_step > 0.0:
            best = full_step if best is None else max(best, full_step)
        rows.append(
            {
                "op": op,
                "pipeline": pipeline,
                "trial": trial,
                "iteration": iteration,
                "correct": int(correct >= 1.0),
                "duplicate": int(bool(record.get("duplicate"))),
                "backward_speedup": _float(metrics, *BACKWARD_KEYS),
                "full_step_speedup": full_step,
                "combined_score": _float(metrics, "combined_score"),
                "saved_memory_ratio": _float(metrics, "saved_memory_ratio"),
                "best_so_far": best,
                "sha1": record.get("sha1", ""),
                "time": record.get("time", ""),
            }
        )
    return rows


def _iteration_field_note(root: Path) -> str | None:
    """Report whether OpenEvolve's checkpoints carry a real iteration number.

    Line order is the axis this script uses. If checkpoint program records turn
    out to hold an explicit iteration, that is a stronger source and worth
    switching to — so surface it rather than assume.
    """
    for checkpoint in root.glob("*/*/trial_*/checkpoints/checkpoint_*/programs/*.json"):
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        keys = sorted(k for k in payload if "iter" in k.lower() or "generation" in k.lower())
        return f"checkpoint records expose {keys}" if keys else None
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        group = summary.setdefault(row["op"], {}).setdefault(row["pipeline"], {})
        trials = group.setdefault("trials", {})
        trial = trials.setdefault(str(row["trial"]), {})
        if row["iteration"] == 0:
            trial["seed_full_step_speedup"] = row["full_step_speedup"]
        trial["iterations"] = row["iteration"] + 1
        if row["best_so_far"] is not None:
            trial["final_best"] = row["best_so_far"]
            if "iterations_to_beat_seed" not in trial:
                seed = trial.get("seed_full_step_speedup")
                if seed and row["best_so_far"] > seed:
                    trial["iterations_to_beat_seed"] = row["iteration"]

    for pipelines in summary.values():
        for stats in pipelines.values():
            finals = [t["final_best"] for t in stats["trials"].values() if "final_best" in t]
            seeds = [
                t["seed_full_step_speedup"]
                for t in stats["trials"].values()
                if t.get("seed_full_step_speedup")
            ]
            stats["n_trials"] = len(stats["trials"])
            if finals:
                stats["final_best_median"] = statistics.median(finals)
                stats["final_best_min"] = min(finals)
                stats["final_best_max"] = max(finals)
            if seeds:
                stats["seed_speedup_median"] = statistics.median(seeds)
                # Every trial of a pipeline evolves the same seed, so a wide
                # spread here means the baseline moved between runs, not that
                # the seed differs.
                stats["seed_speedup_spread"] = max(seeds) - min(seeds)
                if finals:
                    stats["median_gain_over_seed"] = statistics.median(finals) / statistics.median(
                        seeds
                    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="ablation root directory")
    parser.add_argument("--op", action="append", default=None, help="restrict to op (repeatable)")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--summary", type=Path, default=None, help="JSON summary to write")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"no such directory: {args.root}")

    rows: list[dict[str, Any]] = []
    for op, pipeline, trial, index in _trials(args.root, args.op):
        rows.extend(_rows(op, pipeline, trial, index))

    if not rows:
        print(f"no index.jsonl found under {args.root} — was evolve run with --save-programs?")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    note = _iteration_field_note(args.root)
    if note:
        print(f"note: iteration axis is evaluation order; {note}")

    trials = {(r["op"], r["pipeline"], r["trial"]) for r in rows}
    print(f"\n{len(rows)} evaluation(s) from {len(trials)} trial(s)")
    header = f"{'op':<22}{'seed':<6}{'trials':>7}{'seed x':>9}{'final x':>9}{'gain':>7}"
    print(header)
    print("-" * len(header))
    for op, pipelines in sorted(summary.items()):
        for pipeline, stats in sorted(pipelines.items()):
            seed = stats.get("seed_speedup_median")
            final = stats.get("final_best_median")
            gain = stats.get("median_gain_over_seed")
            print(
                f"{op:<22}{pipeline:<6}{stats['n_trials']:>7}"
                + (f"{seed:>9.3f}" if seed else f"{'—':>9}")
                + (f"{final:>9.3f}" if final else f"{'—':>9}")
                + (f"{gain:>7.2f}" if gain else f"{'—':>7}")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
