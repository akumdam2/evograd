"""Plot best-so-far speedup against iteration, one line per seed pipeline.

    python scripts/plot_evolution.py --csv curves.csv --out .

Consumes the CSV written by scripts/collect_evolution.py and emits one PNG per
operator: x is the evolution iteration, y is the best correct full-step speedup
found so far, one line per pipeline (median across trials) with a min-max band.

Best-so-far rather than per-candidate speedup, because that is what "how the
speedup grows" means for a search: the population's current best is what the run
would return if stopped at that iteration. Full-step rather than backward-only,
because a backward-only ratio compares a baseline that includes its forward
against a candidate that does not, inflating the number by roughly 1.5x.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

# Fixed categorical slots, assigned by pipeline identity and never cycled, so a
# pipeline keeps its colour across every figure in the experiment. Validated as
# a 4-slot categorical palette on the light surface (worst adjacent CVD dE 9.1,
# normal-vision dE 22.9); aqua and yellow sit under 3:1 against the surface,
# which is why every line is also directly labelled.
PALETTE = {
    "A": "#2a78d6",
    "B": "#eb6834",
    "C": "#1baf7a",
    "D": "#eda100",
    "E": "#e87ba4",
    "F": "#008300",
    "G": "#4a3aa7",
    "H": "#e34948",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def _read(csv_path: Path) -> dict[str, dict[str, dict[int, dict[int, float]]]]:
    """op -> pipeline -> trial -> {iteration: best_so_far}."""
    curves: dict[str, dict[str, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("best_so_far")
            if not value:
                # No correct candidate yet in this trial; the curve starts later.
                continue
            curves[row["op"]][row["pipeline"]][int(row["trial"])][int(row["iteration"])] = float(
                value
            )
    return curves


def _aggregate(
    trials: dict[int, dict[int, float]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Median / min / max across trials at each iteration they all reached.

    Trials are aggregated only over the ones that actually reached an iteration:
    extending a trial that died early with its last value would draw a flat line
    where no measurement exists.
    """
    iterations = sorted({i for series in trials.values() for i in series})
    xs, medians, lows, highs = [], [], [], []
    for iteration in iterations:
        values = [series[iteration] for series in trials.values() if iteration in series]
        if not values:
            continue
        xs.append(iteration)
        medians.append(statistics.median(values))
        lows.append(min(values))
        highs.append(max(values))
    return xs, medians, lows, highs


def _plot(op: str, pipelines: dict[str, dict[int, dict[int, float]]], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 5), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    # Recessive grid, drawn under the data.
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)

    # Parity with the measured baseline: below this line the candidate is slower.
    # Named in the caption rather than annotated in place, where it would land on
    # whichever series happens to cross 1.0.
    axes.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.6, zorder=1)

    trial_counts, ends = [], []
    for pipeline in sorted(pipelines):
        colour = PALETTE.get(pipeline, INK_MUTED)
        xs, medians, lows, highs = _aggregate(pipelines[pipeline])
        if not xs:
            continue
        trial_counts.append(len(pipelines[pipeline]))
        axes.fill_between(xs, lows, highs, color=colour, alpha=0.13, linewidth=0, zorder=2)
        axes.plot(xs, medians, color=colour, linewidth=2, label=f"pipeline {pipeline}", zorder=3)
        axes.plot(xs[-1], medians[-1], marker="o", markersize=5, color=colour, zorder=4)
        ends.append((medians[-1], xs[-1], pipeline))

    axes.set_xlabel("evolution iteration", fontsize=10, color=INK_MUTED)
    axes.set_ylabel("best-so-far full-step speedup", fontsize=10, color=INK_MUTED)
    axes.tick_params(colors=INK_MUTED, labelsize=9)
    axes.set_title(
        f"{op} — seed pipeline ablation",
        fontsize=13,
        color=INK,
        loc="left",
        pad=18,
        fontweight="bold",
    )
    if not trial_counts:
        trials = "0"
    elif min(trial_counts) == max(trial_counts):
        trials = str(trial_counts[0])
    else:
        trials = f"{min(trial_counts)}–{max(trial_counts)}"
    axes.annotate(
        f"median across {trials} trial(s) · band spans min–max"
        " · dashed line marks baseline parity",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=9,
        color=INK_MUTED,
    )
    legend = axes.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK)

    # Room on the right for the direct labels, which sit outside the last point.
    left, right = axes.get_xlim()
    axes.set_xlim(left, right + (right - left) * 0.08)

    # Direct labels: a coloured mark carries identity, the text stays in ink.
    # Required relief for the palette slots under 3:1 contrast on this surface —
    # so they must stay legible when two pipelines finish at the same speedup.
    # Placed after the limits are final, because the stagger is in data units.
    if ends:
        low, high = axes.get_ylim()
        gap = (high - low) * 0.05
        x_label = max(x for _, x, _ in ends) + (axes.get_xlim()[1] - left) * 0.015
        placed: list[float] = []
        for value, _x_end, pipeline in sorted(ends):
            y = value if not placed else max(value, placed[-1] + gap)
            placed.append(y)
            axes.text(
                x_label,
                y,
                pipeline,
                fontsize=10,
                fontweight="bold",
                color=INK,
                va="center",
                ha="left",
                clip_on=False,
            )

    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{op}_evolution.png"
    figure.savefig(target, facecolor=SURFACE)
    plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="CSV from collect_evolution.py")
    parser.add_argument("--out", type=Path, required=True, help="directory for the PNGs")
    parser.add_argument("--op", action="append", default=None, help="restrict to op (repeatable)")
    args = parser.parse_args(argv)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib is not installed (it is not an evograd dependency).\n"
            "  pip install matplotlib\n"
            f"The data is already in {args.csv} if you would rather plot elsewhere.",
        )
        return 1

    if not args.csv.is_file():
        parser.error(f"no such file: {args.csv}")

    curves = _read(args.csv)
    if not curves:
        print(f"{args.csv} holds no correct candidate — nothing to plot")
        return 1

    for op, pipelines in sorted(curves.items()):
        if args.op and op not in args.op:
            continue
        print(f"wrote {_plot(op, pipelines, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
