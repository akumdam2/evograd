"""Deterministic Speed-of-Light triage for NCU metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineResult:
    bottleneck: str
    efficiency_pct: float
    at_roofline: bool
    compute_pct: float | None
    memory_pct: float | None
    notes: tuple[str, ...]


def analyze(metrics: dict[str, float], threshold: float = 95.0) -> RooflineResult:
    compute = metrics.get("sm_throughput_pct")
    memory = metrics.get("dram_throughput_pct")
    available = [value for value in (compute, memory) if value is not None and value >= 0]
    efficiency = max(available, default=0.0)
    notes = []
    if compute is None:
        notes.append("compute SOL metric is missing")
    if memory is None:
        notes.append("DRAM SOL metric is missing")
    if not available:
        bottleneck = "unknown"
    elif memory is not None and (compute is None or memory > compute + 5):
        bottleneck = "memory"
    elif compute is not None and (memory is None or compute > memory + 5):
        bottleneck = "compute"
    else:
        bottleneck = "balanced"
    occupancy = metrics.get("occupancy_pct")
    if occupancy is not None and occupancy < 35:
        notes.append(f"low achieved occupancy ({occupancy:.1f}%)")
    stalls = metrics.get("long_scoreboard_stall_ratio")
    if stalls is not None and stalls > 0.25:
        notes.append(f"high long-scoreboard stall ratio ({stalls:.3f})")
    return RooflineResult(
        bottleneck=bottleneck,
        efficiency_pct=float(efficiency),
        at_roofline=bool(available and efficiency >= threshold),
        compute_pct=compute,
        memory_pct=memory,
        notes=tuple(notes),
    )


def format_roofline(result: RooflineResult) -> str:
    compute = "missing" if result.compute_pct is None else f"{result.compute_pct:.1f}%"
    memory = "missing" if result.memory_pct is None else f"{result.memory_pct:.1f}%"
    notes = "; ".join(result.notes) or "none"
    return (
        f"bottleneck={result.bottleneck}; compute_SOL={compute}; "
        f"DRAM_SOL={memory}; peak_efficiency={result.efficiency_pct:.1f}%; "
        f"at_roofline={result.at_roofline}; notes={notes}"
    )
