"""Convert Evograd NCU profiles into GEAK profiler-mcp's neutral schema."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

from evograd.ncu.profile import ProfileResult, run_ncu_profile
from evograd.ncu.roofline import analyze
from evograd.opdecl.activity import Workload
from evograd.ops import get_op

_FRIENDLY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": (
        "long_scoreboard_stall_ratio"
    ),
    "smsp__sass_inst_executed_op_local_ld.sum": "local_loads",
    "smsp__sass_inst_executed_op_local_st.sum": "local_stores",
}


def _duration_us(value: float, unit: str) -> float:
    normalized = unit.lower().replace(" ", "")
    if normalized in {"ns", "nsecond", "nseconds", "nanosecond", "nanoseconds"}:
        return value / 1000.0
    if normalized in {"ms", "msecond", "mseconds", "millisecond", "milliseconds"}:
        return value * 1000.0
    if normalized in {"s", "second", "seconds"}:
        return value * 1_000_000.0
    return value


def _gpu_info() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()[0]
        name, capability, memory_mib = [part.strip() for part in output.split(",")]
        return {
            "detected": True,
            "vendor": "NVIDIA",
            "model": name,
            "architecture": f"sm{capability.replace('.', '')}",
            "memory_mib": float(memory_mib),
            "warp_size": 32,
        }
    except Exception as exc:
        return {"detected": False, "vendor": "NVIDIA", "error": str(exc)}


def _observations(metrics: dict[str, float], bottleneck: str) -> list[str]:
    observations = [
        (
            f"NCU bottleneck={bottleneck}; SM SOL={metrics.get('sm_throughput_pct', 0.0):.1f}%; "
            f"DRAM SOL={metrics.get('dram_throughput_pct', 0.0):.1f}%."
        )
    ]
    occupancy = metrics.get("occupancy_pct")
    if occupancy is not None:
        observations.append(f"Achieved occupancy proxy: {occupancy:.1f}%.")
    registers = metrics.get("registers_per_thread")
    if registers is not None:
        observations.append(f"Registers per thread: {registers:.1f}.")
    stalls = metrics.get("long_scoreboard_stall_ratio")
    if stalls is not None and stalls > 0.25:
        observations.append(f"Long-scoreboard stall ratio is high ({stalls:.3f}).")
    if metrics.get("register_spilling", 0.0) > 0:
        observations.append("NCU observed local loads/stores; register spilling is likely.")
    return observations


def profile_result_to_geak(result: ProfileResult) -> dict[str, Any]:
    if not result.ok:
        return {
            "success": False,
            "backend": "nvidia-ncu",
            "error": result.error or "NCU profile failed",
            "results": [],
        }

    grouped: dict[str, dict[str, list[tuple[float, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in result.kernels:
        name = str(row.get("kernel") or "layernorm_kernel")
        grouped[name][str(row["metric"])].append(
            (float(row["value"]), str(row.get("unit") or ""))
        )

    kernels = []
    for name, raw_metrics in grouped.items():
        metrics = {}
        duration = 0.0
        for raw_name, observations in raw_metrics.items():
            average = sum(value for value, _unit in observations) / len(observations)
            if raw_name == "gpu__time_duration.sum":
                duration = sum(_duration_us(value, unit) for value, unit in observations)
                continue
            metrics[_FRIENDLY.get(raw_name, raw_name)] = average
        metrics["duration_us"] = duration
        metrics["register_spilling"] = float(
            metrics.get("local_loads", 0.0) > 0 or metrics.get("local_stores", 0.0) > 0
        )
        metrics["memory.hbm_bandwidth_utilization"] = metrics.get(
            "dram_throughput_pct", 0.0
        )
        metrics["compute_util_pct"] = metrics.get("sm_throughput_pct", 0.0)
        metrics["occupancy_pct"] = metrics.get("occupancy_pct", 0.0)
        roofline = analyze(metrics)
        bottleneck = roofline.bottleneck
        if (
            duration > 0
            and duration < 25.0
            and roofline.efficiency_pct < 35.0
        ):
            bottleneck = "latency"
        kernels.append(
            {
                "name": name,
                "duration_us": duration,
                "bottleneck": bottleneck,
                "observations": _observations(metrics, bottleneck),
                "metrics": metrics,
            }
        )

    if not kernels:
        metrics = dict(result.metrics)
        duration = float(metrics.pop("gpu__time_duration.sum", 0.0))
        metrics["duration_us"] = duration
        metrics["memory.hbm_bandwidth_utilization"] = metrics.get(
            "dram_throughput_pct", 0.0
        )
        metrics["compute_util_pct"] = metrics.get("sm_throughput_pct", 0.0)
        roofline = analyze(metrics)
        kernels = [
            {
                "name": "layernorm_autograd_pair",
                "duration_us": duration,
                "bottleneck": roofline.bottleneck,
                "observations": _observations(metrics, roofline.bottleneck),
                "metrics": metrics,
            }
        ]

    return {
        "success": True,
        "backend": "nvidia-ncu",
        "results": [
            {
                "device_id": "0",
                "gpu_info": _gpu_info(),
                "kernels": kernels,
            }
        ],
    }


def profile_for_geak(
    candidate: str | Path,
    *,
    rows: int = 4096,
    hidden: int = 1024,
    dtype: str = "bfloat16",
    warmup: int = 3,
    timeout: int = 120,
) -> dict[str, Any]:
    candidate = Path(candidate).resolve()
    workload = Workload(dims={"rows": rows, "hidden": hidden}, dtype=dtype)
    result = run_ncu_profile(
        get_op("layernorm"),
        candidate,
        workload=workload,
        warmup=warmup,
        timeout=timeout,
    )
    payload = profile_result_to_geak(result)
    payload["request"] = {
        "candidate": str(candidate),
        "dims": workload.dims,
        "dtype": workload.dtype,
        "warmup": warmup,
        "timeout": timeout,
    }
    return payload


def write_profile(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
