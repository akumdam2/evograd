"""GEAK profiler-mcp contract backed by Evograd's NVIDIA NCU profiler."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from evograd.geak_nvidia.profile_adapter import profile_for_geak

mcp = FastMCP(
    name="profiler",
    instructions="NVIDIA Nsight Compute profiling for Evograd Triton candidates.",
)


def _candidate(workdir: str | None) -> Path:
    roots = [
        Path(os.environ["GEAK_WORK_DIR"]) if os.environ.get("GEAK_WORK_DIR") else None,
        Path(workdir) if workdir else None,
        Path.cwd(),
    ]
    for root in roots:
        if root is None:
            continue
        candidate = root / "candidate.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate candidate.py in GEAK workdir")


@mcp.tool()
def profile_kernel(
    command: str,
    backend: str = "metrix",
    workdir: str | None = None,
    profiling_type: str = "profiling",
    num_replays: int = 1,
    kernel_filter: str | None = None,
    auto_select: bool = False,
    quick: bool = True,
    gpu_devices: str | list[str] | None = None,
    warmup_runs: int = 2,
) -> dict[str, Any]:
    """Profile the current candidate with NCU, preserving GEAK's tool schema."""
    del backend, profiling_type, num_replays, kernel_filter, auto_select, quick, gpu_devices
    try:
        payload = profile_for_geak(
            _candidate(workdir),
            rows=int(os.environ.get("GEAK_NCU_ROWS", "4096")),
            hidden=int(os.environ.get("GEAK_NCU_HIDDEN", "1024")),
            dtype=os.environ.get("GEAK_NCU_DTYPE", "bfloat16"),
            warmup=max(1, int(warmup_runs)),
            timeout=int(os.environ.get("GEAK_PROFILE_TIMEOUT", "120")),
        )
    except Exception as exc:
        return {
            "success": False,
            "backend": "nvidia-ncu",
            "error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }
    payload["invocation"] = {"command": command, "workdir": workdir}
    return payload


if __name__ == "__main__":
    mcp.run()
