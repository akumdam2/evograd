#!/usr/bin/env python3
"""NVIDIA stand-in for GEAK's ``scripts/gpu_identity.py`` (AMD rocminfo probe).

GEAK's Director runs ``python3 $WORKFLOW_DIR/../scripts/gpu_identity.py`` and
requires ``gfx``, ``target``, ``device_name`` and ``physical_cu_count``. On an
NVIDIA box there is no rocminfo; this reports the visible CUDA device in the
same shape, so the workflow's identity checks pass and its knowledge routing
takes the generic (non-RDNA, non-CDNA-specific) path:

    {"gfx": "sm90", "target": "unknown", "device_name": "NVIDIA GH200 120GB",
     "physical_cu_count": 132, "vendor": "NVIDIA", ...}

``gfx`` carries the CUDA compute capability (``sm<major><minor>``), which
GEAK uses only as an identity string; ``physical_cu_count`` is the SM count.
"""

from __future__ import annotations

import json
import subprocess
import sys


def probe() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"gfx": f"sm{props.major}{props.minor}", "target": "unknown",
                    "device_name": props.name, "physical_cu_count": int(props.multi_processor_count),
                    "vendor": "NVIDIA", "source": "torch.cuda"}
    except Exception:
        pass
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20)
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit("no NVIDIA GPU visible (nvidia-smi failed)")
    names = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    if len(names) != 1:
        raise SystemExit(f"mixed visible GPUs {sorted(names)}; restrict CUDA_VISIBLE_DEVICES")
    name, capability = [part.strip() for part in names.pop().split(",")]
    return {"gfx": "sm" + capability.replace(".", ""), "target": "unknown", "device_name": name,
            "physical_cu_count": 132 if "H100" in name or "GH200" in name else 0,
            "vendor": "NVIDIA", "source": "nvidia-smi"}


if __name__ == "__main__":
    json.dump(probe(), sys.stdout)
    sys.stdout.write("\n")
