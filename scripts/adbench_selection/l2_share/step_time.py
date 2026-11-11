"""Eager training-step time of one model under the formal Tier 3 settings (2026-11-10, L2-5).

Record of a measurement, kept as it ran (its paths are DeltaAI's). Run 2026-11-10 on gh096, one
NVIDIA GH200 120GB, code at 3397ea9 (/u/wzhan/evograd): six models by run.sh, Qwen3.5-4B by
run_q35.sh (--mem=64g), OLMoE separately. Output: steps/<model>.json beside this file, the
step times (T3 and T3-G) report.py divides by.

Times the eager provider only, eagerly launched (T3) and replayed from a CUDA graph (T3-G),
in a spawned child as the protocol does, and writes the provider row to OUT/<model>.json.

    python step_time.py MODEL OUTDIR
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/u/wzhan/evograd/candidates")


def main() -> int:
    name, out = sys.argv[1], Path(sys.argv[2])
    import evaluate_model as launcher  # no torch import; sets the launcher's offline environment

    os.environ.update(launcher.environment(name))
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.json"
    if target.exists():
        print(f"exists: {target}")
        return 0

    from evograd.evaluation import protocol as P
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import configure
    from evograd.verification.integrations.gates import open_run

    configure()
    corpus = json.load(open("/work/hdd/bcjw/wzhan/ADBench/cases.json"))[name]["corpus"]
    options = {**P.TIER3_OPTIONS, "blocks": 5}
    with tempfile.TemporaryDirectory(prefix="evograd-step-") as directory:
        run = open_run(load(name), {}, corpus, directory, device="cuda")
        report = P.run_tier3(run, {"eager": "eager"}, options=options)
    row = (report.get("providers") or {}).get("eager") or {}
    graph = row.get("cuda_graph") or {}
    summary = {"model": name, "options": options, "ok": row.get("ok"),
               "t3_step_ms": row.get("step_ms"), "t3g_ok": graph.get("ok"),
               "t3g_step_ms": graph.get("step_ms"), "t3g_error": graph.get("error"),
               "error": row.get("error"), "row": row}
    target.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("model", "ok", "t3_step_ms", "t3g_ok", "t3g_step_ms",
                                              "t3g_error", "error")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
