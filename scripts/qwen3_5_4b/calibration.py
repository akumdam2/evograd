"""Qwen3.5-4B Level-2 tolerance calibration: magnitudes, inventory, fit, apply.

    python scripts/qwen3_5_4b/calibration.py derived-stats --run-dir <run> [--device cpu|cuda]
    python scripts/qwen3_5_4b/calibration.py inventory     --run-dir <run> [--cpu-grid]
    python scripts/qwen3_5_4b/calibration.py fit           --run-dir <run>
    python scripts/qwen3_5_4b/calibration.py apply         --run-dir <run> [--task T ...]

The fit (:mod:`tolerance_fit`): a bfloat16 base atol set by the forward
outputs, per-dtype multipliers at 1.5x the grid need, a ``ReductionScaledAtol``
whose per-result gains carry the grid to the model's size. The apply
(:mod:`tolerance_apply`): a delimited generated block in each declaration --
the Level-2 sites' ``task.py`` and the bound Level-1 primitives'
``ops/level1/<name>/__init__.py``.

Populations, each ``op.forward`` against ``op.runtime_forward`` only (never a
library baseline or a candidate): the declared grid (GPU, and the CPU with
``--cpu-grid``), the ``qwen3_5_4b_observed`` case, and the site's real
invocation from the captured layer (``EVOGRAD_QWEN3_5_LAYER_DIR``, default the
baseline run's ``layer14.pt`` / ``layer15.pt``). One task per process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MODULE = "evograd.evaluation.workloads.qwen3_5_4b.level2.calibrate"
LAYER_DIR = REPO / "results/experiments/qwen3_5_4b_cases/20261010_step1"

#: The Level-1 primitives this model binds, calibrated on its own populations.
LEVEL1 = ("causal_conv1d", "gated_rmsnorm", "gdn_gating", "gemma_rmsnorm", "l2norm",
          "partial_rope")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/qwen3_5_4b" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[calibration {time.strftime('%H:%M:%S')}] {message}", flush=True)


def tasks() -> list[str]:
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest

    return [*manifest.TASKS, *LEVEL1]


# ── derived-stats ────────────────────────────────────────────────────────────


def cmd_derived_stats(args) -> int:
    import torch

    from evograd.benchmark import TASKS
    from evograd.evaluation.workloads.qwen3_5_4b.level2.harvested import captured_invocation
    from evograd.opdecl.inputs import make_case_inputs

    def stats(t):
        if not torch.is_tensor(t) or not t.is_floating_point():
            return None
        f = t.detach().float()
        return {"rms": float(f.pow(2).mean().sqrt()), "absmax": float(f.abs().max())}

    out_path = args.run_dir / "derived_stats.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for task in args.task or tasks():
        if task in out:
            continue
        op = TASKS[task]
        values, workload, label = captured_invocation(op, args.device, layer_dir=str(args.layer_dir))
        generated = make_case_inputs(op, workload, device=args.device)
        out[task] = {"label": label, "tensors": {
            name: {"captured": stats(values[name]), "generated": stats(generated.get(name))}
            for name in values if stats(values[name]) is not None}}
        log(task + " " + json.dumps({
            n: (round(v["captured"]["rms"], 4),
                round(v["generated"]["rms"], 4) if v["generated"] else None)
            for n, v in out[task]["tensors"].items()}))
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        del values, generated
    return 0


# ── inventory ────────────────────────────────────────────────────────────────


def _run_inventory(run_dir: Path, phase: str, task: str, extra: list[str], env: dict,
                   timeout: int = 2400) -> bool:
    out_dir = run_dir / "calibration" / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"{task}.DONE"
    if marker.exists():
        log(f"{phase} {task}: done; skipping")
        return True
    cmd = [sys.executable, "-m", MODULE, "inventory", "--task", task,
           "--report", str(out_dir / f"{task}.json"), *extra]
    log(f"{phase} {task}")
    tick = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        (out_dir / f"{task}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        ok = proc.returncode == 0 and (out_dir / f"{task}.json").exists()
    except subprocess.TimeoutExpired:
        ok = False
        (out_dir / f"{task}.log").write_text(f"killed after {timeout}s\n")
    wall = time.perf_counter() - tick
    if ok:
        marker.write_text(f"{wall:.1f}s\n")
    log(f"{phase} {task}: {'ok' if ok else 'FAILED'} in {wall:.0f}s")
    return ok


def cmd_inventory(args) -> int:
    env = dict(os.environ)
    env["EVOGRAD_QWEN3_5_LAYER_DIR"] = str(args.layer_dir)
    # The children import this checkout, whatever PYTHONPATH the environment set.
    env["PYTHONPATH"] = str(REPO / "src")
    ok = True
    for task in args.task or tasks():
        if args.cpu_grid:
            ok &= _run_inventory(args.run_dir, "inventory_cpu", task,
                                 ["--device", "cpu", "--grid-only"], env)
        else:
            ok &= _run_inventory(args.run_dir, "inventory", task, ["--device", "cuda"], env)
    return 0 if ok else 1


# ── fit and apply ────────────────────────────────────────────────────────────


def cmd_fit(args) -> int:
    fit_module = _load("tolerance_fit")
    inventory = args.run_dir / "calibration" / "inventory"
    fits = []
    for path in sorted(inventory.glob("*.json")):
        try:
            fits.append(fit_module.fit_task(path))
        except Exception as exc:  # noqa: BLE001 - recorded per task
            fits.append({"task": path.stem, "error": f"{type(exc).__name__}: {exc}"})
    out = args.run_dir / "calibration"
    (out / "fit.json").write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n")
    for f in fits:
        if "error" in f:
            log(f"{f['task']}: fit error {f['error']}")
            continue
        gains = {k: v for k, v in f["result_gain"].items() if v}
        log(f"{f['task']}: bf16 base {f['tolerances']['bfloat16']}, gains {gains}, "
            f"bf16 multipliers {f['dtype_multipliers']['bfloat16']}, flagged "
            f"{f['flagged_real_size_sets_multiplier']}, worst over-grant "
            f"{f['worst_over_grant']:.1f}x, noise-floor violations {f['noise_floor_violations']}")
    return 0 if all("error" not in f for f in fits) else 1


def cmd_apply(args) -> int:
    apply_module = _load("tolerance_apply")
    run = args.run_dir.name
    fits = {f["task"]: f for f in json.loads((args.run_dir / "calibration" / "fit.json").read_text())}
    for task, fit in sorted(fits.items()):
        if args.task and task not in args.task:
            continue
        if "error" in fit:
            log(f"{task}: fit error, not applied: {fit['error']}")
            continue
        path = apply_module.declaration_path(task)
        apply_module.apply(path, apply_module.block(fit, run, fit["cases"]))
        log(f"{task}: {path.relative_to(REPO)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("derived-stats", "inventory", "fit", "apply"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-grid", action="store_true")
    parser.add_argument("--layer-dir", type=Path, default=LAYER_DIR)
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    return {"derived-stats": cmd_derived_stats, "inventory": cmd_inventory,
            "fit": cmd_fit, "apply": cmd_apply}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
