#!/usr/bin/env python3
"""Measure formal eager/whole-model compile baselines without evaluating candidates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def measure(model_name, out_root, *, device="cuda", force=False):
    from evograd.evaluation import protocol as P
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import configure
    from evograd.verification.integrations.gates import _write, open_run

    root = Path(out_root).expanduser().resolve() / model_name
    plan_path = root / "all_agents.json"
    plan = json.loads(plan_path.read_text())
    if plan["model"] != model_name:
        raise ValueError("saved plan names another model")
    jobs = plan["jobs"]
    if not jobs:
        raise ValueError("saved plan contains no jobs")
    corpus = jobs[0]["selection"]["corpus"]
    if any(job["selection"]["corpus"] != corpus for job in jobs):
        raise ValueError("saved agents used different corpora")
    reports = [json.loads(path.read_text()) for job in jobs
               if (path := Path(job["out"]) / "evaluation.json").exists()]
    if not reports:
        raise ValueError("no previous evaluation.json available to recover timing settings")
    model = load(model_name)
    options = reports[0]["settings"]["tier3"]
    for report in reports:
        if report.get("model") != model_name or report["workload"] != dict(model.workload_config):
            raise ValueError("saved workload differs from the current formal workload")
        if report["settings"]["tier3"] != options:
            raise ValueError("saved agents used different Tier 3 timing settings")
    configure()
    output = root / "shared" / "tier3" / "baselines_only.json"
    with tempfile.TemporaryDirectory(prefix="evograd-baselines-") as directory:
        run = open_run(model, {}, corpus, directory, device=device)
        binding = {"model": model_name, "workload": dict(model.workload_config),
                   "corpus": run.corpus.binding(), "options": options,
                   "environment": P._environment(),
                   "implementation": P.baseline_implementation(run.workload)}
        if output.exists() and not force:
            previous = json.loads(output.read_text())
            if previous.get("baseline_binding") == binding and previous.get("baseline_complete"):
                print(f"Reused baseline report: {output}", flush=True)
                return 0
        print(f"Measuring eager and whole-model torch.compile for {model_name}; "
              "candidate verification and calibration are not run", flush=True)
        report = P.run_tier3(run, {"eager": "eager", "torch_compile": "torch_compile"},
                             options=options)
        providers = report.get("providers") or {}
        complete = bool(providers.get("eager", {}).get("ok")) and P.compile_baseline_valid(
            providers.get("torch_compile", {}))
        report.update(baseline_binding=binding, baseline_complete=complete,
                      source_plan=str(plan_path), baseline_only=True)
        _write(output, report)
        print(f"Baseline report: {output}", flush=True)
        if not complete:
            print("Baseline execution or compile correctness failed; see the provider report.",
                  file=sys.stderr)
        return 0 if complete else 1


def main(argv=None):
    # Set the same offline data/compiler environment as the evaluation launcher
    # before importing torch. Spawned providers inherit it and the source path.
    import evaluate_model as launcher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=launcher.MODELS)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="remeasure an existing valid baseline")
    args = parser.parse_args(argv)
    os.environ.update(launcher.environment(args.model))
    sys.path.insert(0, str(launcher.REPO / "src"))
    return measure(args.model, args.out_root, device=args.device, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
