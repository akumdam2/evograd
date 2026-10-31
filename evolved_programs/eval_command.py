#!/usr/bin/env python3
"""Print the formal evaluation command for one agent's programs on one model, from this folder.

    python evolved_programs/eval_command.py AGENT MODEL --adbench ADBENCH_ROOT --out OUT_ROOT \\
        [--deadline-min 45] [--tier3-blocks 10]

Run from the evograd repository root, on a tree with the patches the programs' site used
(patches/<site>/; at least deployment_single_grad and deployment_scalars, which the evaluation's
deployment calls depend on). The command is the one `candidates/bench.py evaluate` builds from a
search root, with the programs taken from this folder instead of the search's run directories:
results go to OUT_ROOT/MODEL/evaluation/AGENT, reusable controls to OUT_ROOT/MODEL/state.
Exit 3 from the command means its deadline stopped it; rerun the same command.
"""
import argparse
import csv
import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent")
    p.add_argument("model")
    p.add_argument("--adbench", type=Path, required=True, help="ADBench root holding cases.json")
    p.add_argument("--out", type=Path, required=True, help="evaluation output root")
    p.add_argument("--deadline-min", type=float, default=45)
    p.add_argument("--tier3-blocks", type=int, default=10)
    args = p.parse_args()
    corpus = json.loads((args.adbench / "cases.json").read_text())[args.model]["corpus"]
    rows = [r for r in csv.DictReader((HERE / args.agent / "manifest.csv").open(encoding="utf-8"))
            if r["model"] == args.model and r["program"]]
    if not rows:
        raise SystemExit(f"no {args.agent} programs for {args.model} in {HERE / args.agent}")
    root = args.out / args.model
    command = ["python", "-m", "evograd.evaluation.protocol", args.model, "--corpus", corpus,
               "--out", str(root / "evaluation" / args.agent), "--state", str(root / "state"),
               "--tier3-blocks", str(args.tier3_blocks), "--deadline-min", str(args.deadline_min)]
    for r in sorted(rows, key=lambda r: r["case_id"]):
        command += ["--candidate", f"{r['case_id']}={HERE / args.agent / r['program']}"]
    print("CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src " + shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
