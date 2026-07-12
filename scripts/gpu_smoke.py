"""GPU smoke test for the opdecl stack. Run on a CUDA node:

    # oracle only: shapes/dtypes of ground-truth grads for every op
    PYTHONPATH=src python scripts/gpu_smoke.py --oracle-only

    # verify a candidate seed against the oracle
    PYTHONPATH=src python scripts/gpu_smoke.py --op evoattention --candidate path/to/seed.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_smoke(device: str) -> int:
    from evograd.opdecl import make_case_inputs, oracle
    from evograd.ops import OPS

    failed = []
    for name, op in sorted(OPS.items()):
        if not op.correctness:
            print(f"[skip] {name}: no correctness workloads declared yet")
            continue
        workload = op.correctness[0]
        try:
            inputs = make_case_inputs(op, workload, device=device)
            y, grads = oracle(op, inputs)
            shapes = {g: tuple(t.shape) for g, t in grads.items()}
            print(f"[ok]   {name}: y{tuple(y.shape)} grads {shapes}")
        except Exception as exc:  # noqa: BLE001 — smoke test reports and moves on
            failed.append(name)
            print(f"[FAIL] {name}: {exc}")
    return 1 if failed else 0


def verify_candidate(op_name: str, candidate: Path, device: str) -> int:
    from evograd.opdecl import verify
    from evograd.ops import get_op

    report = verify(get_op(op_name), load_module(candidate), device=device)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", help="operator name (see evograd.ops.OPS)")
    parser.add_argument("--candidate", type=Path, help="path to a seed module to verify")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.oracle_only:
        return oracle_smoke(args.device)
    if not (args.op and args.candidate):
        parser.error("either --oracle-only, or both --op and --candidate")
    return verify_candidate(args.op, args.candidate, args.device)


if __name__ == "__main__":
    sys.exit(main())
