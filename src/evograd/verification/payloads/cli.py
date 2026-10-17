"""Judge one program on one real-payload split with canonical Gate 1.

    python -m evograd.verification.payloads.cli gate1 --declaration D.py:op --program P.py \\
        --corpus CORPUS --case CASE_ID --split acceptance --out report.json

Exit status 0 only when the report passed. A missing or incompatible corpus is
a setup failure (exit 3, ``status: incomplete``), never a fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location(f"payload_cli_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate1_report(declaration: str, program: str, corpus_path: str, case_id: str, split: str,
                 device: str = "cuda") -> dict:
    from evograd.benchmark import load_task
    from evograd.verification import gate1
    from evograd.verification.common.reports import jsonable
    from evograd.verification.payloads import Corpus, CorpusError, coverage, payload_cases

    op = load_task(declaration)
    try:
        corpus = Corpus.open(corpus_path)
        if len(op.correctness) != 1:
            raise CorpusError(f"{op.name}: real payloads judge one configuration")
        cases = payload_cases(op, op.correctness[0], corpus, case_id=case_id, split=split, device=device)
    except (CorpusError, OSError, ValueError, KeyError) as exc:
        return {"status": "incomplete", "setup_error": f"{type(exc).__name__}: {exc}",
                "inputs": "real_training_payload"}
    report = gate1.verify_pair(op, _load_module(program), cases=cases, device=device)
    return {"status": report.status, "complete": report.complete, "inputs": "real_training_payload",
            "payload_binding": corpus.binding(split),
            "coverage": coverage(corpus, case_id=case_id, split=split),
            "report": jsonable(report.to_dict())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evograd.verification.payloads.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate1")
    for name in ("declaration", "program", "corpus", "case", "split", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = gate1_report(args.declaration, args.program, args.corpus, args.case, args.split, args.device)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result.get(k) for k in ("status", "setup_error")}))
    if result.get("setup_error"):
        return 3
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
