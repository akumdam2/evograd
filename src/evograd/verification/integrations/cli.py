"""The command line every model integration shares:
``python -m evograd.verification.integrations.<model> {capture,fidelity,verify}``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(model, argv=None) -> int:
    parser = argparse.ArgumentParser(prog=f"evograd.verification.integrations.{model.name}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture", help="record the real-payload corpus").add_argument("--out", required=True)
    p = sub.add_parser("fidelity", help="replay every payload against the model reference")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify", help="run the gates on candidate artifacts")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidate", action="append", required=True, metavar="CASE_ID=PROGRAM.py")
    p.add_argument("--controls", default=None, help="Gate 3 control runs to reuse (default DIR/controls)")
    args = parser.parse_args(argv)

    def log(*parts):
        print(time.strftime("%H:%M:%S"), *parts, flush=True)

    if args.cmd == "capture":
        from .base import capture

        print(json.dumps({"manifest": str(capture(model, args.out, log=log))}))
        return 0
    if args.cmd == "fidelity":
        from .base import replay_fidelity

        report = replay_fidelity(model, args.corpus, log=log)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("corpus_id", "payloads", "by_task", "layouts_ok",
                                                 "outputs_bitwise")}))
        return 0 if report["layouts_ok"] and report["outputs_bitwise"] else 2
    from .gates import verify

    candidates = dict(item.split("=", 1) for item in args.candidate)
    verdict = verify(model, candidates, args.corpus, args.out, controls=args.controls, log=log)
    print(json.dumps({case: entry["accepted"] for case, entry in verdict["candidates"].items()}))
    return 0 if all(entry["accepted"] for entry in verdict["candidates"].values()) else 1

