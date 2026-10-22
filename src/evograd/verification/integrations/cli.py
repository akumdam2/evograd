"""The command line every model integration shares:
``python -m evograd.verification.integrations <model> {capture,fidelity,verify}``
(or ``python -m evograd.verification.integrations.<package> ...``).

``verify`` exits 0 when every candidate is accepted, 1 when one is not, and 3
when ``--deadline-min`` stopped it: rerun the same command to continue."""

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
    p.add_argument("--state", "--controls", dest="state", default=None,
                   help="candidate-independent results to reuse: Gate 3 control runs and Gate 2 "
                        "calibrations (default DIR/controls)")
    p.add_argument("--deadline-min", type=float, default=None,
                   help="stop after this many minutes at the next stage boundary or Gate 3 "
                        "checkpoint; exit 3; rerun the same command to continue")
    p.add_argument("--gate3-optimizer", type=json.loads, default=None, metavar="JSON",
                   help="Gate 3 optimizer settings over the model's own, as gate3.adamw keywords, "
                        "e.g. '{\"warmup_steps\": 100, \"clip_grad_norm\": 1.0}'")
    p.add_argument("--gate3-accumulation", type=int, default=None, metavar="K",
                   help="training batches per Gate 3 update (gradient accumulation)")
    p.add_argument("--gate3-steps", type=int, default=None, metavar="N",
                   help="Gate 3 updates per training run")
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
    import dataclasses

    if args.gate3_optimizer:
        model = dataclasses.replace(model, gate3_optimizer={**dict(model.gate3_optimizer),
                                                            **args.gate3_optimizer})
    if args.gate3_accumulation is not None:
        model = dataclasses.replace(model, gate3_accumulation=args.gate3_accumulation)
    if args.gate3_steps is not None:
        model = dataclasses.replace(model, gate3_steps=args.gate3_steps)
    verdict = verify(model, candidates, args.corpus, args.out, state=args.state, log=log,
                     deadline_min=args.deadline_min)
    if verdict.get("suspended"):
        print(json.dumps({"suspended": verdict["suspended"]}))
        return 3
    print(json.dumps({case: entry["accepted"] for case, entry in verdict["candidates"].items()}))
    return 0 if all(entry["accepted"] for entry in verdict["candidates"].values()) else 1

