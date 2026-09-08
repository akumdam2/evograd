"""One command for the Llama-3 Level-1 checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evograd.benchmark.topdown.llama3_8b.levels.level1.manifest import Level1Error, mapping
from evograd.evaluation.workloads.llama3_8b.level1.calibrate import run_calibration, summarize_calibration
from evograd.evaluation.workloads.llama3_8b.level1.verify import (
    REPORT_SCHEMA,
    run_cross_entropy_check,
    run_loss_check,
    run_verify,
    summarize_cross_entropy,
    summarize_loss,
    summarize_mapping,
    summarize_verify,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.evaluation.workloads.llama3_8b.level1.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    table = sub.add_parser("mapping", help="print the six-task mapping table")
    table.add_argument("--report", type=Path, default=None)

    calibrate = sub.add_parser("calibrate", help="measure a task's required tolerance")
    calibrate.add_argument("--op", default="causal_gqa_attention")
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--report", type=Path, default=None)

    verify = sub.add_parser("verify", help="check causal_gqa_attention against the model")
    verify.add_argument("--source", type=Path, default=Path("results/llama3-level4/layer16.pt"))
    verify.add_argument("--device", default="cuda")
    verify.add_argument("--noise-repeats", type=int, default=4)
    verify.add_argument("--report", type=Path, default=None)

    loss = sub.add_parser(
        "loss", help="ln(vocab) sanity check at the observed shape (not the proof)"
    )
    loss.add_argument("--device", default="cuda")
    loss.add_argument("--report", type=Path, default=None)

    ce = sub.add_parser(
        "cross-entropy", help="compare the Level-1 contract against the model's own call"
    )
    ce.add_argument("--device", default="cuda")
    ce.add_argument("--report", type=Path, default=None)
    return parser


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mapping":
            report = mapping()
            print(summarize_mapping(report))
            _write(args.report, report)
            return 0
        if args.command == "calibrate":
            report = run_calibration(args.op, device=args.device)
            print(summarize_calibration(report))
            _write(args.report, report)
            return 0
        if args.command == "loss":
            report = run_loss_check(device=args.device)
            print(summarize_loss(report))
            _write(args.report, report)
            return 0
        if args.command == "cross-entropy":
            report = run_cross_entropy_check(device=args.device)
            print(summarize_cross_entropy(report))
            _write(args.report, report)
            return 0 if report["status"] == "pass" else 1
        report = run_verify(
            args.source, device=args.device, noise_repeats=args.noise_repeats
        )
    except (Level1Error, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1




if __name__ == "__main__":
    raise SystemExit(main())
