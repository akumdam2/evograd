"""python -m evograd.adbench_tools <command>

    task       build the task pack of one model's observed case
    statement  print a task pack's shared problem statement
    package    wrap an agent module into an evaluable artifact
    evaluate   evaluate one candidate (see evograd.adbench_tools.evaluate)
    score      final measurement of an agent's program (see .score)
    compare    Markdown table over final.json files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _task(argv: list[str]) -> int:
    from evograd.adbench_tools.task import LIBRARY_POLICIES, build_task

    parser = argparse.ArgumentParser(prog="python -m evograd.adbench_tools task",
                                     description="Build a task pack (see evograd.adbench_tools.task).")
    parser.add_argument("--model", required=True,
                        help="a verified model (evograd.verification.integrations)")
    parser.add_argument("--case-id", required=True, metavar="TASK/COMPONENT",
                        help="the model's observed case, under the model's reference (what "
                             "`evograd evaluate` judges)")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="the model's payload corpus (its manifest.json): search feedback is "
                             "Gate 1 on the case's calibration payloads")
    parser.add_argument("--state", type=Path, default=None,
                        help="reuse or keep the operator Gate 2 calibration here")
    parser.add_argument("--library", choices=LIBRARY_POLICIES, default="auto")
    parser.add_argument("--scoring", default="speed_memory")
    parser.add_argument("--performance-baseline", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    task = build_task(args.output_dir, model=args.model, case_id=args.case_id, corpus=args.corpus,
                      state=args.state, library=args.library, scoring=args.scoring,
                      performance_baseline=args.performance_baseline)
    print(f"task {task.task_id}: {task.root}")
    print(f"  configuration: {task.workload}")
    print(f"  library primitives: {list(task.allowed_primitives)} "
          f"({task.spec['library_policy']['decision']})")
    print(f"  statement: {task.root / 'TASK.md'}")
    return 0


def _statement(argv: list[str]) -> int:
    from evograd.adbench_tools.task import load

    parser = argparse.ArgumentParser(prog="python -m evograd.adbench_tools statement")
    parser.add_argument("task", type=Path)
    print(load(parser.parse_args(argv).task).statement)
    return 0


def _package(argv: list[str]) -> int:
    from evograd.adbench_tools.artifact import PackagingError, extract_code, package
    from evograd.adbench_tools.task import load

    parser = argparse.ArgumentParser(prog="python -m evograd.adbench_tools package")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--origin", default="cli")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("module", type=Path, help="agent module or LLM response text")
    args = parser.parse_args(argv)
    task = load(args.task)
    try:
        artifact = package(task, extract_code(args.module.read_text(encoding="utf-8"), task),
                           origin=args.origin)
    except PackagingError as exc:
        print(f"PackagingError: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact, encoding="utf-8")
    print(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"task": _task, "statement": _statement, "package": _package}
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command in commands:
        return commands[command](rest)
    if command == "evaluate":
        from evograd.adbench_tools.evaluate import main as evaluate_main

        return evaluate_main(rest)
    if command in ("score", "compare"):
        from evograd.adbench_tools.score import main as score_main

        return score_main([command, *rest])
    print(f"unknown command {command!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
