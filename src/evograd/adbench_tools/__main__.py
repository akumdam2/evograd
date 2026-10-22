"""python -m evograd.adbench_tools <command>

    task       build a task pack for one operator configuration
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


def _dims(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    dims: dict[str, int] = {}
    for binding in values:
        name, separator, value = binding.partition("=")
        if not separator or name in dims:
            raise SystemExit(f"invalid or duplicate dimension: {binding}")
        dims[name] = int(value)
    return dims


def _task(argv: list[str]) -> int:
    from evograd.adbench_tools.task import CASE_SELECTIONS, LIBRARY_POLICIES, build_task

    parser = argparse.ArgumentParser(prog="python -m evograd.adbench_tools task",
                                     description="Build a task pack (see evograd.adbench_tools.task).")
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None, help="external path.py:op declaration")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--dims", nargs="+", metavar="NAME=SIZE",
                        help="default: the first declared workload of the dtype, as "
                             "`evograd seed unified` selects it")
    parser.add_argument("--case", choices=CASE_SELECTIONS, default="benchmark",
                        help="without --dims: the first timed benchmark case (default) or, "
                             "as `evograd seed unified` defaults, the first correctness case")
    parser.add_argument("--library", choices=LIBRARY_POLICIES, default="auto")
    parser.add_argument("--scoring", default="speed_memory")
    parser.add_argument("--performance-baseline", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    task = build_task(args.op, args.output_dir, declaration=args.declaration, dtype=args.dtype,
                      dims=_dims(args.dims), library=args.library, scoring=args.scoring,
                      performance_baseline=args.performance_baseline, case=args.case)
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
