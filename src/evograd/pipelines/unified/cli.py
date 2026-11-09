"""evograd seed unified: deterministic compiler-aware AtenIR synthesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.shared.cli_common import add_pack_arg, resolve_op
from evograd.pipelines.unified.synthesize import UnifiedSeedConfig, synthesize_unified_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pack_arg(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.example_input:
        parser.error("unified synthesizes at the pack's own configuration")
    op, _example_input, pack = resolve_op(args)
    (workload,) = op.benchmark
    return synthesize_unified_seed(UnifiedSeedConfig(
        op=op, workload=workload, output_dir=args.output_dir, pack=pack,
        python=args.python, device=args.device, eval_timeout=args.eval_timeout,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
