#!/usr/bin/env python3
"""evograd's frontend alone: the unified seed, with no evolution and no LLM.

    python candidates/seed/run.py --task TASK --run-dir RUN

The program evograd's search starts from, submitted as it is. It is the
no-LLM check of the benchmark pipeline (task pack -> run directory ->
``evograd evaluate``) and the "frontend only" reference point beside the
agents. ``--budget``, ``--model`` and ``--api-base`` are accepted and ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "seed"


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--eval-timeout", type=int, default=180,
                   help="seed acceptance timeout, as `evograd seed unified`")
    args = p.parse_args(argv)
    task, ledger = _common.start(args, AGENT, {"frontend": "unified", "backend": None,
                                               "llm_calls": 0})
    seed = _common.unified_seed(task, ledger, eval_timeout=args.eval_timeout)
    if seed is None:
        return _common.finish(args, task, ledger, None, status="seed_failed",
                              failure=str(ledger.root / "seed" / "failure.json"))
    return _common.finish(args, task, ledger, seed,
                          contract=ledger.root / "seed" / "unified_contract.json")


if __name__ == "__main__":
    raise SystemExit(main())
