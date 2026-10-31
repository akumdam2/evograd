"""Evaluate all selected agents of a model in one job with a shared CPU snapshot."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
import time


def run(plan, *, deadline_min, tier3_blocks=10, weight_cache=True):
    from evograd.verification.reuse import calibration_reuse
    from evograd.evaluation.case_group import evaluate_group
    from evograd.evaluation.model_cache import ENV, checkpoint_model_cache

    deadline = time.time() + deadline_min * 60
    model = plan["model"]
    previous = os.environ.get(ENV)
    if deadline_min <= 0:
        return 3
    with tempfile.TemporaryDirectory(prefix="evograd-model-", dir=os.environ.get("SLURM_TMPDIR")) as directory:
        if weight_cache:
            os.environ[ENV] = directory
        else:
            os.environ.pop(ENV, None)
        try:
            context = checkpoint_model_cache(model, directory) if weight_cache else nullcontext()
            with context as cache, calibration_reuse(plan.get("calibration_reuse")):
                if cache is not None:
                    from evograd.verification.integrations import load
                    from evograd.verification.integrations.base import configure

                    configure()
                    cache.prime(load(model).load_workload().spec)
                status = evaluate_group(plan, deadline=deadline, tier3_options={"blocks": tier3_blocks})
                if cache is not None:
                    print(f"[model-cache] Parent checkpoint loads: {cache.loads}; fresh model copies: {cache.copies}", flush=True)
                return status
        finally:
            if previous is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = previous
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--deadline-min", type=float, required=True)
    parser.add_argument("--tier3-blocks", type=int, default=10)
    parser.add_argument("--no-weight-cache", action="store_true")
    from evograd.verification.reuse import add_arguments, options
    add_arguments(parser)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    explicit = options(args)
    if any(explicit.values()):
        if plan.get("calibration_reuse") and plan["calibration_reuse"] != explicit:
            parser.error("calibration reuse flags disagree with the frozen plan")
        plan["calibration_reuse"] = explicit
    return run(plan, deadline_min=args.deadline_min,
               tier3_blocks=args.tier3_blocks, weight_cache=not args.no_weight_cache)


if __name__ == "__main__":
    raise SystemExit(main())
