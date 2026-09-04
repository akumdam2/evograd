"""Operator-tier benchmark: one nn.Module per provider, one process per shape.

    python -m evograd.bench.tier2_cli --op layernorm \
        --candidate evolved_layernorm.py --out ~/tmp/tier2.json

Providers default to eager PyTorch, `torch.compile`, the declaration's `liger`
pair, and the candidate when one is given. Each shape runs in its own process:
`torch.compile` caches compiled artifacts per process and a candidate can wedge
a CUDA context, and neither should be able to reach the next shape's numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ISOLATION_TIMEOUT = int(os.environ.get("EVOGRAD_TIER2_TIMEOUT", "1800"))


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_tier2_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--baseline", default="liger", help="declared pair baseline")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--suite", default=None, help="named benchmark suite")
    parser.add_argument("--dtype", action="append", dest="dtypes")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rep-ms", type=int, default=None)
    parser.add_argument("--warmup-ms", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-check", action="store_true", help="skip the correctness gate")
    parser.add_argument(
        "--identity-control",
        action="store_true",
        help=(
            "measure the eager module against itself under two names. Must "
            "report ~1.0x; whatever it does report is this tier's noise floor"
        ),
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="run every provider in this process (faster; a hang takes the run "
             "down, and one provider's allocator and Dynamo state reach the next)",
    )
    # Set by the parent when it re-invokes itself for one shape.
    parser.add_argument("--case-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--provider-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser


def _specs(args, candidate_module):
    from evograd.bench.tier2 import default_provider_specs, identity_control_specs

    if args.identity_control:
        return identity_control_specs()
    return default_provider_specs(
        candidate_module=candidate_module,
        baseline=args.baseline or None,
        compile_baseline=not args.no_compile,
    )


def _run_one_case(args, op, workload, only: str | None = None):
    from evograd.bench.tier2 import DEFAULT_REP_MS, DEFAULT_WARMUP_MS, run_case

    candidate_module = load_candidate(args.candidate) if args.candidate else None
    return run_case(
        op,
        workload,
        _specs(args, candidate_module),
        device=args.device,
        rep_ms=args.rep_ms or DEFAULT_REP_MS,
        warmup_ms=args.warmup_ms or DEFAULT_WARMUP_MS,
        check=not args.no_check,
        only=only,
    )


def _run_isolated(argv: list[str], index: int, provider: str | None = None) -> dict:
    """Re-invoke this module for one shape (and one provider) and read its JSON.

    One provider per process, not one shape per process. Providers are not
    inert neighbours: `torch.compile` leaves Dynamo caches and compiled artifacts
    behind, Triton autotuning caches per process, and the caching allocator
    keeps whatever the previous provider's peak reserved. Sharing a process
    means the first provider measured is not measured under the same conditions
    as the last.
    """
    import tempfile

    handle, result_path = tempfile.mkstemp(prefix="evograd_tier2_", suffix=".json")
    os.close(handle)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "evograd.bench.tier2_cli", *argv,
             "--case-index", str(index), "--result-json", result_path]
            + (["--provider-name", provider] if provider else []),
            capture_output=True, text=True, timeout=ISOLATION_TIMEOUT,
        )
        try:
            with open(result_path, encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {
                "ok": False,
                "error": (
                    f"tier2 worker exited rc={process.returncode} without a result"
                ),
                "stderr_tail": process.stderr[-4000:],
            }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"shape exceeded {ISOLATION_TIMEOUT}s and was killed"}
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)

    from evograd.bench.tier2 import TIER2_PROTOCOL_VERSION
    from evograd.ops import get_op

    op = get_op(args.op)
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtypes = (
        tuple(aliases.get(d.lower(), d.lower()) for d in args.dtypes)
        if args.dtypes
        else None
    )
    workloads = op.benchmark_workloads(suite=args.suite, dtypes=dtypes)

    # Worker mode: one shape, print/write JSON, exit.
    if args.case_index is not None:
        case = _run_one_case(args, op, workloads[args.case_index],
                             only=args.provider_name)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(case), encoding="utf-8")
        return 0

    from evograd.bench.tier1 import environment_fingerprint
    from evograd.bench.tier2 import (
        DEFAULT_REP_MS,
        DEFAULT_WARMUP_MS,
        REPETITIONS,
        WARMUP_ITERS,
        _require_declared_split,
    )

    _require_declared_split(op)

    parent_argv = [a for a in argv]
    cases = []
    for index, workload in enumerate(workloads):
        print(
            f"[tier2] {index + 1}/{len(workloads)} {workload.dims} {workload.dtype}",
            file=sys.stderr, flush=True,
        )
        if args.no_isolate:
            cases.append(_run_one_case(args, op, workload))
        else:
            # One child per provider; the parent stitches the case back together
            # so the report shape is unchanged.
            # The candidate module must be loaded to enumerate the spec list:
            # without it `default_provider_specs` omits the candidate entirely
            # and the parent never spawns a child for it.
            names = [
                spec.name for spec in _specs(
                    args, load_candidate(args.candidate) if args.candidate else None
                )
            ]
            merged: dict | None = None
            for name in names:
                part = _run_isolated(parent_argv, index, name)
                if merged is None:
                    merged = {k: v for k, v in part.items() if k != "providers"}
                    merged["providers"] = {}
                merged["providers"].update(part.get("providers", {}))
            cases.append(merged or {"ok": False, "error": "no providers"})

    report = {
        "protocol": TIER2_PROTOCOL_VERSION,
        "op": op.name,
        "timing_protocol": {
            "driver": "triton.testing.do_bench",
            "rep_ms": args.rep_ms or DEFAULT_REP_MS,
            "warmup_ms": args.warmup_ms or DEFAULT_WARMUP_MS,
            "quantiles": [0.5, 0.2, 0.8],
            "grad_to_none": "activations only, matching Liger; parameter .grad accumulates",
            "step": "y = model(*activations); torch.autograd.backward(y, output_grads)",
            "isolation": ("one process per provider per shape"
                          if not args.no_isolate else "single process"),
            "repetitions": REPETITIONS,
            "warmup_iterations": WARMUP_ITERS,
            "driver": "cuda events, fixed repetition count, L2 flushed between samples",
        },
        "environment": environment_fingerprint(),
        "cases": cases,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
