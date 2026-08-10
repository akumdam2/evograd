#!/usr/bin/env python3
"""Run GEAK ShapeFixer on a disposable harness copy; never mutate the oracle."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from evograd.ops import get_op


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="geak_shape_fixer_audit_") as raw:
        repo = Path(raw)
        harness = repo / "geak_evaluator.py"
        candidate = repo / "candidate.py"
        source = repo / "authoritative_shapes.py"
        shutil.copy2(args.harness, harness)
        shutil.copy2(args.candidate, candidate)
        workloads = get_op("layernorm").benchmark_workloads(suite="industrial_mixed")
        shape_rows = [
            {"dims": dict(workload.dims), "dtype": workload.dtype}
            for workload in workloads
        ]
        source.write_text(
            "# Authoritative Evograd LayerNorm benchmark shape manifest.\n"
            f"SHAPES = {shape_rows!r}\n",
            encoding="utf-8",
        )
        before = harness.read_text(encoding="utf-8")
        before_hash = _hash(harness)

        from minisweagent.models import get_model
        from minisweagent.run.preprocess.shape_fixer_agent import run_shape_fixer

        model = get_model(
            args.model,
            {
                "model_class": "litellm",
                "model_name": args.model,
                "api_key": "",
                "reasoning": None,
                "model_kwargs": {
                    "temperature": 1.0,
                    "max_tokens": 8000,
                    "drop_params": True,
                },
            },
        )
        os.environ.setdefault("GEAK_SHAPE_FIXER_TIMEOUT", "300")
        ok = run_shape_fixer(
            model=model,
            repo=repo,
            harness_path=harness,
            benchmark_file=source,
            kernel_path=candidate,
            log_dir=args.output_dir,
            validation_feedback=[
                "Static audit only. The production harness delegates all shape generation "
                "to Evograd's immutable OpDecl; do not run GPU benchmarks and do not rewrite "
                "the oracle. Verify that no local fabricated shape list conflicts with the "
                "authoritative manifest."
            ],
            user_task=(
                "The immutable production contract is Evograd LayerNorm industrial_mixed. "
                "This audit copy must remain byte-identical; report SHAPES_VERIFIED when "
                "delegation preserves the source-of-truth semantics."
            ),
        )
        after = harness.read_text(encoding="utf-8")
        after_hash = _hash(harness)
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="before/geak_evaluator.py",
                tofile="after/geak_evaluator.py",
            )
        )
        (args.output_dir / "audited_harness.py").write_text(after, encoding="utf-8")
        (args.output_dir / "shape_fixer.diff").write_text(diff, encoding="utf-8")
        result = {
            "shape_fixer_returned_success": bool(ok),
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "byte_identical": before_hash == after_hash,
            "accepted_for_experiment": bool(ok and before_hash == after_hash),
            "authoritative_shapes": shape_rows,
            "note": "Standalone audit only; formal GEAK run remains pre-validated Path-A.",
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["accepted_for_experiment"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
