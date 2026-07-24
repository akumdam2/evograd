"""Profile one correct candidate, generate a fix, and retain it only if faster."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evograd.dispatch import _evaluate_program
from evograd.ncu.optimizer import optimize_from_profile
from evograd.ncu.profile import run_ncu_profile
from evograd.opdecl.activity import OpDecl


def _write_record(output_dir: Path, record: dict) -> None:
    (output_dir / "pass.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# NCU-guided refinement",
        "",
        f"Outcome: **{record.get('outcome', 'unknown')}**",
        "",
        "## NCU metrics",
        "",
        "```json",
        json.dumps(record.get("profile", {}).get("metrics", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    if record.get("diagnosis"):
        lines.extend(("## Diagnosis", "", str(record["diagnosis"]), ""))
    (output_dir / "pass.md").write_text("\n".join(lines), encoding="utf-8")


def refine_candidate(
    op: OpDecl,
    candidate: Path,
    *,
    output_dir: Path,
    baseline: str = "auto",
    model: str = "gpt-5.5",
    api_base: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    eval_timeout: int = 850,
    ncu_timeout: int = 120,
    optimizer_timeout: int = 360,
    skip_at_roofline_pct: float = 95.0,
) -> dict:
    """Run one accepted-only NCU pass and return its durable pass record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidate.resolve()
    cache = output_dir.parent / ".baseline_timing_cache.json"
    before = _evaluate_program(
        op,
        candidate,
        baseline=baseline,
        warmup=10,
        reps=50,
        timeout=eval_timeout,
        cache_path=cache,
    )
    record = {"candidate": str(candidate), "before": before}
    if float(before["metrics"].get("correct", 0.0)) != 1.0:
        record["outcome"] = "skipped: input candidate failed correctness"
        _write_record(output_dir, record)
        return record

    profile = run_ncu_profile(op, candidate, timeout=ncu_timeout)
    record["profile"] = profile.to_dict()
    if not profile.ok:
        record["outcome"] = f"skipped: {profile.error}"
        _write_record(output_dir, record)
        return record
    if profile.report_path:
        report_target = output_dir / "profile.ncu-rep"
        shutil.copy2(profile.report_path, report_target)
        record["profile"]["report_path"] = str(report_target)

    code = candidate.read_text(encoding="utf-8")
    generated, optimizer_record = optimize_from_profile(
        code,
        profile.metrics,
        before["metrics"],
        kernel_metrics=profile.kernels,
        model=model,
        api_base=api_base,
        api_key=api_key,
        timeout=optimizer_timeout,
        skip_at_roofline_pct=skip_at_roofline_pct,
    )
    record.update(optimizer_record)
    if generated is None:
        record.setdefault("outcome", optimizer_record.get("outcome", "skipped"))
        _write_record(output_dir, record)
        return record

    proposed = output_dir / "proposed.py"
    proposed.write_text(generated, encoding="utf-8")
    after = _evaluate_program(
        op,
        proposed,
        baseline=baseline,
        warmup=10,
        reps=50,
        timeout=eval_timeout,
        cache_path=cache,
    )
    record["after"] = after
    before_score = float(before["metrics"].get("combined_score", -1e9))
    after_score = float(after["metrics"].get("combined_score", -1e9))
    improved = (
        float(after["metrics"].get("correct", 0.0)) == 1.0
        and after_score > before_score
    )
    record["accepted"] = improved
    record["outcome"] = (
        f"accepted: score {before_score:.6g} -> {after_score:.6g}"
        if improved
        else f"rejected: score {before_score:.6g} -> {after_score:.6g}"
    )
    if improved:
        original = output_dir / "original.py"
        shutil.copy2(candidate, original)
        shutil.copy2(proposed, candidate)
    _write_record(output_dir, record)
    return record


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="auto")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ncu-timeout", type=int, default=120)
    parser.add_argument("--optimizer-timeout", type=int, default=360)
    parser.add_argument("--skip-at-roofline-pct", type=float, default=95.0)
    args = parser.parse_args(argv)
    from evograd.ops import get_op, load_op

    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    record = refine_candidate(
        op,
        args.candidate,
        output_dir=args.output_dir,
        baseline=args.baseline,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        ncu_timeout=args.ncu_timeout,
        optimizer_timeout=args.optimizer_timeout,
        skip_at_roofline_pct=args.skip_at_roofline_pct,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not str(record["outcome"]).startswith("skipped: input") else 1
