#!/usr/bin/env python3
"""Compare shape-aware MAP-Elites harvest+dispatch vs explicit small/large evolve.

Budget (matched LLM iterations):
  - explicit arm: 3 groups x N iterations
  - MAP arm: 1 run x (3N) iterations with regime feature axes

Both arms emit a dispatched dual-specialist program, then fair-bench on
industrial_mixed (and optionally coverage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


def _load_module(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"evograd_map_cmp_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fair_bench(op_name: str, candidate: Path, suite: str, out: Path) -> dict:
    from evograd.bench.tier1 import (
        candidate_provider,
        liger_provider,
        run_fair_benchmarks,
    )
    from evograd.opdecl.baselines import verify_performance_baseline
    from evograd.opdecl.verify import verify
    from evograd.ops import get_op

    op = get_op(op_name)
    verify_performance_baseline(op, "liger")
    module = _load_module(candidate)
    correctness = verify(op, module)
    if not correctness.ok:
        raise RuntimeError(
            f"fair-bench correctness failed for {candidate}:\n"
            + json.dumps(correctness.to_dict(), indent=2)
        )
    if suite == "coverage":
        workloads = op.coverage
        if not workloads:
            raise ValueError(f"{op_name}: no coverage workloads declared")
    else:
        workloads = op.benchmark_workloads(suite=suite)
    report = run_fair_benchmarks(
        op,
        candidate_provider(op, module),
        liger_provider(op),
        workloads=workloads,
        warmup=10,
        reps=50,
        blocks=3,
        seed=0,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _summary_speedups(fair_report: dict) -> dict:
    aggregate = fair_report.get("aggregate") or {}
    keys = (
        "speedup_pair_full",
        "speedup_pair_full_ci95",
        "speedup_backward",
        "speedup_backward_ci95",
        "speedup_forward",
        "speedup_liger_compatible_raw_fused_full",
    )
    return {key: aggregate[key] for key in keys if key in aggregate}


def _paired_fair_bench(
    op_name: str,
    map_candidate: Path,
    explicit_candidate: Path,
    suite: str,
    out: Path,
) -> dict:
    """Direct paired timing; speedup > 1 means MAP is faster than Explicit."""
    from evograd.bench.tier1 import (
        candidate_provider,
        renamed_provider,
        run_fair_benchmarks,
    )
    from evograd.opdecl.verify import verify
    from evograd.ops import get_op

    op = get_op(op_name)
    map_module = _load_module(map_candidate)
    explicit_module = _load_module(explicit_candidate)
    for label, module in (("map", map_module), ("explicit", explicit_module)):
        correctness = verify(op, module)
        if not correctness.ok:
            raise RuntimeError(
                f"paired fair correctness failed for {label}:\n"
                + json.dumps(correctness.to_dict(), indent=2)
            )
    workloads = op.coverage if suite == "coverage" else op.benchmark_workloads(suite)
    report = run_fair_benchmarks(
        op,
        renamed_provider(candidate_provider(op, map_module), "map"),
        renamed_provider(candidate_provider(op, explicit_module), "explicit"),
        workloads=workloads,
        warmup=10,
        reps=50,
        blocks=3,
        seed=1729,
    )
    report["comparison_direction"] = (
        "speedup_pair_full = explicit latency / MAP latency; >1 favors MAP"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _write_stage(root: Path, name: str, payload: dict) -> None:
    stage_dir = root / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _resume_state(
    evolve_dir: Path, target: Path, requested_iterations: int
) -> tuple[Path | None, int]:
    """Return latest checkpoint and remaining evolution iterations."""
    if target.is_file():
        return None, 0
    checkpoints = []
    for path in (evolve_dir / "checkpoints").glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None, requested_iterations
    completed, checkpoint = max(checkpoints, key=lambda item: item[0])
    remaining = max(0, requested_iterations - completed)
    if remaining == 0:
        checkpoint_best = checkpoint / "best_program.py"
        if not checkpoint_best.is_file():
            raise RuntimeError(
                f"checkpoint reached budget but has no best program: {checkpoint}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_best, target)
        return None, 0
    return checkpoint, remaining


def run_explicit_arm(
    *,
    op,
    seed: Path,
    root: Path,
    iterations: int,
    model: str,
    api_base: str,
    baseline: str,
    force: bool,
) -> dict:
    from evograd.dispatch import dispatch
    from evograd.evolve.run import run_evolve

    evolve_root = root / "explicit" / "evolve"
    programs: dict[str, Path] = {}
    for group in ("full", "small", "large"):
        group_dir = evolve_root / group
        target = group_dir / "evolved_best_program.py"
        if force or not target.is_file():
            checkpoint, remaining = (
                (None, iterations)
                if force
                else _resume_state(group_dir, target, iterations)
            )
            if remaining == 0:
                programs[group] = target
                continue
            scoring = (
                "speed_memory_min"
                if group == "full"
                else "speed_memory_min_weighted_geomean"
            )
            rc = run_evolve(
                op,
                seed_path=seed,
                output_dir=group_dir,
                scoring=scoring,
                iterations=remaining,
                checkpoint_path=checkpoint,
                save_best_to=target,
                primary_model=model,
                secondary_model=model,
                api_base=api_base,
                benchmark_suite=group,
                performance_baseline=baseline,
            )
            if rc != 0:
                raise RuntimeError(f"explicit evolve failed for group={group}")
        programs[group] = target

    deploy = root / "explicit" / "deploy"
    report_path = deploy / f"{op.name}_dispatch_report.json"
    if force or not report_path.is_file():
        report = dispatch(op, programs, output_dir=deploy, baseline=baseline)
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    final_program = Path(report["final_program"])
    result = {
        "arm": "explicit",
        "programs": {tag: str(path) for tag, path in programs.items()},
        "dispatch_report": report,
        "final_program": str(final_program),
        "iterations_per_group": iterations,
        "total_iterations": 3 * iterations,
    }
    _write_stage(root, "explicit_complete", result)
    return result


def run_map_arm(
    *,
    op,
    seed: Path,
    root: Path,
    iterations: int,
    model: str,
    api_base: str,
    baseline: str,
    force: bool,
) -> dict:
    from evograd.dispatch import dispatch
    from evograd.evolve.map_harvest import harvest_regime_elites
    from evograd.evolve.run import render_map_shape_config, run_evolve

    evolve_dir = root / "map" / "evolve"
    best = evolve_dir / "evolved_best_program.py"
    config_path = evolve_dir / "openevolve_config.yaml"
    if force or not best.is_file():
        evolve_dir.mkdir(parents=True, exist_ok=True)
        checkpoint, remaining = (
            (None, iterations)
            if force
            else _resume_state(evolve_dir, best, iterations)
        )
        if remaining == 0:
            checkpoint = None
        config_path.write_text(
            render_map_shape_config(
                op,
                iterations=remaining or iterations,
                primary_model=model,
                secondary_model=model,
                api_base=api_base,
            ),
            encoding="utf-8",
        )
        if remaining:
            rc = run_evolve(
                op,
                seed_path=seed,
                output_dir=evolve_dir,
                scoring="speed_memory_min",
                iterations=remaining,
                config_path=config_path,
                checkpoint_path=checkpoint,
                save_best_to=best,
                primary_model=model,
                secondary_model=model,
                api_base=api_base,
                benchmark_suite="full",
                performance_baseline=baseline,
            )
            if rc != 0:
                raise RuntimeError("MAP evolve failed")

    harvest = harvest_regime_elites(
        evolve_dir,
        archive_only=True,
        output_dir=root / "map" / "harvested",
    )
    programs = {tag: Path(path) for tag, path in harvest["programs"].items()}
    if "full" not in programs:
        raise RuntimeError("MAP harvest missing full elite")
    distinct_elites = bool(harvest["distinct_regime_elites"])
    if not distinct_elites:
        # One program winning both descriptors is a valid MAP outcome, but it
        # is a generalist—not a dispatched pair of specialists.
        programs = {"full": programs["full"]}

    deploy = root / "map" / "deploy"
    report_path = deploy / f"{op.name}_dispatch_report.json"
    if force or not report_path.is_file():
        report = dispatch(op, programs, output_dir=deploy, baseline=baseline)
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    result = {
        "arm": "map",
        "harvest": harvest,
        "distinct_regime_elites": distinct_elites,
        "programs": {tag: str(path) for tag, path in programs.items()},
        "dispatch_report": report,
        "final_program": report["final_program"],
        "iterations": iterations,
        "total_iterations": iterations,
    }
    _write_stage(root, "map_complete", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", default="layernorm")
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--explicit-iterations",
        type=int,
        default=10,
        help="iterations per explicit group (full/small/large)",
    )
    parser.add_argument(
        "--map-iterations",
        type=int,
        default=None,
        help="MAP iterations (default: 3 * explicit-iterations)",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--baseline", default="liger")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-fair",
        action="store_true",
        help="stop after dispatch; do not run fair protocol",
    )
    parser.add_argument(
        "--arm",
        choices=("both", "explicit", "map"),
        default="both",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.seed.is_file():
        print(f"ERROR: seed not found: {args.seed}", file=sys.stderr)
        return 2

    from evograd.ops import get_op

    op = get_op(args.op)
    if op.regime_feature is None:
        print(f"ERROR: op {args.op} lacks regime_feature", file=sys.stderr)
        return 2

    map_iterations = (
        args.map_iterations
        if args.map_iterations is not None
        else 3 * args.explicit_iterations
    )
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    comparison: dict = {
        "op": args.op,
        "seed": str(args.seed),
        "seed_sha256": _sha256(args.seed),
        "explicit_iterations_per_group": args.explicit_iterations,
        "map_iterations": map_iterations,
        "model": args.model,
        "baseline": args.baseline,
        "budget_note": (
            "matched LLM iterations: explicit=3*N, map=3N "
            f"(N={args.explicit_iterations})"
        ),
    }
    _write_stage(root, "experiment_manifest", comparison)

    if args.arm in ("both", "explicit"):
        print("+ running explicit small/large arm", flush=True)
        comparison["explicit"] = run_explicit_arm(
            op=op,
            seed=args.seed,
            root=root,
            iterations=args.explicit_iterations,
            model=args.model,
            api_base=args.api_base,
            baseline=args.baseline,
            force=args.force,
        )
    if args.arm in ("both", "map"):
        print("+ running MAP shape-feature arm", flush=True)
        comparison["map"] = run_map_arm(
            op=op,
            seed=args.seed,
            root=root,
            iterations=map_iterations,
            model=args.model,
            api_base=args.api_base,
            baseline=args.baseline,
            force=args.force,
        )

    if not args.skip_fair:
        fair_dir = root / "fair"
        fair_dir.mkdir(parents=True, exist_ok=True)
        for arm_name in ("explicit", "map"):
            arm = comparison.get(arm_name)
            if not arm:
                continue
            candidate = Path(arm["final_program"])
            for suite in ("industrial_mixed", "coverage"):
                out = fair_dir / f"{arm_name}_{suite}.json"
                print(f"+ fair-bench {arm_name} suite={suite}", flush=True)
                report = _fair_bench(args.op, candidate, suite, out)
                arm.setdefault("fair", {})[suite] = {
                    "path": str(out),
                    "summary": _summary_speedups(report),
                    "protocol": report.get("protocol"),
                }
            _write_stage(root, f"{arm_name}_fair_complete", arm["fair"])

        if comparison.get("explicit") and comparison.get("map"):
            paired = {}
            for suite in ("industrial_mixed", "coverage"):
                out = fair_dir / f"map_vs_explicit_{suite}.json"
                print(f"+ paired fair MAP vs Explicit suite={suite}", flush=True)
                report = _paired_fair_bench(
                    args.op,
                    Path(comparison["map"]["final_program"]),
                    Path(comparison["explicit"]["final_program"]),
                    suite,
                    out,
                )
                paired[suite] = {
                    "path": str(out),
                    "summary": _summary_speedups(report),
                    "protocol": report.get("protocol"),
                    "direction": report.get("comparison_direction"),
                }
            comparison["paired_map_vs_explicit"] = paired
            _write_stage(root, "paired_fair_complete", paired)

    comparison["verdict"] = _verdict(comparison)

    comparison["elapsed_sec"] = time.time() - started
    out_path = root / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    md_path = root / "COMPARISON.md"
    md_path.write_text(_render_markdown(comparison), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


def _verdict(comparison: dict) -> dict:
    if not bool((comparison.get("map") or {}).get("distinct_regime_elites")):
        return {
            "outcome": "inconclusive",
            "reason": "MAP did not produce distinct correct small/large elites",
        }
    primary = (
        (comparison.get("paired_map_vs_explicit") or {})
        .get("industrial_mixed", {})
        .get("summary", {})
    )
    ci = primary.get("speedup_pair_full_ci95") or {}
    low, high = ci.get("low"), ci.get("high")
    if low is None or high is None:
        return {"outcome": "inconclusive", "reason": "missing paired 95% CI"}
    if float(low) > 1.0:
        return {
            "outcome": "map_wins",
            "reason": "paired industrial_mixed pair_full CI is entirely above 1",
        }
    if float(high) < 1.0:
        return {
            "outcome": "explicit_wins",
            "reason": "paired industrial_mixed pair_full CI is entirely below 1",
        }
    return {
        "outcome": "inconclusive",
        "reason": "paired industrial_mixed pair_full CI overlaps 1",
    }


def _render_markdown(comparison: dict) -> str:
    lines = [
        "# LayerNorm shape-aware MAP vs explicit specialists",
        "",
        f"- op: `{comparison.get('op')}`",
        f"- budget: {comparison.get('budget_note')}",
        f"- model: `{comparison.get('model')}`",
        f"- baseline: `{comparison.get('baseline')}`",
        f"- verdict: `{(comparison.get('verdict') or {}).get('outcome')}`",
        f"- verdict reason: {(comparison.get('verdict') or {}).get('reason')}",
        f"- elapsed_sec: {comparison.get('elapsed_sec')}",
        "",
    ]
    for arm_name in ("explicit", "map"):
        arm = comparison.get(arm_name)
        if not arm:
            continue
        lines.append(f"## {arm_name}")
        lines.append("")
        lines.append(f"- final_program: `{arm.get('final_program')}`")
        dispatch = arm.get("dispatch_report") or {}
        lines.append(f"- regime_collapsed: `{dispatch.get('regime_collapsed')}`")
        lines.append(f"- best_threshold: `{dispatch.get('best_threshold')}`")
        if arm_name == "map":
            harvest = arm.get("harvest") or {}
            lines.append(
                f"- distinct_regime_elites: `{harvest.get('distinct_regime_elites')}`"
            )
            lines.append(f"- correct_archive_programs: `{harvest.get('correct_pool_size')}`")
            lines.append(
                "- occupied_correct_descriptor_points: "
                f"`{harvest.get('occupied_correct_descriptor_points')}`"
            )
        lines.append(
            f"- threshold_dispatch_geomean: `{dispatch.get('threshold_dispatch_geomean')}`"
        )
        fair = arm.get("fair") or {}
        for suite, payload in fair.items():
            lines.append(f"- fair[{suite}]: `{payload.get('summary')}`")
        lines.append("")
    paired = comparison.get("paired_map_vs_explicit") or {}
    if paired:
        lines.extend(["## Direct paired MAP vs Explicit", ""])
        for suite, payload in paired.items():
            lines.append(f"- paired[{suite}]: `{payload.get('summary')}`")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
