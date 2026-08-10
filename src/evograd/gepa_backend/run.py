"""Run GEPA direct-code evolution over LayerNorm shape instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from evograd.gepa_backend.candidate import EvolveBlockTemplate
from evograd.gepa_backend.evaluator import ShapeBatchEvaluator, examples_for_suite

CODE_REFLECTION_PROMPT = r"""
You are optimizing only the EVOLVE-BLOCK body of an Evograd Triton LayerNorm
forward/backward pair.

Current complete EVOLVE-BLOCK body:
```python
<curr_param>
```

Per-shape compilation, correctness, latency, saved-memory, and speedup feedback:
```
<side_info>
```

Return one complete drop-in replacement for the EVOLVE-BLOCK body.

Hard requirements:
- Define layernorm_forward_with_saved and layernorm_backward_from_saved.
- Preserve the public signatures and forward/backward saved-state consistency.
- Support every correctness, coverage, and benchmark shape and dtype.
- A score of zero means compilation, API, timeout, or numerical failure.
- Keep generated math Triton-only; torch may allocate tensors and inspect shapes,
  but do not call autograd, PyTorch LayerNorm, Liger, torch.compile, or another
  high-level implementation.
- Do not modify or reproduce EVOLVE-BLOCK marker lines.
- Do not emit a patch, explanation, analysis, or multiple alternatives.
- Return exactly one complete Python EVOLVE-BLOCK body inside one fenced block.

Optimization goal:
- Preserve strategies that are especially strong on individual shapes.
- Improve min(backward speedup, raw full-step speedup), with the reported
  saved-memory penalty.
- Use shape-aware launch configuration or internal dispatch when useful.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposals", type=int, choices=(10, 30), required=True)
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=850)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    try:
        from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything
    except ImportError as exc:
        raise RuntimeError(
            "GEPA is not installed; install pinned commit "
            "8a2bed96385202f69caaeb5327a843ed2f5ea225"
        ) from exc

    if args.resume:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"resume output directory not found: {args.output_dir}")
        for name in ("run_summary", "gepa_result"):
            source = args.output_dir / f"{name}.json"
            backup = args.output_dir / f"{name}_before_{args.proposals}.json"
            if source.is_file() and not backup.exists():
                shutil.copy2(source, backup)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.output_dir / "evaluation_cache"
    evaluator = ShapeBatchEvaluator(
        seed_path=args.seed,
        cache_dir=cache_dir,
        warmup=args.warmup,
        reps=args.reps,
        timeout=args.timeout,
    )
    trainset = examples_for_suite("layernorm", "industrial_mixed")
    valset = examples_for_suite("layernorm", "tb_mixed")
    max_evals = 400 if args.proposals == 10 else 1200
    gepa_output = args.output_dir / "gepa_output"
    gepa_state = args.output_dir / "gepa_state"

    config = OptimizeAnythingConfig(
        engine="gepa",
        name=f"evograd-layernorm-{args.proposals}",
        max_evals=max_evals,
        max_concurrency=1,
        output_dir=gepa_output,
        run_dir=str(gepa_state),
        sandbox=False,
        engine_config={
            "engine": {
                "seed": args.random_seed,
                "parallel": False,
                "max_workers": 1,
                "max_candidate_proposals": args.proposals,
                "candidate_selection_strategy": "pareto",
                "frontier_type": "instance",
                "acceptance_criterion": "strict_improvement",
                "cache_evaluation": True,
                "write_agent_state": True,
                "display_progress_bar": True,
            },
            "reflection": {
                "reflection_lm": args.model,
                "reflection_lm_kwargs": {
                    "temperature": 1.0,
                    "max_tokens": 20000,
                },
                "reflection_minibatch_size": 3,
                "module_selector": "all",
                "reflection_prompt_template": CODE_REFLECTION_PROMPT,
            },
            "merge": None,
            "refiner": None,
        },
    )

    started = time.time()
    result = optimize_anything(
        seed_candidate=evaluator.seed_body,
        evaluator=evaluator.evaluate,
        batch_evaluator=evaluator.evaluate_batch,
        dataset=trainset,
        valset=valset,
        objective=None,
        background=None,
        config=config,
    )
    elapsed = time.time() - started
    archive = result.metadata.get("gepa_result")
    if archive is None:
        raise RuntimeError("GEPA completed without a persisted GEPAResult archive")

    archive_dir = args.output_dir / "candidates"
    archive_dir.mkdir(exist_ok=True)
    for index, candidate in enumerate(archive.candidates):
        body = candidate.get("current_candidate", next(iter(candidate.values())))
        (archive_dir / f"candidate_{index:04d}_block.py").write_text(
            body.strip() + "\n", encoding="utf-8"
        )
        (archive_dir / f"candidate_{index:04d}.py").write_text(
            evaluator.template.render(body), encoding="utf-8"
        )

    archive_payload = archive.to_dict()
    (args.output_dir / "gepa_result.json").write_text(
        json.dumps(archive_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / f"gepa_result_{args.proposals}.json").write_text(
        json.dumps(archive_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    best_body = archive.best_candidate
    if not isinstance(best_body, str):
        best_body = best_body.get("current_candidate", next(iter(best_body.values())))
    best_program = args.output_dir / "best_generalist.py"
    best_program.write_text(evaluator.template.render(best_body), encoding="utf-8")

    summary = {
        "gepa_commit": "8a2bed96385202f69caaeb5327a843ed2f5ea225",
        "proposals_budget": args.proposals,
        "max_evals": max_evals,
        "elapsed_s": elapsed,
        "num_candidates_accepted": archive.num_candidates,
        "total_metric_calls": archive.total_metric_calls,
        "best_idx": archive.best_idx,
        "best_score": archive.val_aggregate_scores[archive.best_idx],
        "best_program": str(best_program),
        "trainset": trainset,
        "valset": valset,
        "per_val_instance_best_candidates": {
            str(key): sorted(value)
            for key, value in archive.per_val_instance_best_candidates.items()
        },
        "adapter_cost": result.metadata.get("adapter_cost"),
        "total_cost": result.metadata.get("total_cost"),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / f"run_summary_{args.proposals}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
