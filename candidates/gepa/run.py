#!/usr/bin/env python3
"""GEPA (``gepa.optimize_anything``) on an adbench-tools task.

    pip install "gepa @ git+https://github.com/gepa-ai/gepa@8f287de942f41a6d1e0e7e5b2d576247c4bedee9" litellm
    python candidates/gepa/run.py --task TASK --run-dir RUN --budget 30

GEPA as its authors ship it, in single-task mode (one candidate text, no
dataset), with its own reflection prompt:

* candidate  -- the whole agent module, as text;
* objective  -- one sentence naming the deliverable and the score;
* background -- the task's problem statement, verbatim (GEPA surfaces it to
  every proposal);
* seed       -- none by default (GEPA's seedless mode: the reflection LM writes
  the first module from objective + background), or ``--seed-from skeleton``;
* evaluator  -- ``evograd.adbench_tools.evaluate.evaluate_source``: the same
  evaluator and the same feedback text evograd's OpenEvolve loop receives.
  ``score`` is ``combined_score`` when correct and 0 otherwise (GEPA needs a
  non-negative scale); ``side_info`` carries the verdict, metrics and
  artifacts.

Budget mapping: ``max_evals = budget + 1`` -- GEPA spends one evaluation on
its starting candidate before its first proposal, as OpenEvolve does on its
seed; cached re-evaluations of an identical candidate are free in both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "gepa"
GEPA_COMMIT = "8f287de942f41a6d1e0e7e5b2d576247c4bedee9"


def objective(task) -> str:
    abi = task.spec["abi"]
    return (f"Return one self-contained Python module defining {abi['forward_fn']} and "
            f"{abi['backward_fn']} for `{task.op_name}` that passes every correctness check "
            "and maximizes combined_score (the evograd evaluator's speed_memory score).")


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--seed-from", choices=("none", "skeleton"), default="none")
    p.add_argument("--temperature", type=float, default=None,
                   help="reflection LM temperature (default: GEPA's)")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="reflection LM max tokens (evograd's OpenEvolve config uses 16000)")
    p.add_argument("--eval-timeout", type=int, default=900)
    args = p.parse_args(argv)
    task, ledger = _common.start(args, AGENT, {
        "gepa_commit": GEPA_COMMIT, "mode": "single-task", "seed_from": args.seed_from,
        "max_evals": args.budget + 1, "reflection_prompt": "gepa default",
    })

    try:
        from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything
    except ImportError as exc:
        raise SystemExit(f"GEPA is not installed ({exc}); see the module docstring") from exc
    from evograd.adbench_tools.artifact import extract_code
    from evograd.adbench_tools.evaluate import evaluate_source

    best = {"score": -1.0, "source": None}

    def evaluator(candidate):
        source = extract_code(candidate if isinstance(candidate, str)
                              else next(iter(candidate.values())), task)
        record = evaluate_source(task, source, run_dir=ledger.root, origin=AGENT,
                                 timeout=args.eval_timeout)
        score = record.agent_score()
        if record.correct and score > best["score"]:
            best.update(score=score, source=source)
        return score, {
            "correct": record.correct,
            "combined_score": record.combined_score,
            "evaluator_feedback": record.feedback(),
        }

    lm_kwargs = {"api_base": args.api_base, "max_tokens": args.max_tokens}
    if args.temperature is not None:
        lm_kwargs["temperature"] = args.temperature
    seed = None
    if args.seed_from == "skeleton":
        seed = (task.root / "skeleton.py").read_text(encoding="utf-8")
    config = OptimizeAnythingConfig(
        engine="gepa",
        name=f"adbench-tools-{task.task_id}",
        max_evals=args.budget + 1,
        max_concurrency=1,
        output_dir=str(ledger.root / "gepa_output"),
        run_dir=str(ledger.root / "gepa_state"),
        sandbox=False,
        engine_config={
            "engine": {"seed": args.seed, "parallel": False, "max_workers": 1,
                       "cache_evaluation": True, "display_progress_bar": False,
                       "raise_on_exception": False},
            "reflection": {"reflection_lm": f"openai/{args.model}",
                           "reflection_lm_kwargs": lm_kwargs},
        },
    )
    result = optimize_anything(seed, evaluator=evaluator, objective=objective(task),
                               background=task.statement, config=config)

    chosen = result.best_candidate
    chosen = chosen if isinstance(chosen, str) else next(iter(chosen.values()))
    metadata = {k: v for k, v in (result.metadata or {}).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
    (ledger.root / "gepa_result.json").write_text(
        json.dumps({"best_score": result.best_score, "total_evals": result.total_evals,
                    "metadata": metadata}, indent=2, default=str), encoding="utf-8")
    # GEPA's own choice is the deliverable; fall back to the best correct
    # candidate it evaluated only when its choice was never correct.
    final = ledger.root / "best.py"
    source = extract_code(chosen, task)
    final.write_text(source if result.best_score and result.best_score > 0
                     else (best["source"] or source), encoding="utf-8")
    return _common.finish(args, task, ledger, final, gepa_total_evals=result.total_evals,
                          gepa_metadata=metadata)


if __name__ == "__main__":
    raise SystemExit(main())
