"""Cross-evaluate GEPA's Pareto archive and emit a contiguous regime dispatcher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

from evograd.gepa_backend.archive import (
    best_generalist_index,
    load_result,
    pareto_union,
)
from evograd.gepa_backend.evaluator import ShapeBatchEvaluator, examples_for_suite


def optimize_contiguous(
    rows: list[int],
    candidate_scores: dict[int, list[float]],
    *,
    max_segments: int = 3,
) -> tuple[float, list[tuple[int, int, int]]]:
    """Maximize sum(log(score)) with up to ``max_segments`` contiguous ranges."""
    if not rows or not candidate_scores:
        raise ValueError("rows and candidate_scores must be non-empty")
    count = len(rows)
    candidates = sorted(candidate_scores)
    logs = {
        candidate: [math.log(max(float(score), 1e-12)) for score in scores]
        for candidate, scores in candidate_scores.items()
    }
    prefix = {
        candidate: [0.0]
        for candidate in candidates
    }
    for candidate in candidates:
        for value in logs[candidate]:
            prefix[candidate].append(prefix[candidate][-1] + value)

    neg_inf = float("-inf")
    dp = [[neg_inf] * (count + 1) for _ in range(max_segments + 1)]
    paths: list[list[list[tuple[int, int, int]] | None]] = [
        [None] * (count + 1) for _ in range(max_segments + 1)
    ]
    dp[0][0] = 0.0
    paths[0][0] = []
    for segments in range(1, max_segments + 1):
        for end in range(1, count + 1):
            for start in range(segments - 1, end):
                if dp[segments - 1][start] == neg_inf:
                    continue
                for candidate in candidates:
                    segment_score = prefix[candidate][end] - prefix[candidate][start]
                    score = dp[segments - 1][start] + segment_score
                    if score > dp[segments][end]:
                        dp[segments][end] = score
                        previous = paths[segments - 1][start] or []
                        paths[segments][end] = previous + [(start, end, candidate)]

    best_segments = max(
        range(1, max_segments + 1), key=lambda segments: dp[segments][count]
    )
    return dp[best_segments][count], paths[best_segments][count] or []


def _dispatcher_source(
    *,
    segments: list[tuple[int, int, int]],
    rows: list[int],
    generalist: int,
    filenames: dict[int, str],
) -> str:
    indices = sorted(set([generalist, *(candidate for _, _, candidate in segments)]))
    load_lines = "\n".join(
        f"_PROGRAMS[{index}] = _load({filenames[index]!r}, {index})" for index in indices
    )
    route_lines = []
    for _start, end, candidate in segments[:-1]:
        route_lines.append(f"    if rows <= {rows[end - 1]}:\n        return {candidate}")
    route_lines.append(f"    return {segments[-1][2]}")
    route_body = "\n".join(route_lines)
    return f'''"""Generated GEPA contiguous LayerNorm specialist dispatcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PROGRAMS = {{}}

def _load(filename, index):
    path = Path(__file__).with_name("specialists") / filename
    spec = importlib.util.spec_from_file_location(f"_gepa_specialist_{{index}}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

{load_lines}

def _route(rows, hidden):
    if hidden != 1024:
        return {generalist}
{route_body}

def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    route = _route(int(x.shape[0]), int(x.shape[1]))
    y, saved = _PROGRAMS[route].layernorm_forward_with_saved(x, weight, bias, eps)
    inner = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
    return y, (route, *inner)

def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    route, *inner = tuple(saved_tensors)
    return _PROGRAMS[int(route)].layernorm_backward_from_saved(dy, tuple(inner), eps)
'''


def build_dispatch(
    *,
    run_dir: Path,
    output_dir: Path,
    seed_path: Path,
    max_segments: int = 3,
) -> dict[str, Any]:
    result = load_result(run_dir / "gepa_result.json")
    summary = load_result(run_dir / "run_summary.json")
    generalist = best_generalist_index(result)
    indices = sorted(set([generalist, *pareto_union(result)]))
    shapes = examples_for_suite("layernorm", "tb_sweep")
    rows = [int(shape["dims"]["rows"]) for shape in shapes]
    evaluator = ShapeBatchEvaluator(
        seed_path=seed_path,
        cache_dir=output_dir / "cross_eval_cache",
        warmup=10,
        reps=50,
    )
    bodies = {
        index: (run_dir / "candidates" / f"candidate_{index:04d}_block.py").read_text(
            encoding="utf-8"
        )
        for index in indices
    }
    pairs = [(body, shape) for index, body in bodies.items() for shape in shapes]
    evaluated = evaluator.evaluate_batch(pairs)
    score_matrix = {}
    records = {}
    cursor = 0
    for index in indices:
        scores = []
        records[str(index)] = {}
        for shape in shapes:
            score, info = evaluated[cursor]
            cursor += 1
            scores.append(float(score))
            records[str(index)][shape["id"]] = {"score": score, "info": info}
        score_matrix[index] = scores

    objective, segments = optimize_contiguous(
        rows, score_matrix, max_segments=max_segments
    )
    chosen = sorted(set([generalist, *(candidate for _, _, candidate in segments)]))
    specialists = output_dir / "specialists"
    specialists.mkdir(parents=True, exist_ok=False)
    filenames = {}
    for index in chosen:
        filename = f"candidate_{index:04d}.py"
        filenames[index] = filename
        shutil.copy2(run_dir / "candidates" / filename, specialists / filename)

    dispatcher = output_dir / "dispatched_program.py"
    dispatcher.write_text(
        _dispatcher_source(
            segments=segments,
            rows=rows,
            generalist=generalist,
            filenames=filenames,
        ),
        encoding="utf-8",
    )
    payload = {
        "generalist_idx": generalist,
        "pareto_indices": pareto_union(result),
        "candidate_indices_cross_evaluated": indices,
        "rows": rows,
        "segments": [
            {
                "start_index": start,
                "end_index": end,
                "rows_min": rows[start],
                "rows_max": rows[end - 1],
                "candidate_idx": candidate,
            }
            for start, end, candidate in segments
        ],
        "log_fitness_objective": objective,
        "records": records,
        "dispatcher": str(dispatcher),
        "gepa_summary": summary,
    }
    (output_dir / "dispatch_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_dispatch(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        seed_path=args.seed,
        max_segments=args.max_segments,
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
