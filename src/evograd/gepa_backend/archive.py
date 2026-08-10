"""Utilities for extracting instance-wise Pareto candidates from GEPA results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pareto_union(result: dict[str, Any]) -> list[int]:
    indices = set()
    for winners in result["per_val_instance_best_candidates"].values():
        indices.update(int(index) for index in winners)
    return sorted(indices)


def best_generalist_index(result: dict[str, Any]) -> int:
    scores = [float(score) for score in result["val_aggregate_scores"]]
    return max(range(len(scores)), key=scores.__getitem__)


def shape_winner_map(
    result: dict[str, Any],
    valset: list[dict[str, Any]],
) -> dict[str, list[int]]:
    output = {}
    frontier = result["per_val_instance_best_candidates"]
    for index, example in enumerate(valset):
        winners = frontier.get(str(index), frontier.get(index, []))
        output[example["id"]] = sorted(int(candidate) for candidate in winners)
    return output
