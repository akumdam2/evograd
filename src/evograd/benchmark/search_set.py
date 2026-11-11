"""Which observed case of each model the benchmark searches, and why each other one is not.

``search_set.json`` beside this file is generated, never edited by hand:

    PYTHONPATH=src:. python scripts/adbench_selection/write_search_set.py [--check]

It names, for every model, the tracked snapshot it was generated from
(``snapshot_hash``) and, for every observed case of that model
(``observed_cases``), whether it is searched; a case that is not names one
reason from :data:`REASONS`, and ``of`` the case it defers to
(``identical_config_of``: ``<model>:<case_id>``; ``nested_in``: a case id of the
same model). The rules it applies are in the file's ``rules`` field. It holds no
gate settings and no corpus paths: the gates read the model integrations, and a
task pack binds its corpus.

*Benchmark rows*, what an evaluation's summary counts, are the searched cases
plus the ``identical_config_of`` cases, which are evaluated with the source
case's submission on their own model's payloads (:func:`evaluated`).

Imports json, functools and pathlib only: the launchers load this file by path,
without importing ``evograd`` (which imports torch).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).with_name("search_set.json")
SCHEMA = "evograd-search-set/1"
REASONS = ("vendor", "not_time_max_config", "identical_config_of", "nested_in",
           "optimizable_share_below_1pct", "removed")


@lru_cache(maxsize=None)
def _read(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    return data


def load(path: str | Path | None = None) -> dict:
    """The search set (default :data:`PATH`, read at call time)."""
    return _read(str(path or PATH))


def entry(model: str, path: str | Path | None = None) -> dict | None:
    """``{"snapshot_hash", "cases"}`` of ``model``, or ``None`` when it has none."""
    return load(path)["models"].get(model)


def _cases(model: str, path) -> dict:
    found = entry(model, path)
    if found is None:
        raise KeyError(f"{model} is not in the search set {path or PATH}")
    return found["cases"]


def searched(model: str, path: str | Path | None = None) -> list[str]:
    """The case ids of ``model`` the benchmark searches, sorted."""
    return sorted(cid for cid, case in _cases(model, path).items() if case["searched"])


def evaluated(model: str, path: str | Path | None = None) -> dict[str, tuple[str, str] | None]:
    """``{case_id: source}`` for every benchmark row of ``model``: ``None`` for a
    searched case, ``(source model, source case id)`` for an ``identical_config_of``
    case, which is evaluated with that source case's submission."""
    rows = {}
    for cid, case in sorted(_cases(model, path).items()):
        if case["searched"]:
            rows[cid] = None
        elif case["reason"] == "identical_config_of":
            source_model, source_case = case["of"].split(":", 1)
            rows[cid] = (source_model, source_case)
    return rows


def row_tag(model: str, case_id: str, path: str | Path | None = None) -> dict:
    """``{"searched", "reason", "of"}`` for an evaluation row of ``model``;
    ``{}`` for a model with no entry (a test model). A case the entry does not
    list means the file predates the snapshot: regenerate it."""
    found = entry(model, path)
    if found is None:
        return {}
    case = found["cases"].get(case_id)
    if case is None:
        raise ValueError(f"the search set lists no {model} case {case_id!r}; it predates the "
                         "snapshot: run scripts/adbench_selection/write_search_set.py")
    return {"searched": case["searched"], "reason": case["reason"], "of": case["of"]}


def benchmark_row(row: dict) -> bool:
    """Whether an evaluation row counts: searched, or ``identical_config_of``.
    A row without a tag (a model with no entry) counts."""
    return row.get("searched", True) or row.get("reason") == "identical_config_of"


__all__ = ["PATH", "REASONS", "SCHEMA", "benchmark_row", "entry", "evaluated", "load", "row_tag",
           "searched"]
