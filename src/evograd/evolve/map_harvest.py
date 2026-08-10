"""Harvest shape-regime specialists from an OpenEvolve MAP-Elites archive."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")


def latest_checkpoint_dir(evolve_dir: Path) -> Path:
    root = Path(evolve_dir) / "checkpoints"
    if not root.is_dir():
        raise FileNotFoundError(f"no checkpoints under {evolve_dir}")
    numbered: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _CHECKPOINT_RE.match(path.name)
        if match and path.is_dir():
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError(f"no checkpoint_* dirs under {root}")
    return max(numbered, key=lambda item: item[0])[1]


def load_checkpoint_programs(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    programs_dir = Path(checkpoint_dir) / "programs"
    if not programs_dir.is_dir():
        raise FileNotFoundError(f"programs dir missing: {programs_dir}")
    programs: dict[str, dict[str, Any]] = {}
    for path in programs_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        program_id = str(payload.get("id") or path.stem)
        programs[program_id] = payload
    return programs


def _metric(program: dict[str, Any], name: str) -> float:
    metrics = program.get("metrics") or {}
    try:
        return float(metrics.get(name, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _pick_best(
    programs: dict[str, dict[str, Any]],
    ids: list[str],
    primary: str,
    *,
    tie_break: str = "combined_score",
    require_correct: bool = True,
) -> str | None:
    candidates = [pid for pid in ids if pid in programs]
    if require_correct:
        candidates = [
            pid for pid in candidates if _metric(programs[pid], "correct") >= 1.0
        ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda pid: (_metric(programs[pid], primary), _metric(programs[pid], tie_break)),
    )


def harvest_regime_elites(
    evolve_dir: Path,
    *,
    checkpoint_dir: Path | None = None,
    archive_only: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Select full / small-strong / large-strong elites and write their code.

    Selection (方案 B):
    - ``full``: max ``combined_score``
    - ``small``: max ``small_regime_speedup`` (tie-break ``combined_score``)
    - ``large``: max ``large_regime_speedup`` (tie-break ``combined_score``)
    """
    checkpoint = checkpoint_dir or latest_checkpoint_dir(evolve_dir)
    programs = load_checkpoint_programs(checkpoint)
    metadata_path = checkpoint / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    archive_ids = [str(pid) for pid in metadata.get("archive", [])]
    pool_ids = archive_ids if archive_only and archive_ids else list(programs)
    if not pool_ids:
        raise RuntimeError(f"no programs available in {checkpoint}")
    correct_pool_ids = [
        pid
        for pid in pool_ids
        if pid in programs and _metric(programs[pid], "correct") >= 1.0
    ]
    if not correct_pool_ids:
        raise RuntimeError(f"no correct programs available in {checkpoint}")

    full_id = _pick_best(programs, correct_pool_ids, "combined_score")
    small_id = _pick_best(programs, correct_pool_ids, "small_regime_speedup")
    large_id = _pick_best(programs, correct_pool_ids, "large_regime_speedup")
    if full_id is None:
        raise RuntimeError("could not select a full elite")

    out = Path(output_dir) if output_dir is not None else Path(evolve_dir) / "harvested"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    selected = {"full": full_id, "small": small_id, "large": large_id}
    for tag, program_id in selected.items():
        if program_id is None:
            continue
        code = programs[program_id].get("code")
        if not isinstance(code, str) or not code.strip():
            raise RuntimeError(f"{tag} elite {program_id} has empty code")
        path = out / f"{tag}_elite.py"
        path.write_text(code, encoding="utf-8")
        paths[tag] = path

    report = {
        "checkpoint": str(checkpoint),
        "archive_only": archive_only,
        "pool_size": len(pool_ids),
        "correct_pool_size": len(correct_pool_ids),
        "correct_pool_ids": correct_pool_ids,
        "distinct_regime_elites": (
            small_id is not None and large_id is not None and small_id != large_id
        ),
        "occupied_correct_descriptor_points": len(
            {
                (
                    _metric(programs[pid], "small_regime_speedup"),
                    _metric(programs[pid], "large_regime_speedup"),
                )
                for pid in correct_pool_ids
            }
        ),
        "selected": {
            tag: {
                "id": program_id,
                "metrics": {
                    key: _metric(programs[program_id], key)
                    for key in (
                        "combined_score",
                        "small_regime_speedup",
                        "large_regime_speedup",
                    )
                },
                "path": str(paths[tag]),
            }
            for tag, program_id in selected.items()
            if program_id is not None and tag in paths
        },
        "programs": {tag: str(path) for tag, path in paths.items()},
    }
    report_path = out / "harvest_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
