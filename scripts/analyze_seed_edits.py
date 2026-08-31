"""Measure what evolution actually edited, per seed pipeline.

    python scripts/analyze_seed_edits.py --root ~/evograd_runs/ablation --op layernorm

The claim this tests: Inductor-captured seeds (Pipeline D) carry properties that
trap the search — machine-named SSA temporaries, a huge inert config decorator,
a permutation table coupling the save-set to two call sites — so the LLM makes
cosmetic edits (block sizes, constants) instead of structural ones, and never
touches the saved-tensor contract.

That is falsifiable. For every archived candidate this compares the program text
against its trial's seed and classifies what moved:

  config-only   every changed line is inside a fixed_config/heuristics decorator
                or a launch-knob assignment (XBLOCK, R0_BLOCK, num_warps, ...)
  kernel-body   at least one line inside an @triton.jit function changed
  save-set      the backward argument permutation or the forward's return tuple
                changed — the structural move the memory axis rewards
  stale-doc     Inductor's '#   %node = call_function[...]' provenance comments
                survived unchanged in a program whose kernels did change

A pipeline whose candidates are mostly config-only, never save-set, and carry
stale provenance comments is being led around by its seed's form rather than
searching. Compare D's row against B/A/C on the same op: the seeds differ in
quality too, so only the within-op comparison is meaningful.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Launch knobs and the generated metadata that wraps them. Edits confined here
# retune a kernel; they never change what it computes.
CONFIG_LINE = re.compile(
    r"(@triton_heuristics\.|fixed_config|triton_meta|inductor_meta|DeviceProperties"
    r"|\b(XBLOCK|YBLOCK|R0_BLOCK|RBLOCK|num_warps|num_stages)\b)"
)
JIT_DECORATOR = re.compile(r"^\s*@triton\.jit")
DEF_LINE = re.compile(r"^\s*def\s+\w+\s*\(")
# Inductor reproduces the ATen graph as comments; they do not track later edits.
PROVENANCE = re.compile(r"^#\s+%|^#\s+Topologically Sorted|^#\s+Source node")
SAVE_SET = re.compile(r"_BWD_ARG_SPEC|return\s+_out\[0\]|saved_tensors\[|return\s+y,\s")


def _jit_body_lines(source: str) -> set[int]:
    """Line numbers inside an @triton.jit function body."""
    lines = source.splitlines()
    inside, body = False, set()
    for i, line in enumerate(lines):
        if JIT_DECORATOR.match(line):
            inside = True
            continue
        if inside and DEF_LINE.match(line):
            continue
        if inside:
            if line.strip() and not line.startswith((" ", "\t")) and not DEF_LINE.match(line):
                inside = False
            else:
                body.add(i)
    return body


def _classify(seed: str, candidate: str) -> dict[str, Any]:
    seed_lines, cand_lines = seed.splitlines(), candidate.splitlines()
    changed_new, changed_old = [], []
    matcher = difflib.SequenceMatcher(None, seed_lines, cand_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_old.extend(range(i1, i2))
        changed_new.extend(range(j1, j2))

    touched = [cand_lines[j] for j in changed_new] + [seed_lines[i] for i in changed_old]
    if not touched:
        return {"identical": True}

    jit_seed, jit_cand = _jit_body_lines(seed), _jit_body_lines(candidate)
    kernel_body = any(j in jit_cand for j in changed_new) or any(
        i in jit_seed for i in changed_old
    )
    config_only = all(CONFIG_LINE.search(line) for line in touched)
    save_set = any(SAVE_SET.search(line) for line in touched)
    provenance_seed = [line for line in seed_lines if PROVENANCE.match(line)]
    provenance_cand = [line for line in cand_lines if PROVENANCE.match(line)]

    return {
        "identical": False,
        "changed_lines": len(touched),
        "delta_lines": len(cand_lines) - len(seed_lines),
        "config_only": config_only,
        "kernel_body": kernel_body,
        "save_set": save_set,
        # Kernels changed but Inductor's graph comments did not: the model edited
        # code the comments claim to describe and left the claim standing.
        "stale_doc": bool(kernel_body and provenance_seed and provenance_seed == provenance_cand),
        "provenance_comments": len(provenance_cand),
    }


def _trials(root: Path, op: str):
    op_dir = root / op
    if not op_dir.is_dir():
        raise SystemExit(f"no such op directory: {op_dir}")
    for pipeline_dir in sorted(p for p in op_dir.iterdir() if p.is_dir()):
        for trial_dir in sorted(pipeline_dir.glob("trial_*")):
            index = trial_dir / "programs" / "index.jsonl"
            if index.is_file():
                yield pipeline_dir.name, trial_dir


def _programs(trial_dir: Path) -> list[tuple[dict, Path]]:
    index = trial_dir / "programs" / "index.jsonl"
    out = []
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = trial_dir / "programs" / record.get("file", "")
        if path.is_file():
            out.append((record, path))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--out", type=Path, default=None, help="write per-program JSON")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for pipeline, trial_dir in _trials(args.root, args.op):
        programs = _programs(trial_dir)
        if not programs:
            continue
        seed_source = programs[0][1].read_text(encoding="utf-8", errors="replace")
        for record, path in programs[1:]:
            source = path.read_text(encoding="utf-8", errors="replace")
            metrics = record.get("metrics") or {}
            rows.append(
                {
                    "pipeline": pipeline,
                    "trial": trial_dir.name,
                    "file": path.name,
                    "duplicate": bool(record.get("duplicate")),
                    "correct": float(metrics.get("correct", 0.0)) >= 1.0,
                    "saved_memory_ratio": metrics.get("saved_memory_ratio"),
                    "full_step_speedup": metrics.get("full_step_speedup"),
                    "seed_lines": len(seed_source.splitlines()),
                    **_classify(seed_source, source),
                }
            )

    if not rows:
        raise SystemExit(f"no archived programs under {args.root / args.op}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pipeline[row["pipeline"]].append(row)

    def pct(values: list[bool]) -> str:
        return f"{100.0 * sum(values) / len(values):5.0f}%" if values else "    —"

    header = (
        f"{'seed':<5}{'seed LOC':>9}{'cands':>7}{'dup':>6}{'wrong':>7}"
        f"{'config-only':>13}{'kernel':>8}{'save-set':>10}{'stale doc':>11}"
        f"{'|Δlines|':>10}{'smr vals':>10}"
    )
    print(f"\n{args.op}\n")
    print(header)
    print("-" * len(header))
    for pipeline, group in sorted(by_pipeline.items()):
        real = [r for r in group if not r.get("identical")]
        smr = {round(float(r["saved_memory_ratio"]), 4) for r in group if r["saved_memory_ratio"]}
        deltas = [abs(r.get("delta_lines", 0)) for r in real]
        print(
            f"{pipeline:<5}{group[0]['seed_lines']:>9}{len(group):>7}"
            f"{pct([r['duplicate'] for r in group]):>6}"
            f"{pct([not r['correct'] for r in group]):>7}"
            f"{pct([r.get('config_only', False) for r in real]):>13}"
            f"{pct([r.get('kernel_body', False) for r in real]):>8}"
            f"{pct([r.get('save_set', False) for r in real]):>10}"
            f"{pct([r.get('stale_doc', False) for r in real]):>11}"
            f"{(statistics.median(deltas) if deltas else 0):>10.0f}"
            f"{len(smr):>10}"
        )
    print(
        "\nconfig-only = every changed line is a launch knob or generated metadata"
        "\nsave-set    = the saved-tensor contract moved (the memory-axis move)"
        "\nstale doc   = kernels changed, Inductor's ATen provenance comments did not"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
