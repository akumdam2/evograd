#!/usr/bin/env python3
"""Write ``src/evograd/benchmark/search_set.json``: what the benchmark searches, and why not.

    PYTHONPATH=src:. python scripts/adbench_selection/write_search_set.py [--check]

(``src/evograd/benchmark/README.md`` has the full command, with the test.)

Inputs: every observed case of every model (``observed_cases``, from the tracked
snapshots) and the measurements committed beside this script, nothing else:

* ``l1_timing.json``: eager forward+backward ms per call of every configuration of
  every multi-configuration L1 task (``l1_timing.py``, GH200, 2026-11-10), found by
  task, dims and dtype;
* ``l2_share/report.json``: per L2 configuration its eager wall time and its
  optimizable share of the eager training step (``l2_share/``, GH200, 2026-11-10),
  with the calls per step and dims it measured.

The rules are :data:`RULES`, applied by :func:`select`. An observed case the inputs
do not cover, or cover with other dims, dtype or (outside :data:`RESCALED`) calls,
stops the script; it never defaults to searched or reuses a measurement. ``--check`` builds
the file in memory and exits 1 when the committed one differs: regenerate after
any snapshot or input change (the CPU test ``tests/test_search_set.py`` pins the
file to the snapshots).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "src" / "evograd" / "benchmark" / "search_set.json"
L1_TIMING = HERE / "l1_timing.json"
L2_REPORT = HERE / "l2_share" / "report.json"
SCHEMA = "evograd-search-set/1"

VENDOR = ("causal_gqa_attention", "linear", "linear_no_bias")
SHARE_MIN_PCT = 1.0
#: Removed L2 tasks: note.
REMOVED = {"megafold_layernorm_linear": (
    "removed 2026-11-10: MegaFold's own LayerNorm+Linear fusion, not in upstream; 0.66x in "
    "the earlier campaign; nested in four composite L2 tasks; ~1000 calls per step")}
#: L2 task -> (module-path marker, the tasks whose cases contain such configurations).
NESTED = {"megafold_evoattention": (".tri_attn", ("megafold_triangle_attention_starting",
                                                  "megafold_triangle_attention_ending"))}
SWIGLU = ("qwen3_swiglu_mlp", "llama3_swiglu_mlp", "qwen3_5_swiglu_mlp", "qwen2_vl_2b_text_mlp")
#: L2 tasks searched whatever their optimizable share: why.
EXCEPTIONS = {**{task: "the swiglu_mlp shape-generalisation ablation group" for task in SWIGLU},
              "torchrec_dlrm_dlrm_interaction": "keeps DLRM an L2/L3 case (decided 2026-11-10)"}
#: L2 tasks report.json measured under another name: task -> (report task, the report
#: config that measured a configuration, from its dims and short dtype). The merged
#: attention blocks are the sum of the two tasks each replaces; the others were synthetic
#: blocks before the task existed, their labels naming the dims (and dtype) they ran.
MEASURED_AS = {
    "qwen3_attention_block": ("attention_block", lambda d, t: "qwen3_qkv_norm_rope+qwen3_attention"),
    "llama3_attention_block": ("attention_block", lambda d, t: "llama3_qkv_rope+llama3_attention"),
    "olmoe_attention_block": ("attention_block", lambda d, t: "olmoe_qkv_norm_rope+olmoe_attention"),
    "qwen3_5_attention_block": ("gated_attention_block",
                                lambda d, t: "qwen3_5_qkv_norm_partial_rope+qwen3_5_gated_attention"),
    "qwen3_lm_head_ce": ("lm_head_cross_entropy",
                         lambda d, t: f"x[{d['rows']},{d['hidden']}] V{d['vocab']}"),
    "convnextv2_tiny_dwconv_layernorm": ("dwconv_layernorm",
                                         lambda d, t: f"C{d['C']} {d['Y']}x{d['X']}"),
    "dit_s_2_dit_attention": ("dit_attention", lambda d, t: f"x[{d['B']},{d['T']},{d['H']}] {t}"),
    "dit_s_2_dit_adaln_residual": ("dit_adaln_residual",
                                   lambda d, t: f"x[{d['B']},{d['T']},{d['H']}] {t}"),
    "qwen2_vl_2b_vision_attention": ("vision_attention", lambda d, t: f"x[{d['T']},{d['H']}] {t}"),
    "qwen2_vl_2b_vision_residual_layernorm": ("vision_residual_layernorm",
                                              lambda d, t: f"[{d['T']},{d['H']}] {t}"),
    "qwen2_vl_2b_text_attention_block": ("text_attention_block",
                                         lambda d, t: f"x[{d['B']},{d['T']},{d['H']}] {t}"),
    "qwen2_vl_2b_text_residual_rmsnorm": ("residual_rmsnorm_text",
                                          lambda d, t: f"[{d['B'] * d['T']},{d['H']}] {t}"),
}
SHORT_DTYPE = {"bfloat16": "bf16", "float32": "fp32"}
#: The only L2 configurations whose calls per step may differ from the calls report.json
#: measured, the share then scaled by calls: task -> (calls measured, calls in the
#: snapshot), or None for a new task measured as one synthetic block, whose row stands
#: for each of its configurations. Any other difference stops the script.
RESCALED = {
    "olmoe_residual_rmsnorm": (16, 32),    # 4a2de4f: the residual site at 32 calls
    "qwen3_5_residual_rmsnorm": (32, 64),  # cc0f94e: both residual adds per layer fused
    **dict.fromkeys(("qwen3_lm_head_ce", "convnextv2_tiny_dwconv_layernorm",
                     "dit_s_2_dit_attention", "dit_s_2_dit_adaln_residual",
                     "qwen2_vl_2b_vision_attention", "qwen2_vl_2b_vision_residual_layernorm",
                     "qwen2_vl_2b_text_attention_block", "qwen2_vl_2b_text_residual_rmsnorm")),
}

RULES = (
    "L1: linear, linear_no_bias and causal_gqa_attention are vendor primitives (GEMM, SDPA); "
    "their cases are not searched (vendor).",
    "L1: per (model, task) the searched configuration is the one with the largest eager time "
    "share per step, forward+backward ms per call (l1_timing.json, GH200, order 303) x calls "
    "per step; a single-configuration task searches its one case. The others: "
    "not_time_max_config. A configuration's timing is the l1_timing.json row of its task, dims "
    "and dtype (its case id breaks a tie between layouts); a row under another case id stands "
    "for it only when that id is no longer observed (a renamed case).",
    "L1: a time-max configuration identical to another model's (same task, dims, dtype, layout, "
    "strides and free sizes) is searched once, in the model that calls it most per step; the "
    "others are identical_config_of that case (of = model:case_id) and are evaluated with its "
    "submission on their own payloads.",
    "L2: megafold_layernorm_linear is removed (2026-11-10).",
    "L2: a megafold_evoattention configuration inside triangle attention (module paths "
    "containing tri_attn) whose containing triangle_attention case is searched is nested_in "
    "it (L2-3: the two searches would install one inside the other; of = that case's id, the "
    "note names any other searched container); one whose containers are not searched is a "
    "configuration of the task like the others.",
    "L2: per (model, task) the searched configuration is the one with the largest eager time "
    "per step, forward+backward wall ms per call x calls per step (l2_share/report.json), "
    "among the configurations not nested_in a searched case; the others: not_time_max_config.",
    "L2: a searched configuration's optimizable share of the eager step (L2-5: non-library "
    "kernel ms, or wall minus library ms where the step is not CUDA-graph captured, x calls / "
    "eager step ms; l2_share/report.json) must be at least 1%; below it: "
    "optimizable_share_below_1pct.",
    "L2 exceptions, searched whatever their share (reason null, note says why): every swiglu "
    "MLP (qwen3_swiglu_mlp, llama3_swiglu_mlp, qwen3_5_swiglu_mlp, qwen2_vl_2b_text_mlp) as "
    "the shape-generalisation ablation group; torchrec_dlrm_dlrm_interaction, which keeps "
    "DLRM an L2/L3 case (2026-11-10).",
    "Benchmark rows, the evaluation's denominators: the searched cases plus the "
    "identical_config_of cases.",
)
NOTES = (
    "MegaFold AdaLN (megafold_adaptive_layernorm in l2_share/report.json) is not a task: "
    "deferred 2026-11-10.",
    "report.json measured the merged attention blocks as the sum of the two tasks each "
    "replaces (rows attention_block, gated_attention_block) and the tasks added since as "
    "synthetic blocks; MEASURED_AS in write_search_set.py maps them by meaning to the task "
    "names, a configuration to a row by its dims.",
    "An L2 share is report.json's per-call optimizable time x this configuration's calls per "
    "step / the eager step. report.json's row must have measured these calls and (a corpus "
    "row, or the corpus rows a merged row sums) these dims and dtype; only RESCALED in "
    "write_search_set.py may differ in calls (olmoe and qwen3_5 residual_rmsnorm, 16->32 and "
    "32->64; the tasks measured as synthetic blocks).",
    "No gate settings and no corpus paths: the gates read the model integrations, a task "
    "pack binds its corpus.",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(case) -> tuple:
    """What makes two configurations of one task identical across models."""
    w = case.workload
    p = w.provenance
    return (case.task, json.dumps(dict(w.dims), sort_keys=True), str(w.dtype), p.layout,
            json.dumps(p.strides), json.dumps(dict(p.free or {}), sort_keys=True))


def row(searched, reason=None, of=None, note=None, **evidence) -> dict:
    return {"searched": searched, "reason": reason, "of": of, "note": note,
            "evidence": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in evidence.items()}}


def _content(model: str, task: str, dims, dtype) -> tuple:
    return model, task, json.dumps(dict(dims), sort_keys=True), str(dtype)


def timing_index(l1: dict) -> tuple[dict, dict]:
    """l1_timing.json's rows by content (``_content``) and by ``(model, case id)``."""
    by_content: dict[tuple, list] = {}
    for r in l1["rows"]:
        by_content.setdefault(_content(r["model"], r["task"], r["dims"], r["dtype"]), []).append(r)
    return by_content, {(r["model"], r["case"]): r for r in l1["rows"]}


def timed_l1(model: str, case, timing: tuple[dict, dict], observed: set[str]) -> dict:
    """The l1_timing.json row that timed ``case``: the one of its task, dims and dtype,
    so a renamed case id still resolves, but only to a row whose case id is no longer
    among ``observed`` (the model's case ids): a still-observed configuration's timing
    is never reused. The id breaks a tie between configurations that differ only in
    layout, and must not name another configuration's row."""
    by_content, by_id = timing
    w = case.workload
    hits = by_content.get(_content(model, case.task, w.dims, w.dtype), [])
    named = by_id.get((model, case.case_id))
    where = f"{model} {case.case_id} ({case.task} {dict(w.dims)} {w.dtype})"
    if len(hits) > 1:
        if named is None or not any(r is named for r in hits):
            raise SystemExit(f"{where}: {len(hits)} {L1_TIMING.name} rows have its task, dims and "
                             f"dtype ({[r['case'] for r in hits]}) and none its case id; time it "
                             "(l1_timing.py) before regenerating")
        hits = [named]
    if not hits:
        timed = "" if named is None else (f"; its case id was timed as {named['task']} "
                                          f"{named['dims']} {named['dtype']}")
        raise SystemExit(f"{where}: no eager timing of its task, dims and dtype in "
                         f"{L1_TIMING.name}{timed}; time it (l1_timing.py) before regenerating")
    (found,) = hits
    if named is not None and named is not found:
        raise SystemExit(f"{where}: {L1_TIMING.name} timed its configuration as {found['case']} "
                         f"and its case id as {named['dims']} {named['dtype']}; re-time both")
    if found["case"] != case.case_id and found["case"] in observed:
        raise SystemExit(f"{where}: its task, dims and dtype were timed only as {found['case']}, "
                         f"still an observed case of {model}; time it (l1_timing.py) before "
                         "regenerating")
    if found.get("ms") is None:
        raise SystemExit(f"{where}: {L1_TIMING.name} row {found['case']} has no ms; time it again")
    return found


def _by_task(cases):
    tasks: dict[str, list] = {}
    for case in cases:
        tasks.setdefault(case.task, []).append(case)
    return tasks


def level1(model: str, cases, timing: tuple[dict, dict]) -> tuple[dict, dict]:
    """Rows of ``model``'s L1 cases and ``{task: time-max case}``."""
    rows, mains = {}, {}
    observed = {c.case_id for c in cases}
    for task, group in _by_task(cases).items():
        if task in VENDOR:
            rows.update({c.case_id: row(False, "vendor", calls=c.frequency) for c in group})
            continue
        if len(group) == 1:
            (case,) = group
            rows[case.case_id] = row(True, calls=case.frequency)
            mains[task] = case
            continue
        timed = {c.case_id: timed_l1(model, c, timing, observed) for c in group}
        best = max(group, key=lambda c: timed[c.case_id]["ms"] * c.frequency)
        mains[task] = best
        for case in group:
            ms, renamed = timed[case.case_id]["ms"], timed[case.case_id]["case"]
            rows[case.case_id] = row(case is best, None if case is best else "not_time_max_config",
                                     ms=ms, calls=case.frequency, ms_x_calls=ms * case.frequency,
                                     **({} if renamed == case.case_id else {"timed_as": renamed}))
    return rows, mains


def identical(mains: dict[str, dict], rows: dict[str, dict], order: list[str]) -> None:
    """Search an identical L1 configuration once, in the model that calls it most."""
    groups: dict[tuple, list] = {}
    for model in order:
        for case in mains[model].values():
            groups.setdefault(signature(case), []).append((model, case))
    for members in groups.values():
        if len(members) < 2:
            continue
        source_model, source = max(members, key=lambda m: (m[1].frequency, -order.index(m[0])))
        copies = [(model, case) for model, case in members if case is not source]
        rows[source_model][source.case_id]["note"] = (
            "searched once for the identical " + ", ".join(f"{m}:{c.case_id}" for m, c in copies))
        for model, case in copies:
            old = rows[model][case.case_id]
            rows[model][case.case_id] = {
                **old, "searched": False, "reason": "identical_config_of",
                "of": f"{source_model}:{source.case_id}",
                "note": (f"same task, dims, dtype, layout, strides and free sizes; "
                         f"{source.frequency} calls per step there, {case.frequency} here")}


def report_rows(report: dict) -> dict:
    """``{(model, report task, config): row}``."""
    return {(r["model"], r["task"], r["config"]): r for r in report["rows"]}


def merged_dims(model: str, found: dict, rows: dict, where: str) -> tuple[dict, str]:
    """A merged row's dims and dtype: those of the corpus rows it sums (its config
    names their tasks, ``a+b``), which must measure its calls and agree on shared dims."""
    dims, dtypes = {}, set()
    for part in found["config"].split("+"):
        hits = [r for (m, t, _), r in rows.items() if (m, t) == (model, part)]
        if len(hits) != 1 or hits[0]["kind"] != "corpus" or hits[0].get("dims") is None:
            raise SystemExit(f"{where}: {found['task']} {found['config']} in {L2_REPORT.name} "
                             f"sums {part!r}: {len(hits)} rows of it, not one corpus row with dims")
        (part_row,) = hits
        clash = {k for k, v in part_row["dims"].items() if dims.get(k, v) != v}
        if clash or part_row["calls"] != found["calls"]:
            raise SystemExit(f"{where}: the rows {found['config']} sums disagree ({part} "
                             f"{part_row['calls']} calls, {part_row['dims']}); measure it again")
        dims.update(part_row["dims"])
        dtypes.add(part_row["dtype"])
    if len(dtypes) != 1:
        raise SystemExit(f"{where}: the rows {found['config']} sums ran {sorted(dtypes)}")
    return dims, dtypes.pop()


def measured_l2(model: str, case, rows: dict) -> dict:
    """The report.json row that measured ``case``, and how much it says. The row must
    have measured the snapshot's calls (outside :data:`RESCALED`) and, where it records
    them (a merged row: the corpus rows it sums), its dims and dtype."""
    task, config = case.task, case.case_id
    dims, dtype = dict(case.workload.dims), str(case.workload.dtype)
    where = f"{model} {case.case_id}"
    if task in MEASURED_AS:
        name, label = MEASURED_AS[task]
        try:
            task, config = name, label(dims, SHORT_DTYPE.get(dtype, dtype))
        except KeyError as exc:
            raise SystemExit(f"{where}: MEASURED_AS[{case.task!r}] needs dim {exc}; dims "
                             f"{dims}") from None
    found = rows.get((model, task, config))
    if found is None or found.get("wall_ms") is None or found.get("share_pct") is None:
        raise SystemExit(f"{where}: no measurement in {L2_REPORT.name} "
                         f"(looked for task {task!r}, config {config!r}); measure it (l2_share/) "
                         "or map it in MEASURED_AS before regenerating")
    measured = (merged_dims(model, found, rows, where) if found["kind"] == "merged"
                else (found.get("dims"), found.get("dtype")))
    if measured[0] is not None and measured != (dims, dtype):
        raise SystemExit(f"{where}: {L2_REPORT.name} measured {measured[0]} {measured[1]}, "
                         f"the snapshot has {dims} {dtype}; measure it again (l2_share/)")
    if found["calls"] != case.frequency:
        allowed = RESCALED.get(case.task, ())
        if not (allowed == (found["calls"], case.frequency)
                or (allowed is None and found["kind"] == "synthetic")):
            raise SystemExit(f"{where}: {L2_REPORT.name} measured {found['calls']} calls per step, "
                             f"the snapshot has {case.frequency}; measure it again (l2_share/) or, "
                             "for a documented call-count change, list it in RESCALED")
    share = found["share_pct"] * case.frequency / found["calls"]
    return {"wall_ms": found["wall_ms"], "calls": case.frequency,
            "wall_x_calls": found["wall_ms"] * case.frequency, "optimizable_share_pct": share,
            "measured_as": f"{found['kind']}: {found['task']} {found['config']}"}


def nested_in(case, cases, searched: set[str]) -> tuple[str, str] | None:
    """``(container case id, note)`` when ``case`` runs inside a searched case of
    another L2 task (L2-3: a real co-install conflict); ``None`` when it does not,
    including when no case holding it is searched."""
    marker, tasks = NESTED.get(case.task, (None, ()))
    if not marker or not any(marker in path for path in case.paths):
        return None
    hits = {}
    for other in cases:
        if other.task in tasks:
            inside = [p for p in case.paths if any(p.startswith(q + ".") for q in other.paths)]
            if inside:
                hits[other.case_id] = inside
    if not hits:
        raise SystemExit(f"{case.case_id}: its paths name {marker!r} but no case of {tasks} "
                         "holds any of its modules")
    held = [cid for cid in hits if cid in searched]
    if not held:
        return None
    first = next((cid for cid in held if case.paths[0] in hits[cid]), held[0])
    note = "; ".join(f"{len(hits[cid])} of its {len(case.paths)} calls inside {cid}"
                     for cid in [first, *sorted(set(held) - {first})])
    return first, note


def level2(model: str, cases, report: dict) -> dict:
    rows = {}
    groups = _by_task(cases)
    # The containers' tasks first: nested_in needs to know which of their cases are searched.
    for task in sorted(groups, key=lambda name: name in NESTED):
        group = groups[task]
        if task in REMOVED:
            rows.update({c.case_id: row(False, "removed", note=REMOVED[task], calls=c.frequency)
                         for c in group})
            continue
        evidence = {c.case_id: measured_l2(model, c, report) for c in group}
        shared = {}
        for case in group:
            shared.setdefault(evidence[case.case_id]["measured_as"], set()).add(
                json.dumps(dict(case.workload.dims), sort_keys=True))
        if any(len(dims) > 1 for dims in shared.values()):
            raise SystemExit(f"{model} {task}: one report.json row measured configurations of "
                             f"different dims ({shared}); measure each configuration")
        rest = []
        for case in group:
            inside = nested_in(case, cases, {cid for cid, r in rows.items() if r["searched"]})
            if inside is None:
                rest.append(case)
            else:
                rows[case.case_id] = row(False, "nested_in", inside[0], inside[1],
                                         **evidence[case.case_id])
        best = max(rest, key=lambda c: evidence[c.case_id]["wall_x_calls"], default=None)
        for case in rest:
            measured = evidence[case.case_id]
            if case is not best:
                rows[case.case_id] = row(False, "not_time_max_config", **measured)
                continue
            share = measured["optimizable_share_pct"]
            if task in EXCEPTIONS:
                note = (f"exception: {EXCEPTIONS[task]}; optimizable share {share:.2f}% < "
                        f"{SHARE_MIN_PCT:g}%" if share < SHARE_MIN_PCT else EXCEPTIONS[task])
                rows[case.case_id] = row(True, note=note, **measured)
            elif share < SHARE_MIN_PCT:
                rows[case.case_id] = row(False, "optimizable_share_below_1pct", **measured)
            else:
                rows[case.case_id] = row(True, **measured)
    return rows


def select() -> dict:
    """The search set from the snapshots and the committed measurements."""
    from evograd.benchmark.topdown import load_snapshot
    from evograd.verification.integrations import PACKAGES, load
    from evograd.verification.integrations.base import observed_cases

    timing = timing_index(json.loads(L1_TIMING.read_text(encoding="utf-8")))
    report = report_rows(json.loads(L2_REPORT.read_text(encoding="utf-8")))
    order = list(PACKAGES)
    observed, rows, mains, snapshots = {}, {}, {}, {}
    for name in order:
        model = load(name)
        observed[name] = observed_cases(model)
        snapshots[name] = load_snapshot(model.name)["snapshot_hash"]
        rows[name], mains[name] = level1(name, [c for c in observed[name] if c.level == 1], timing)
        rows[name].update(level2(name, [c for c in observed[name] if c.level == 2], report))
    identical(mains, rows, order)
    models = {}
    for name in order:
        models[name] = {"snapshot_hash": snapshots[name],
                        "cases": {c.case_id: {"level": c.level, "task": c.task, **rows[name][c.case_id]}
                                  for c in observed[name]}}
    return {"schema": SCHEMA, "generated_by": "scripts/adbench_selection/write_search_set.py",
            "rules": list(RULES), "notes": list(NOTES),
            "inputs": {"l1_timing": {"path": str(L1_TIMING.relative_to(REPO)),
                                     "sha256": sha256(L1_TIMING)},
                       "l2_share": {"path": str(L2_REPORT.relative_to(REPO)),
                                    "sha256": sha256(L2_REPORT),
                                    "threshold_pct": SHARE_MIN_PCT}},
            "models": models}


def render(data: dict) -> str:
    """Indented JSON with one line per case, so a diff shows each changed case."""
    head = json.dumps({k: v for k, v in data.items() if k != "models"}, indent=1)
    models = []
    for name, entry in data["models"].items():
        cases = ",\n".join(f"    {json.dumps(cid)}: {json.dumps(case)}"
                           for cid, case in entry["cases"].items())
        models.append(f'  {json.dumps(name)}: {{\n   "snapshot_hash": '
                      f'{json.dumps(entry["snapshot_hash"])},\n   "cases": {{\n{cases}\n   }}\n  }}')
    text = head[:-2] + ',\n "models": {\n' + ",\n".join(models) + "\n }\n}\n"
    if json.loads(text) != data:
        raise AssertionError("render() changed the data")
    return text


def summary(data: dict) -> str:
    lines = []
    for name, entry in data["models"].items():
        counts: dict[str, int] = {}
        for case in entry["cases"].values():
            key = f"L{case['level']} " + ("searched" if case["searched"] else case["reason"])
            counts[key] = counts.get(key, 0) + 1
        lines.append(f"{name}: {len(entry['cases'])} observed; "
                     + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the committed file differs from a fresh build")
    args = parser.parse_args(argv)
    data = select()
    text = render(data)
    print(summary(data))
    if args.check:
        if OUT.is_file() and OUT.read_text(encoding="utf-8") == text:
            print(f"{OUT.relative_to(REPO)} is current")
            return 0
        print(f"{OUT.relative_to(REPO)} is stale: run {Path(__file__).relative_to(REPO)} "
              "without --check and commit the result", file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
