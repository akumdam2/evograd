"""L2-5 report (2026-11-10): optimizable share of each L2 case, E-1 selection of multi-config tasks.

Record of the report, kept as it ran (its paths are DeltaAI's). Run 2026-11-10 18:41 CST over
the GH200 rows of l2_share.py (gh096, rows.jsonl) and steps/*.json; output
report.json beside this file. write_search_set.py reads report.json's rows: per configuration
its wall time and optimizable share, and the merged attention-block rows.

share = optimizable ms per call x calls per step / eager step ms, in the execution that decides the
model's Level 3:
  T3-G (CUDA-graph replay) models: optimizable = non-library kernel time; step = T3-G eager step.
  models whose step is not captured (MegaFold, OLMoE): optimizable = wall time - library kernel
  time (launch overhead counts there); step = eager T3 step.
Library kernels = GEMM and SDPA. Threshold 1 %.
"""

import collections
import json
import re
import sys

D = "/u/wzhan/.cache/case_rules/l2_share"
THRESHOLD = 1.0
NOT_CAPTURED = {"megafold", "olmoe_1b_7b"}
# MegaFold's eager T3 step: docs/AF3.md section 11, measured 2026-11-06 at 0d1a761 (384 residues).
EXTERNAL_STEP = {"megafold": (8340.0, "docs/AF3.md T3 reference line, 2026-11-06")}
# Proposed merged attention blocks = the two existing tasks (o_proj is a GEMM: library only).
MERGED = {
    ("qwen3_0_6b", "attention_block"): ["qwen3_qkv_norm_rope", "qwen3_attention"],
    ("llama_3_2_1b", "attention_block"): ["llama3_qkv_rope", "llama3_attention"],
    ("olmoe_1b_7b", "attention_block"): ["olmoe_qkv_norm_rope", "olmoe_attention"],
    ("qwen3_5_4b", "gated_attention_block"): ["qwen3_5_qkv_norm_partial_rope", "qwen3_5_gated_attention"],
}
PENDING = {"convnextv2_tiny_convnext_mlp", "qwen2_vl_2b_vision_mlp", "dit_s_2_dit_mlp", "dit_attention",
           "dit_adaln_residual", "dlrm_bottom_mlp", "dlrm_top_mlp", "megafold_transition",
           "megafold_adaptive_layernorm", "megafold_transition_no_ln"}


def nested(model, task, config):
    # evoattention's triangle configurations sit inside triangle_attention_* (L2-3).
    return task == "megafold_evoattention" and "tri_attn" in config


def main():
    rows = [json.loads(line) for line in open(f"{D}/rows.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["model"], r["task"], r["config"])][r["pass"]] = r
    steps = {}
    for m in {k[0] for k in by}:
        if m in EXTERNAL_STEP:
            steps[m] = {"ms": EXTERNAL_STEP[m][0], "regime": "T3 (eager launches)", "src": EXTERNAL_STEP[m][1]}
            continue
        try:
            s = json.load(open(f"{D}/steps/{m}.json"))
        except FileNotFoundError:
            continue
        if m in NOT_CAPTURED or not s.get("t3g_ok"):
            steps[m] = {"ms": s["t3_step_ms"], "regime": "T3 (eager launches)", "src": "measured gh096"}
        else:
            steps[m] = {"ms": s["t3g_step_ms"], "regime": "T3-G (graph replay)", "src": "measured gh096",
                        "t3_ms": s["t3_step_ms"]}
    out = []
    for (m, task, config), passes in sorted(by.items()):
        t, p = passes.get("time", {}), passes.get("profile", {})
        calls = t.get("calls") or p.get("calls")
        wall, lib, nonlib = t.get("wall_ms"), p.get("lib_ms"), p.get("nonlib_ms")
        if m in NOT_CAPTURED:
            opt = None if wall is None or lib is None else max(wall - lib, 0.0)
        else:
            opt = nonlib
        step = steps.get(m)
        share = None if opt is None or not step else 100.0 * opt * calls / step["ms"]
        out.append({"model": m, "task": task, "config": config, "kind": t.get("kind") or p.get("kind"),
                    "calls": calls, "wall_ms": wall, "kernel_ms": p.get("kernel_ms"), "lib_ms": lib,
                    "opt_ms": opt, "time_share_ms": None if wall is None else wall * calls,
                    "share_pct": share, "nested": nested(m, task, config),
                    "step": step, "error": t.get("error") or p.get("error")})
    # E-1: per (model, task), the configuration with the largest eager time (wall x calls), among
    # configurations not nested in another searched L2 case.
    groups = collections.defaultdict(list)
    for r in out:
        groups[(r["model"], r["task"])].append(r)
    for rs in groups.values():
        cand = [r for r in rs if not r["nested"] and r["time_share_ms"] is not None]
        best = max(cand, key=lambda r: r["time_share_ms"]) if cand else None
        for r in rs:
            r["selected"] = r is best
            r["n_configs"] = len(rs)
    # Merged attention blocks.
    merged = []
    for (m, name), parts in MERGED.items():
        rs = [r for r in out if r["model"] == m and r["task"] in parts]
        if len(rs) != len(parts) or any(r["opt_ms"] is None for r in rs):
            merged.append({"model": m, "task": name, "config": "+".join(parts), "share_pct": None,
                           "calls": rs[0]["calls"] if rs else None, "kind": "merged", "selected": True,
                           "n_configs": 1, "nested": False})
            continue
        opt = sum(r["opt_ms"] for r in rs)
        step = steps.get(m)
        merged.append({"model": m, "task": name, "config": "+".join(parts), "kind": "merged",
                       "calls": rs[0]["calls"], "wall_ms": sum(r["wall_ms"] for r in rs),
                       "lib_ms": sum(r["lib_ms"] for r in rs), "opt_ms": opt, "selected": True,
                       "n_configs": 1, "nested": False, "step": step,
                       "share_pct": None if not step else 100.0 * opt * rs[0]["calls"] / step["ms"]})
    out += merged
    for r in out:
        s = r.get("share_pct")
        r["pending"] = r["task"] in PENDING
        r["verdict"] = None if s is None else ("keep" if s >= THRESHOLD else "drop")
    json.dump({"threshold_pct": THRESHOLD, "steps": steps, "rows": out}, open(f"{D}/report.json", "w"), indent=1)
    # Console summary: selected rows.
    print("steps:", {m: (round(v["ms"], 2), v["regime"]) for m, v in sorted(steps.items())})
    print(f"{'model':16} {'task':42} {'cfgs':>4} {'calls':>5} {'wall':>8} {'lib':>8} {'opt':>8} {'share%':>7}  verdict")
    for r in sorted(out, key=lambda r: (r["model"], r["task"])):
        if not r.get("selected"):
            continue
        f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"
        flag = " PENDING" if r["pending"] else ""
        print(f"{r['model']:16} {r['task'][:42]:42} {r['n_configs']:>4} {r['calls']!s:>5} {f(r.get('wall_ms')):>8} "
              f"{f(r.get('lib_ms')):>8} {f(r.get('opt_ms')):>8} {f(r.get('share_pct'), 2):>7}  {r['verdict']}{flag}")


if __name__ == "__main__":
    main()
