"""Eager forward+backward time of every configuration of the multi-configuration L1 tasks.

Record of a measurement, kept as it ran (its paths are DeltaAI's). Measured 2026-11-10
02:15-02:17 CST on one NVIDIA GH200 120GB (DeltaAI ghx4-interactive, job 3568146), code from
the evograd-gate1 worktree (the Gate-1-only branch, merged as 3397ea9). Its case list was
/u/wzhan/.cache/case_rules/scratch/all_cases.json (observed_cases of every model then) and
its payloads the corpora named in /work/hdd/bcjw/wzhan/ADBench/cases.json (MegaFold: Ajay's);
neither is committed. Output: l1_timing.json beside this file, read by write_search_set.py
(the L1 time-max rule).

For each (model, task) with more than one observed configuration (GEMM/SDPA tasks excluded),
time the model reference under autograd on one recorded call per configuration (order 303,
first call), and report per-call median, frequency and time share. The single-configuration
pairs need no timing: the one configuration is both the time-max and the count-max case.

    python l1_timing.py --out l1_timing.json [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
import json
import time

VENDOR = {"linear", "linear_no_bias", "causal_gqa_attention"}
CORPORA = {
    "megafold": "/work/hdd/bcjw/akumdam/af3_revamp/corpus/manifest.json",
}
ORDER = 303


def corpus_path(model: str) -> str:
    if model in CORPORA:
        return CORPORA[model]
    return json.load(open("/work/hdd/bcjw/wzhan/ADBench/cases.json"))[model]["corpus"]


NOT_SEARCHED_L2 = {"megafold_layernorm_linear"}


def targets(level=1):
    rows = json.load(open("/u/wzhan/.cache/case_rules/scratch/all_cases.json"))
    skip = VENDOR if level == 1 else NOT_SEARCHED_L2
    groups = collections.defaultdict(list)
    for r in rows:
        if r["level"] == level and r["task"] not in skip:
            groups[(r["model"], r["task"])].append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}


def time_pair(op, values, *, warmup: int, iters: int):
    import torch

    from evograd.evaluation.common.provider import pytorch_autograd_provider
    from evograd.opdecl.inputs import upstream_grad_values

    provider = pytorch_autograd_provider(op)
    dout = upstream_grad_values(op, values)

    def step():
        output, saved = provider.forward(values)
        provider.backward(dout, saved, values)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()

    from evograd.verification.integrations import load
    from evograd.verification.payloads.corpus import Corpus

    results, corpora = [], {}
    for (model_name, task), rows in sorted(targets(args.level).items()):
        model = load(model_name)
        op = model.reference_op(task)
        corpus = corpora.get(model_name) or Corpus.open(corpus_path(model_name))
        corpora[model_name] = corpus
        for r in sorted(rows, key=lambda r: -r["freq"]):
            recs = [x for x in corpus.payloads(case_id=r["case"], split="acceptance")
                    if (x.get("batch") or {}).get("order") == ORDER]
            recs.sort(key=lambda x: (x.get("call") or {}).get("index_in_step", 0))
            row = {"model": model_name, "task": task, "case": r["case"], "dims": r["dims"],
                   "dtype": r["dtype"], "freq": r["freq"], "payload": recs[0]["payload_id"] if recs else None}
            if not recs:
                row["error"] = "no order-303 payload"
            elif not args.dry_run:
                try:
                    values = corpus.load(recs[0], device="cuda")
                    med, lo, hi = time_pair(op, values, warmup=args.warmup, iters=args.iters)
                    row.update(ms=med, ms_min=lo, ms_max=hi, share_ms=med * r["freq"])
                    del values
                except Exception as exc:  # report and continue with the next configuration
                    row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                import torch
                torch.cuda.empty_cache()
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"order": ORDER, "warmup": args.warmup, "iters": args.iters,
               "written": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": results},
              open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
