"""In-situ check (2026-11-10): profile one eager training step of a model as Tier 3 builds it,
and report the total kernel time, the library (GEMM/SDPA) part, and the top kernels, to compare
with the sum of the isolated L2 block timings.

Record of a measurement, kept as it ran (its paths are DeltaAI's). Run 2026-11-10 18:25-18:40
CST on one NVIDIA GH200 120GB (DeltaAI), code at 3397ea9 (/u/wzhan/evograd); Qwen3.5-4B by
run_q35.sh (--mem=64g). Outputs: insitu_<model>.json beside this file. A cross-check of the
isolated timings only; write_search_set.py does not read them.

    python insitu.py MODEL OUT.json
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, "/u/wzhan/evograd/candidates")
sys.path.insert(0, "/u/wzhan/.cache/case_rules/l2_share")


def main():
    name, out = sys.argv[1], sys.argv[2]
    import evaluate_model as launcher

    os.environ.update(launcher.environment(name))
    import collections

    import torch
    from torch.profiler import ProfilerActivity, profile

    from evograd.evaluation import protocol as P
    from evograd.evaluation.tier3.runner import OPTIMIZER_DEFAULTS, build_with_provenance, make_training_step
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import configure
    from evograd.verification.integrations.gates import open_run
    from l2_share import is_library

    configure()
    corpus = json.load(open("/work/hdd/bcjw/wzhan/ADBench/cases.json"))[name]["corpus"]
    with tempfile.TemporaryDirectory() as directory:
        run = open_run(load(name), {}, corpus, directory, device="cuda")
        workload = run.workload
        model, _ = build_with_provenance(workload, P.provider_kernels(run, "eager"), provider="eager")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, **OPTIMIZER_DEFAULTS)
        batch = workload.batch_for(seed=0)
        step = make_training_step(workload, model, optimizer, batch)
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        reps = 3
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                step()
            torch.cuda.synchronize()
    per = collections.Counter()
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        if t:
            per[ev.key] += t / 1000.0 / reps
    total = sum(per.values())
    lib = sum(v for k, v in per.items() if is_library(k))
    res = {"model": name, "kernel_ms_per_step": total, "lib_ms": lib, "nonlib_ms": total - lib,
           "top": [[k[:140], round(v, 3), is_library(k)] for k, v in per.most_common(25)]}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("model", "kernel_ms_per_step", "lib_ms", "nonlib_ms")}))
    for row in res["top"][:15]:
        print(row)


if __name__ == "__main__":
    main()
