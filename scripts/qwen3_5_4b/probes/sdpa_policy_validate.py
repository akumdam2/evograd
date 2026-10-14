"""Does ``evograd.sdpa_policy`` close the cuDNN hazard where it was found?

    python scripts/qwen3_5_4b/probes/sdpa_policy_validate.py --run-dir <dir>

One child per policy (``cudnn_guarded``, the default, and ``torch_default`` as the
control that must still show the hazard). Each child, in ONE process -- the
situation the hazard needs -- runs at Qwen3.5-4B's observed attention shape:

1. the Level-1 ``causal_gqa_attention`` task (its contiguous cotangent),
   oracle vs production;
2. the Level-2 ``qwen3_5_gated_attention`` task (the transposed cotangent a
   Transformers module hands back), oracle vs production -- session 1's trigger;
3. the Level-1 task again;
4. plain ``F.scaled_dot_product_attention`` with dO alternating t, c, t, c.

Every gradient is compared with its float32 reference; also recorded: which
attention kernels ran and the guard's cost (guarded vs ``__wrapped__``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def child() -> dict:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from triton.testing import do_bench

    from evograd import sdpa_policy
    from evograd.benchmark import TASKS
    from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
    from evograd.opdecl.oracle import oracle, resolve_runtime_forward

    torch.backends.cuda.matmul.allow_tf32 = False

    def task_check(name: str, suite: str) -> dict:
        op = TASKS[name]
        workload = op.benchmark_workloads(suite=suite)[0]
        values = make_case_inputs(op, workload, device="cuda")
        _, ref = oracle(op, values)
        production = resolve_runtime_forward(op)
        active = [a for a in op.args if a.name in {x.name for x in op.active_args()}]
        leaves = {a.name: values[a.name].detach().clone().requires_grad_(True) for a in active}
        args = [leaves.get(a.name, values.get(a.name, getattr(a, "default", None))) for a in op.args]
        outs = as_output_tuple(op, production(*args))
        grads = torch.autograd.grad(outs, list(leaves.values()), upstream_grad_values(op, values))
        return {"task": name, "dims": workload.dims, "rel_l2": {
            a.grad_name: float((g.double() - ref[a.grad_name].double()).norm()
                               / ref[a.grad_name].double().norm()) for a, g in zip(active, grads)}}

    out = {"policy": sdpa_policy.describe(), "steps": []}
    out["steps"].append(task_check("causal_gqa_attention", "qwen3_5_4b_observed"))
    out["steps"].append(task_check("qwen3_5_gated_attention", "qwen3_5_4b_observed"))
    out["steps"].append(task_check("causal_gqa_attention", "qwen3_5_4b_observed"))

    hq, hk, d, b, t = 16, 4, 256, 1, 2048
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(b, t, hq, d, generator=gen).cuda().bfloat16().transpose(1, 2)
    k = torch.randn(b, t, hk, d, generator=gen).cuda().bfloat16().transpose(1, 2)
    v = torch.randn(b, t, hk, d, generator=gen).cuda().bfloat16().transpose(1, 2)
    g = torch.randn(b, t, hq, d, generator=gen).cuda().bfloat16()
    layouts = {"t": g.transpose(1, 2), "c": g.transpose(1, 2).contiguous()}

    def grads(fn, q, k, v, dout):
        q, k, v = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
        fn(q, k, v, is_causal=True, enable_gqa=True).backward(dout)
        return q.grad, k.grad, v.grad

    with sdpa_kernel([SDPBackend.MATH]):
        ref = grads(F.scaled_dot_product_attention.__wrapped__ if sdpa_policy.guard_installed()
                    else F.scaled_dot_product_attention, q.float(), k.float(), v.float(),
                    layouts["c"].float())
    mixed = []
    for layout in "tctc":
        got = grads(F.scaled_dot_product_attention, q, k, v, layouts[layout])
        mixed.append({"dout": layout, "rel_l2": {n: float((a.double() - r.double()).norm()
                                                           / r.double().norm())
                                                  for n, a, r in zip(("dq", "dk", "dv"), got, ref)}})
    out["plain_sdpa_mixed"] = mixed

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        grads(F.scaled_dot_product_attention, q, k, v, layouts["t"])
        torch.cuda.synchronize()
    out["kernels"] = sorted({e.name[:100] for e in prof.events()
                             if any(s in e.name.lower() for s in ("cudnn", "flash", "fmha", "sdpa"))})

    qq, kk, vv = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    raw = getattr(F.scaled_dot_product_attention, "__wrapped__", F.scaled_dot_product_attention)
    out["fwd_bwd_ms"] = {
        name: do_bench(lambda fn=fn: fn(qq, kk, vv, is_causal=True, enable_gqa=True)
                       .backward(layouts["t"]), warmup=25, rep=200, grad_to_none=[qq, kk, vv])
        for name, fn in (("as_installed", F.scaled_dot_product_attention), ("unguarded", raw))}
    worst = max(max(s["rel_l2"].values()) for s in out["steps"])
    worst = max(worst, max(max(m["rel_l2"].values()) for m in mixed))
    out["worst_rel_l2"] = worst
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(child()), flush=True)
        return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for policy in ("cudnn_guarded", "torch_default"):
        env = dict(os.environ, EVOGRAD_SDPA_POLICY=policy)
        proc = subprocess.run([sys.executable, __file__, "--child"], capture_output=True,
                              text=True, env=env, timeout=1200)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        res = json.loads(lines[-1]) if lines else {"error": proc.stderr[-2000:]}
        (args.run_dir / f"validate.{policy}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(f"[validate {time.strftime('%H:%M:%S')}] {policy}: worst rel_l2 "
              f"{res.get('worst_rel_l2')}; steps "
              f"{[(s['task'], round(max(s['rel_l2'].values()), 4)) for s in res.get('steps', [])]}; "
              f"mixed {[round(max(m['rel_l2'].values()), 4) for m in res.get('plain_sdpa_mixed', [])]}; "
              f"ms {res.get('fwd_bwd_ms')}; kernels {res.get('kernels', [])[:3]} "
              f"{res.get('error', '')[-300:]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
