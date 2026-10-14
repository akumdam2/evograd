"""Which SDPA policy should evograd run under? Evidence for the cuDNN decision.

Session 1 found two separate cuDNN SDPA problems on this stack (torch 2.11.0+cu128,
GH200): its backward caches a plan per shape without keying on the upstream
gradient's strides (a later call with the other dO layout returns wrong
gradients, rel_l2 ~1.4, silently), and its default backward is nondeterministic.
This probe measures, for every candidate policy, the three things the decision
needs, each in a fresh child process because both the plan cache and the
deterministic-algorithms flag are process state:

* ``mix``   -- correctness under layout mixing, both orders, plus run-to-run
              bitwise repeatability, at each repository model's attention shape;
* ``speed`` -- forward + backward time at those shapes;
* ``l4``    -- Qwen3.5-4B whole-model structural identity (eager vs eager and
              identity vs eager, every parameter gradient) and step time.

Policies: ``cudnn`` (sdpa_kernel([CUDNN_ATTENTION])), ``nocudnn``
(``enable_cudnn_sdp(False)``, default dispatch otherwise), ``flash``
(sdpa_kernel([FLASH_ATTENTION])); each with deterministic algorithms off and on
(``warn_only=False`` -- the only setting that selects cuDNN's deterministic
backward -- with ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).

    python scripts/qwen3_5_4b/probes/sdpa_policy.py --run-dir <dir> [--only mix,speed,l4]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

#: Each repository model's attention shape at its canonical batch, T = 2048.
CONFIGS = {
    "qwen3_5_4b": dict(hq=16, hk=4, d=256, b=1),
    "qwen3_0_6b": dict(hq=16, hk=8, d=128, b=2),
    "llama3_2_1b": dict(hq=32, hk=8, d=64, b=2),
}
POLICIES = ("cudnn", "nocudnn", "flash")
L4_MODES = ("nocudnn", "nocudnn_det", "cudnn_det", "cudnn")


def _env(det: bool) -> dict:
    env = dict(os.environ)
    if det:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _setup(policy: str, det: bool):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if det:
        torch.use_deterministic_algorithms(True)
    if policy.startswith("nocudnn"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if policy.startswith("cudnn"):
        return lambda: sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
    if policy == "flash":
        return lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    return contextlib.nullcontext


def _kernel_names(fn) -> list[str]:
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    keys = ("flash", "fmha", "cudnn", "attention", "attn", "sdpa", "cutlass")
    return sorted({e.name[:120] for e in prof.events()
                   if any(k in e.name.lower() for k in keys)})


def child_mix(config: str, policy: str, det: bool) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ctx = _setup(policy, det)
    c = CONFIGS[config]

    def inputs(tokens: int, seed: int):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(c["b"], c["hq"], tokens, c["d"], generator=gen).to("cuda", torch.bfloat16)
        k = torch.randn(c["b"], c["hk"], tokens, c["d"], generator=gen).to("cuda", torch.bfloat16)
        # v as the model hands it: a head-major view of a token-major buffer.
        v = torch.randn(c["b"], tokens, c["hk"], c["d"], generator=gen).to(
            "cuda", torch.bfloat16).transpose(1, 2)
        g = torch.randn(c["b"], tokens, c["hq"], c["d"], generator=gen).to("cuda", torch.bfloat16)
        return q, k, v, g

    def run(q, k, v, dout, context):
        q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        with context():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out.backward(dout)
        return out.detach(), q.grad, k.grad, v.grad

    def rel(a, b):
        return float((a.double() - b.double()).norm() / b.double().norm())

    result = {"config": config, "policy": policy, "det": det, "torch": torch.__version__,
              "cudnn": torch.backends.cudnn.version(), "shapes": []}
    try:
        for tokens, order in ((2048, "ttct"), (1984, "cctc")):
            q, k, v, g = inputs(tokens, seed=tokens)
            layouts = {"t": g.transpose(1, 2), "c": g.transpose(1, 2).contiguous()}
            ref = run(q.float(), k.float(), v.float(), layouts["c"].float(),
                      lambda: sdpa_kernel([SDPBackend.MATH]))
            calls = []
            for layout in order:
                got = run(q, k, v, layouts[layout], ctx)
                calls.append({"dout": layout, "got": got,
                              "rel": {n: rel(a, b) for n, a, b in
                                      zip(("o", "dq", "dk", "dv"), got, ref)}})
            repeat_bitwise = all(torch.equal(a, b) for a, b in zip(calls[0]["got"], calls[1]["got"]))
            result["shapes"].append({
                "T": tokens, "order": order,
                "calls": [{"dout": x["dout"], "rel": x["rel"]} for x in calls],
                "repeat_bitwise": repeat_bitwise,
                "worst_grad_rel": max(max(x["rel"][n] for n in ("dq", "dk", "dv")) for x in calls),
            })
        q, k, v, g = inputs(2048, seed=7)
        result["kernels"] = _kernel_names(lambda: run(q, k, v, g.transpose(1, 2), ctx))
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def child_speed(policy: str, det: bool) -> dict:
    import torch
    import torch.nn.functional as F
    from triton.testing import do_bench

    ctx = _setup(policy, det)
    out = {"policy": policy, "det": det, "configs": {}}
    for name, c in CONFIGS.items():
        try:
            q = torch.randn(c["b"], c["hq"], 2048, c["d"], device="cuda", dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(c["b"], c["hk"], 2048, c["d"], device="cuda", dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn(c["b"], c["hk"], 2048, c["d"], device="cuda", dtype=torch.bfloat16,
                            requires_grad=True)
            dout = torch.randn(c["b"], 2048, c["hq"], c["d"], device="cuda",
                               dtype=torch.bfloat16).transpose(1, 2)

            def fwd():
                with ctx():
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

            def fwd_bwd():
                fwd().backward(dout)

            out["configs"][name] = {
                "fwd_ms": do_bench(fwd, warmup=25, rep=200),
                "fwd_bwd_ms": do_bench(fwd_bwd, warmup=25, rep=200, grad_to_none=[q, k, v]),
            }
        except Exception as exc:  # noqa: BLE001
            out["configs"][name] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
    return out


def child_l4(mode: str) -> dict:
    import torch

    det = mode.endswith("_det")
    policy = mode.removesuffix("_det")
    ctx = _setup(policy, det)

    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.sites import structural_identity_kernels
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

    workload = Qwen3_5Workload()
    data = workload.batch_for(seed=0)

    def step(model):
        for p in model.parameters():
            p.grad = None
        with ctx():
            loss = workload.loss(model, data)
            loss.backward()
        return loss.detach().clone()

    def snapshot(model, loss):
        return loss, {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    def compare(a, b) -> dict:
        differing, worst = [], (None, 0.0)
        for name in a[1]:
            if torch.equal(a[1][name], b[1][name]):
                continue
            differing.append(name)
            r = float((a[1][name].double() - b[1][name].double()).norm()
                      / b[1][name].double().norm().clamp(min=1e-30))
            if r > worst[1]:
                worst = (name, r)
        return {"loss_bitwise": bool(torch.equal(a[0], b[0])), "params": len(a[1]),
                "params_not_bitwise": len(differing), "worst_param": worst[0],
                "worst_rel_l2": worst[1],
                "all_bitwise": bool(torch.equal(a[0], b[0])) and not differing}

    result = {"mode": mode}
    try:
        model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
        eager = snapshot(model, step(model))
        again = snapshot(model, step(model))
        result["eager_vs_eager"] = compare(again, eager)
        del again
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(model)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        result["fwd_bwd_ms"] = times
        result["kernels"] = _kernel_names(lambda: step(model))
        del model
        torch.cuda.empty_cache()
        model, _ = workload.build_patched(structural_identity_kernels(workload.site_registry))
        identity = snapshot(model, step(model))
        result["identity_vs_eager"] = compare(identity, eager)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"[:800]
    return result


def _spawn(run_dir: Path, key: str, argv: list[str], det: bool, timeout: int) -> dict:
    done = run_dir / f"{key}.json"
    if done.exists():
        return json.loads(done.read_text())
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, __file__, "--child", *argv], env=_env(det),
                          capture_output=True, text=True, timeout=timeout)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        result = {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-2000:]}
    else:
        result = json.loads(lines[-1])
    result["wall_s"] = time.perf_counter() - started
    done.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--only", default="mix,speed,l4")
    parser.add_argument("--child", nargs="*")
    parser.add_argument("--l4-modes", default=",".join(L4_MODES),
                        help="also: default, default_det (the process's own evograd policy)")
    args = parser.parse_args(argv)
    if args.child is not None:
        kind, *rest = args.child
        if kind == "mix":
            out = child_mix(rest[0], rest[1], rest[2] == "1")
        elif kind == "speed":
            out = child_speed(rest[0], rest[1] == "1")
        else:
            out = child_l4(rest[0])
        print(json.dumps(out), flush=True)
        return 0

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    log = lambda msg: print(f"[sdpa_policy {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    if "mix" in only:
        for config in CONFIGS:
            for policy in POLICIES:
                for det in (False, True):
                    r = _spawn(run_dir, f"mix.{config}.{policy}.det{int(det)}",
                               ["mix", config, policy, str(int(det))], det, 900)
                    summary = ([(s["order"], round(s["worst_grad_rel"], 4), s["repeat_bitwise"])
                                for s in r.get("shapes", [])] if r.get("ok") else r.get("error", r.get("stderr", ""))[-200:])
                    log(f"mix {config} {policy} det={int(det)}: {summary}")
    if "speed" in only:
        for policy in POLICIES:
            for det in (False, True):
                r = _spawn(run_dir, f"speed.{policy}.det{int(det)}",
                           ["speed", policy, str(int(det))], det, 900)
                log(f"speed {policy} det={int(det)}: " + json.dumps(
                    {k: v.get("fwd_bwd_ms", v.get("error")) for k, v in r.get("configs", {}).items()}))
    if "l4" in only:
        for mode in args.l4_modes.split(","):
            r = _spawn(run_dir, f"l4.{mode}", ["l4", mode], mode.endswith("_det"), 1800)
            log(f"l4 {mode}: ok={r.get('ok')} " + json.dumps(
                {k: (r[k]["all_bitwise"], r[k]["params_not_bitwise"], r[k]["worst_rel_l2"])
                 for k in ("eager_vs_eager", "identity_vs_eager") if k in r})
                + f" fwd_bwd_ms={[round(t) for t in r.get('fwd_bwd_ms', [])]} {r.get('error', '')[:300]}")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
