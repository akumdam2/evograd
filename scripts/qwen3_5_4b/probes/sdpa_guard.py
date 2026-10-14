"""Can cuDNN SDPA be kept, with its plan-cache hazard closed by a layout guard?

``sdpa_policy.py`` showed the cuDNN backward returns wrong gradients when one
process sees two upstream-gradient (dO) layouts at a shape, in either order, and
that disabling cuDNN (flash) is correct but 1.5-1.8x slower. This probe asks
whether the hazard is *only* dO's stride:

* ``inputs`` -- cuDNN with dO fixed and q, k or v alternating between a
  contiguous tensor and the head-major view of a token-major buffer;
* ``guard``  -- cuDNN behind a guard that makes dO contiguous before the
  backward (an identity autograd node after SDPA), dO alternating;
* ``guard_all`` -- the guard plus q, k, v made contiguous, everything alternating;
* ``speed``  -- forward + backward time: cuDNN, cuDNN + guard, cuDNN + guard_all,
  the model's own (transposed) dO layout.

Each case in a fresh child (the plan cache is process state).

    python scripts/qwen3_5_4b/probes/sdpa_guard.py --run-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "qwen3_5_4b": dict(hq=16, hk=4, d=256, b=1),
    "qwen3_0_6b": dict(hq=16, hk=8, d=128, b=2),
    "llama3_2_1b": dict(hq=32, hk=8, d=64, b=2),
}


def _sdpa(mode: str):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    class ContiguousGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return grad.contiguous()

    def call(q, k, v):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            if mode == "guard_all":
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return ContiguousGrad.apply(out) if mode in ("guard", "guard_all") else out

    return call


def child_mix(config: str, mode: str, vary: str) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    c = CONFIGS[config]
    call = _sdpa(mode)

    def tensors(tokens, seed):
        gen = torch.Generator().manual_seed(seed)
        dev = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(c["b"], tokens, c["hq"], c["d"], generator=gen).to(**dev)
        k = torch.randn(c["b"], tokens, c["hk"], c["d"], generator=gen).to(**dev)
        v = torch.randn(c["b"], tokens, c["hk"], c["d"], generator=gen).to(**dev)
        g = torch.randn(c["b"], tokens, c["hq"], c["d"], generator=gen).to(**dev)
        # 't': head-major view of a token-major buffer; 'c': contiguous head-major.
        return {n: {"t": x.transpose(1, 2), "c": x.transpose(1, 2).contiguous()}
                for n, x in (("q", q), ("k", k), ("v", v), ("dout", g))}

    def run(q, k, v, dout, fn):
        q, k, v = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
        fn(q, k, v).backward(dout)
        return q.grad, k.grad, v.grad

    def math(q, k, v):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    out = {"config": config, "mode": mode, "vary": vary, "shapes": []}
    try:
        for tokens, order in ((2048, "tct"), (1984, "ctc")):
            t = tensors(tokens, tokens)
            fixed = {"q": "t", "k": "t", "v": "t", "dout": "t"}
            ref = run(*(t[n]["c"].float() for n in ("q", "k", "v")), t["dout"]["c"].float(), math)
            calls = []
            for layout in order:
                pick = dict(fixed)
                for name in (("q", "k", "v", "dout") if vary == "all" else (vary,)):
                    pick[name] = layout
                got = run(*(t[n][pick[n]] for n in ("q", "k", "v")), t["dout"][pick["dout"]], call)
                calls.append({"layout": layout, "rel": {
                    n: float((a.double() - b.double()).norm() / b.double().norm())
                    for n, a, b in zip(("dq", "dk", "dv"), got, ref)}})
            out["shapes"].append({"T": tokens, "order": order, "calls": calls,
                                  "worst": max(max(x["rel"].values()) for x in calls)})
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["ok"], out["error"] = False, f"{type(exc).__name__}: {exc}"[:500]
    return out


def child_speed() -> dict:
    import torch
    from triton.testing import do_bench

    out = {}
    for name, c in CONFIGS.items():
        dev = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(c["b"], 2048, c["hq"], c["d"], **dev).transpose(1, 2).requires_grad_(True)
        k = torch.randn(c["b"], 2048, c["hk"], c["d"], **dev).transpose(1, 2).requires_grad_(True)
        v = torch.randn(c["b"], 2048, c["hk"], c["d"], **dev).transpose(1, 2).requires_grad_(True)
        dout = torch.randn(c["b"], 2048, c["hq"], c["d"], **dev).transpose(1, 2)
        out[name] = {}
        for mode in ("plain", "guard", "guard_all"):
            call = _sdpa(mode)
            out[name][mode] = do_bench(lambda: call(q, k, v).backward(dout), warmup=25, rep=200,
                                       grad_to_none=[q, k, v])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child", nargs="*")
    args = parser.parse_args(argv)
    if args.child is not None:
        kind, *rest = args.child
        res = child_mix(*rest) if kind == "mix" else child_speed()
        print(json.dumps(res), flush=True)
        return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f"inputs.{cfg}.{v}", ["mix", cfg, "plain", v]) for cfg in CONFIGS for v in ("q", "k", "v")]
    jobs += [(f"guard.{cfg}", ["mix", cfg, "guard", "dout"]) for cfg in CONFIGS]
    jobs += [(f"guard_all.{cfg}", ["mix", cfg, "guard_all", "all"]) for cfg in CONFIGS]
    jobs += [("speed", ["speed"])]
    for key, child in jobs:
        path = args.run_dir / f"{key}.json"
        if not path.exists():
            proc = subprocess.run([sys.executable, __file__, "--child", *child],
                                  capture_output=True, text=True, timeout=900)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            res = json.loads(lines[-1]) if lines else {"ok": False, "stderr": proc.stderr[-1500:]}
            path.write_text(json.dumps(res, indent=2) + "\n")
        res = json.loads(path.read_text())
        summary = (json.dumps({k: {m: round(t, 4) for m, t in v.items()} for k, v in res.items()})
                   if key == "speed" else
                   [(s["order"], round(s["worst"], 4)) for s in res.get("shapes", [])] or res.get("error") or res.get("stderr", "")[-300:])
        print(f"[sdpa_guard {time.strftime('%H:%M:%S')}] {key}: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
