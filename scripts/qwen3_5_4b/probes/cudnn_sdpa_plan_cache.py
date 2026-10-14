"""Reproducer: cuDNN SDPA backward reuses a cached plan across upstream-gradient layouts.

``python cudnn_sdpa_plan_cache.py ORDER`` with ORDER a comma list of dout layouts
(``t`` transposed view, ``c`` contiguous), run in sequence in this one process
at Qwen3.5's attention shape; each result is compared with an fp32 math
reference computed on the CPU (so no GPU kernel state can touch it).
"""

import json
import sys

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
HQ, HK, D, T, B = 16, 4, 256, 2048, 1
torch.manual_seed(0)
q = torch.randn(B, HQ, T, D, dtype=torch.bfloat16)
k = torch.randn(B, HK, T, D, dtype=torch.bfloat16)
v = torch.randn(B, T, HK, D, dtype=torch.bfloat16).transpose(1, 2)
g = torch.randn(B, T, HQ, D, dtype=torch.bfloat16)


def grads(q, k, v, dout):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).backward(dout)
    return q.grad, k.grad, v.grad


ref = grads(q.float(), k.float(), v.float(), g.transpose(1, 2).contiguous().float())
qc, kc, vc, gc = (t.cuda() for t in (q, k, v, g))
out = []
for step, layout in enumerate(sys.argv[1].split(",")):
    dout = gc.transpose(1, 2) if layout == "t" else gc.transpose(1, 2).contiguous()
    got = grads(qc, kc, vc, dout)
    rel = {n: float((a.cpu().double() - b.double()).norm() / b.double().norm())
           for n, a, b in zip(("dq", "dk", "dv"), got, ref)}
    out.append({"step": step, "dout": layout, **rel})
    print(json.dumps(out[-1]), flush=True)
