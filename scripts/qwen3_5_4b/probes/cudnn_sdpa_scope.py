"""Which attention configurations hit the wrong cuDNN SDPA backward (transposed dout)?"""

import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.backends.cuda.matmul.allow_tf32 = False
CONFIGS = {"qwen3_5_4b": (16, 4, 256), "qwen3_0_6b": (16, 8, 128),
           "llama3_2_1b": (32, 8, 64), "mha_d256": (16, 16, 256), "gqa4_d128": (16, 4, 128)}
results = []


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def grads(q, k, v, dout, backend=None, gqa=True):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    if backend is None:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    else:
        with sdpa_kernel(backend):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    o.backward(dout)
    return q.grad, k.grad, v.grad


for name, (hq, hk, d) in CONFIGS.items():
    for batch in (1, 2):
        torch.manual_seed(0)
        T = 2048
        q = torch.randn(batch, hq, T, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, hk, T, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, T, hk, d, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        g = torch.randn(batch, T, hq, d, device="cuda", dtype=torch.bfloat16)
        gqa = hq != hk
        ref = grads(q.float(), k.float(), v.float(), g.transpose(1, 2).float(), SDPBackend.MATH, gqa)
        for layout, dout in (("transposed", g.transpose(1, 2)),
                             ("contiguous", g.transpose(1, 2).contiguous())):
            for label, backend in (("default", None), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                entry = {"config": name, "hq": hq, "hk": hk, "d": d, "batch": batch,
                         "dout": layout, "backend": label}
                try:
                    got = grads(q, k, v, dout, backend, gqa)
                    entry.update({n: rel(a, b) for n, a, b in zip(("dq", "dk", "dv"), got, ref)})
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                results.append(entry)
                print(json.dumps(entry), flush=True)
json.dump({"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
           "results": results}, open(sys.argv[1], "w"), indent=1)
