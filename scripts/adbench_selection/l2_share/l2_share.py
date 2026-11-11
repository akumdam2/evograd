"""L2-5 measurement (2026-11-10): the optimizable share of every Level-2 case.

Record of a measurement, kept as it ran (its paths are DeltaAI's). Run 2026-11-10 17:56-18:27
CST on gh096, one NVIDIA GH200 120GB (900 W), code at 3397ea9 (/u/wzhan/evograd), by run.sh
(time pass, then profile pass). Its rows (rows.jsonl, not committed) are folded by report.py
into report.json beside this file, which write_search_set.py reads (the L2 rules).

Rows come in two kinds:
  corpus    -- an existing L2 task, run through its model reference (autograd provider) on
               the first order-303 recorded call of each configuration;
  synthetic -- a proposed L2 block that has no task yet, run through a plain eager spelling
               of the model's code on random inputs of the recorded shapes.

Pass "time": wall time of one forward+backward call (CUDA events, median of --iters).
Pass "profile" (run in a separate process, after "time", because the profiler's CUPTI state
disturbs later timing): kernel time of one call, split into library kernels (GEMM, SDPA:
the vendor primitives a candidate may call) and the rest.

Rows append to a JSONL file; a row already present is skipped, so an interrupted run resumes.

    python l2_share.py --pass time --out rows.jsonl
    python l2_share.py --pass profile --out rows.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time

ALL_CASES = "/u/wzhan/.cache/case_rules/scratch/all_cases.json"
CASES_JSON = "/work/hdd/bcjw/wzhan/ADBench/cases.json"
CORPORA = {"megafold": "/work/hdd/bcjw/akumdam/af3_revamp/corpus/manifest.json"}
ORDER = 303
SKIP_L2 = {"megafold_layernorm_linear"}  # removed from the benchmark on 2026-11-10

# Library kernels: GEMM (cuBLAS/cuBLASLt/CUTLASS) and SDPA (flash, memory-efficient, cuDNN).
# Convolutions are not a granted primitive, so a cuDNN conv kernel counts as optimizable.
LIB = re.compile(r"gemm|gemv|xmma|cutlass|nvjet|cublas|splitk|flash|fmha|sdpa|efficient_attention", re.I)
CONV = re.compile(r"convolution|conv[123_]?d|cudnn.*conv|fprop|dgrad|wgrad|depthwise|winograd|implicit_?gemm",
                  re.I)


def is_library(name: str) -> bool:
    return bool(LIB.search(name)) and not CONV.search(name)


def corpus_path(model: str) -> str:
    if model in CORPORA:
        return CORPORA[model]
    return json.load(open(CASES_JSON))[model]["corpus"]


# --------------------------------------------------------------------------- synthetic blocks


def synthetic_blocks():
    """(model, name, config label, calls per step, builder) for every proposed block.

    A builder returns (fn, inputs, grad_inputs): fn(*inputs) gives a tensor or a tuple; the
    backward differentiates with respect to grad_inputs.
    """
    import torch
    import torch.nn.functional as F

    dev = "cuda"

    def p(*shape, dtype, grad=True, scale=0.02):
        t = (torch.randn(*shape, device=dev, dtype=torch.float32) * scale).to(dtype)
        return t.requires_grad_(grad)

    def a(*shape, dtype, grad=True):
        return p(*shape, dtype=dtype, grad=grad, scale=1.0)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    blocks = []

    # DiT-S/2 attention: timm Attention (qkv with bias, SDPA, proj with bias), float32.
    def dit_attention():
        x = a(256, 256, 384, dtype=torch.float32)
        wq, bq = p(1152, 384, dtype=torch.float32), p(1152, dtype=torch.float32)
        wp, bp = p(384, 384, dtype=torch.float32), p(384, dtype=torch.float32)

        def fn(x, wq, bq, wp, bp):
            B, N, C = x.shape
            qkv = F.linear(x, wq, bq).reshape(B, N, 3, 6, 64).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            o = F.scaled_dot_product_attention(q, k, v)
            return F.linear(o.transpose(1, 2).reshape(B, N, C), wp, bp)
        ins = [x, wq, bq, wp, bp]
        return fn, ins, ins
    blocks.append(("dit_s_2", "dit_attention", "x[256,256,384] fp32", 12, dit_attention))

    # DiT-S/2 gated residual + affine-free LayerNorm + modulation, float32.
    def dit_adaln_residual():
        x, br = a(256, 256, 384, dtype=torch.float32), a(256, 256, 384, dtype=torch.float32)
        shift, scale, gate = (a(256, 384, dtype=torch.float32) for _ in range(3))

        def fn(x, br, shift, scale, gate):
            s = x + gate.unsqueeze(1) * br
            n = F.layer_norm(s, (384,), eps=1e-6)
            return n * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), s
        ins = [x, br, shift, scale, gate]
        return fn, ins, ins
    blocks.append(("dit_s_2", "dit_adaln_residual", "x[256,256,384] fp32", 24, dit_adaln_residual))

    # DLRM dense (bottom) and over (top) MLPs: linear + ReLU per layer, float32, batch 32.
    def mlp(sizes, last_act):
        def build():
            x = a(32, sizes[0], dtype=torch.float32)
            ws = []
            for i in range(len(sizes) - 1):
                ws += [p(sizes[i + 1], sizes[i], dtype=torch.float32), p(sizes[i + 1], dtype=torch.float32)]

            def fn(x, *ws):
                h = x
                n = len(ws) // 2
                for i in range(n):
                    h = F.linear(h, ws[2 * i], ws[2 * i + 1])
                    if i < n - 1 or last_act:
                        h = torch.relu(h)
                return h
            ins = [x, *ws]
            return fn, ins, ins
        return build
    blocks.append(("torchrec_dlrm", "dlrm_bottom_mlp", "13-512-256-64", 1, mlp([13, 512, 256, 64], True)))
    # Over arch: three linear+ReLU layers, then the final 256->1 linear without activation.
    blocks.append(("torchrec_dlrm", "dlrm_top_mlp", "415-512-512-256-1", 1, mlp([415, 512, 512, 256, 1], False)))

    # ConvNeXt V2-Tiny: 7x7 depthwise conv (bias) -> NHWC -> LayerNorm(eps 1e-6), bfloat16.
    def dwconv_ln(C, H):
        def build():
            # The model runs channels_last (cuDNN NHWC depthwise kernels, checked in situ 11-10).
            x = a(32, C, H, H, dtype=torch.bfloat16).detach().to(memory_format=torch.channels_last).requires_grad_()
            w = p(C, 1, 7, 7, dtype=torch.bfloat16).detach().to(memory_format=torch.channels_last).requires_grad_()
            b = p(C, dtype=torch.bfloat16)
            lw, lb = p(C, dtype=torch.bfloat16), p(C, dtype=torch.bfloat16)

            def fn(x, w, b, lw, lb):
                y = F.conv2d(x, w, b, padding=3, groups=C).permute(0, 2, 3, 1)
                return F.layer_norm(y, (C,), lw, lb, 1e-6)
            ins = [x, w, b, lw, lb]
            return fn, ins, ins
        return build
    for C, H, calls in ((96, 56, 3), (192, 28, 3), (384, 14, 9), (768, 7, 3)):
        blocks.append(("convnextv2_tiny", "dwconv_layernorm", f"C{C} {H}x{H}", calls, dwconv_ln(C, H)))

    # Qwen2-VL-2B vision attention: qkv with bias, 2-D RoPE in float32, non-causal SDPA, proj.
    def vision_attention():
        x = a(256, 1280, dtype=torch.bfloat16)
        wq, bq = p(3840, 1280, dtype=torch.bfloat16), p(3840, dtype=torch.bfloat16)
        wp, bp = p(1280, 1280, dtype=torch.bfloat16), p(1280, dtype=torch.bfloat16)
        ang = torch.randn(256, 40, device=dev)
        emb = torch.cat((ang, ang), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def fn(x, wq, bq, wp, bp):
            S = x.shape[0]
            q, k, v = F.linear(x, wq, bq).reshape(S, 3, 16, 80).permute(1, 0, 2, 3).unbind(0)
            c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
            qf, kf = q.float(), k.float()
            q = (qf * c + rotate_half(qf) * s).to(x.dtype)
            k = (kf * c + rotate_half(kf) * s).to(x.dtype)
            q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.squeeze(0).transpose(0, 1).reshape(S, -1)
            return F.linear(o, wp, bp)
        ins = [x, wq, bq, wp, bp]
        return fn, ins, ins
    blocks.append(("qwen2_vl_2b", "vision_attention", "x[256,1280] bf16", 32, vision_attention))

    # Qwen2-VL-2B vision residual + LayerNorm (affine, eps 1e-6).
    def vision_res_ln():
        x, r = a(256, 1280, dtype=torch.bfloat16), a(256, 1280, dtype=torch.bfloat16)
        w, b = p(1280, dtype=torch.bfloat16), p(1280, dtype=torch.bfloat16)

        def fn(x, r, w, b):
            s = r + x
            return F.layer_norm(s, (1280,), w, b, 1e-6), s
        ins = [x, r, w, b]
        return fn, ins, ins
    blocks.append(("qwen2_vl_2b", "vision_residual_layernorm", "[256,1280] bf16", 64, vision_res_ln))

    # Qwen2-VL-2B text attention: q/k/v with bias, M-RoPE, causal GQA SDPA (repeat_kv), o_proj.
    def text_attention():
        x = a(1, 512, 1536, dtype=torch.bfloat16)
        wq, bq = p(1536, 1536, dtype=torch.bfloat16), p(1536, dtype=torch.bfloat16)
        wk, bk = p(256, 1536, dtype=torch.bfloat16), p(256, dtype=torch.bfloat16)
        wv, bv = p(256, 1536, dtype=torch.bfloat16), p(256, dtype=torch.bfloat16)
        wo = p(1536, 1536, dtype=torch.bfloat16)
        ang = torch.randn(1, 512, 64, device=dev)
        emb = torch.cat((ang, ang), dim=-1)
        cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        mask = torch.ones(512, 512, device=dev, dtype=torch.bool).tril()

        def fn(x, wq, bq, wk, bk, wv, bv, wo):
            B, T, _ = x.shape
            q = F.linear(x, wq, bq).view(B, T, 12, 128).transpose(1, 2)
            k = F.linear(x, wk, bk).view(B, T, 2, 128).transpose(1, 2)
            v = F.linear(x, wv, bv).view(B, T, 2, 128).transpose(1, 2)
            c, s = cos.unsqueeze(1), sin.unsqueeze(1)
            q = q * c + rotate_half(q) * s
            k = k * c + rotate_half(k) * s
            k = k[:, :, None].expand(B, 2, 6, T, 128).reshape(B, 12, T, 128)
            v = v[:, :, None].expand(B, 2, 6, T, 128).reshape(B, 12, T, 128)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return F.linear(o.transpose(1, 2).reshape(B, T, -1), wo)
        ins = [x, wq, bq, wk, bk, wv, bv, wo]
        return fn, ins, ins
    blocks.append(("qwen2_vl_2b", "text_attention_block", "x[1,512,1536] bf16", 28, text_attention))

    # Qwen2-VL-2B text residual + Qwen2RMSNorm (float32 variance, cast back, weight).
    def text_res_rms():
        x, r = a(512, 1536, dtype=torch.bfloat16), a(512, 1536, dtype=torch.bfloat16)
        w = p(1536, dtype=torch.bfloat16)

        def fn(x, r, w):
            s = r + x
            h = s.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
            return w * h.to(s.dtype), s
        ins = [x, r, w]
        return fn, ins, ins
    blocks.append(("qwen2_vl_2b", "residual_rmsnorm_text", "[512,1536] bf16", 56, text_res_rms))

    # Qwen3-0.6B lm_head + cross entropy (tied weight, bf16 logits, float32 CE).
    def flce():
        x = a(4096, 1024, dtype=torch.bfloat16)
        w = p(151936, 1024, dtype=torch.bfloat16)
        t = torch.randint(0, 151936, (4096,), device=dev)
        t[::97] = -100

        def fn(x, w):
            return F.cross_entropy(F.linear(x, w).float(), t, ignore_index=-100)
        return fn, [x, w], [x, w]
    blocks.append(("qwen3_0_6b", "lm_head_cross_entropy", "x[4096,1024] V151936", 1, flce))

    # MegaFold AdaptiveLayerNorm (diffusion, float32): LN(x) no affine, LN(cond) weight only,
    # gamma = sigmoid(linear(cond)), beta = linear_no_bias(cond), gamma * x + beta.
    def adaln(rows, d, dc):
        def build():
            x, c = a(rows, d, dtype=torch.float32), a(rows, dc, dtype=torch.float32)
            wc = p(dc, dtype=torch.float32)
            wg, bg = p(d, dc, dtype=torch.float32), p(d, dtype=torch.float32)
            wb = p(d, dc, dtype=torch.float32)

            def fn(x, c, wc, wg, bg, wb):
                xn = F.layer_norm(x, (d,))
                cn = F.layer_norm(c, (dc,), wc, None)
                return torch.sigmoid(F.linear(cn, wg, bg)) * xn + F.linear(cn, wb)
            ins = [x, c, wc, wg, bg, wb]
            return fn, ins, ins
        return build
    blocks.append(("megafold", "megafold_adaptive_layernorm", "atom_encoder [11764,128]", 144, adaln(11764, 128, 128)))
    blocks.append(("megafold", "megafold_adaptive_layernorm", "token_transformer [1536,768] cond 801", 576, adaln(1536, 768, 801)))

    # MegaFold Transition without its LayerNorm: linear_no_bias K->2I, SwiGLU, linear_no_bias I->K.
    def transition_no_ln(rows, k, i):
        def build():
            x = a(rows, k, dtype=torch.float32)
            w1, w2 = p(2 * i, k, dtype=torch.float32), p(k, i, dtype=torch.float32)

            def fn(x, w1, w2):
                h, g = F.linear(x, w1).chunk(2, dim=-1)
                return F.linear(F.silu(g) * h, w2)
            ins = [x, w1, w2]
            return fn, ins, ins
        return build
    blocks.append(("megafold", "megafold_transition_no_ln", "atom_encoder [11764,128] I256", 72, transition_no_ln(11764, 128, 256)))
    blocks.append(("megafold", "megafold_transition_no_ln", "token_transformer [1536,768] I1536", 288, transition_no_ln(1536, 768, 1536)))
    return blocks


# --------------------------------------------------------------------------- measurement


def make_synthetic_step(build):
    import torch

    torch.manual_seed(0)
    fn, ins, grads_of = build()
    out = fn(*ins)
    outs = out if isinstance(out, tuple) else (out,)
    gouts = [torch.randn_like(o) for o in outs]
    del out, outs

    def step():
        o = fn(*ins)
        o = o if isinstance(o, tuple) else (o,)
        torch.autograd.grad(o, grads_of, gouts)
    return step


def make_corpus_step(model_name, task, case, corpora, models):
    from evograd.evaluation.common.provider import pytorch_autograd_provider
    from evograd.opdecl.inputs import upstream_grad_values
    from evograd.verification.integrations import load
    from evograd.verification.payloads.corpus import Corpus

    model = models.get(model_name) or load(model_name)
    models[model_name] = model
    op = model.reference_op(task)
    corpus = corpora.get(model_name) or Corpus.open(corpus_path(model_name))
    corpora[model_name] = corpus
    recs = [x for x in corpus.payloads(case_id=case, split="acceptance")
            if (x.get("batch") or {}).get("order") == ORDER]
    if not recs:
        raise LookupError("no order-303 payload")
    recs.sort(key=lambda x: (x.get("call") or {}).get("index_in_step", 0))
    values = corpus.load(recs[0], device="cuda")
    provider = pytorch_autograd_provider(op)
    dout = upstream_grad_values(op, values)

    def step():
        output, saved = provider.forward(values)
        provider.backward(dout, saved, values)
    return step, recs[0]["payload_id"]


def wall_ms(step, warmup, iters):
    import torch

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def kernel_split(step, warmup, reps=3):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
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
            per[ev.key] += t / 1000.0 / reps  # us -> ms per call
    total = sum(per.values())
    lib = sum(v for k, v in per.items() if is_library(k))
    top = [[k[:120], round(v, 4), is_library(k)] for k, v in per.most_common(10)]
    return {"kernel_ms": total, "lib_ms": lib, "nonlib_ms": total - lib, "top_kernels": top,
            "kernels": {k: v for k, v in per.items()}}


def targets():
    rows = json.load(open(ALL_CASES))
    out = []
    for r in rows:
        if r["level"] == 2 and r["task"] not in SKIP_L2:
            out.append({"kind": "corpus", "model": r["model"], "task": r["task"], "config": r["case"],
                        "dims": r["dims"], "dtype": r["dtype"], "calls": r["freq"]})
    for model, name, label, calls, _ in synthetic_blocks_meta():
        out.append({"kind": "synthetic", "model": model, "task": name, "config": label, "calls": calls})
    # Small models first, MegaFold (large payloads) last.
    order = {"megafold": 2}
    out.sort(key=lambda t: order.get(t["model"], 0))
    return out


_META = None


def synthetic_blocks_meta():
    global _META
    if _META is None:
        _META = synthetic_blocks()
    return _META


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="which", choices=("time", "profile"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--only-model")
    args = parser.parse_args()

    import torch
    from evograd.verification.integrations.base import configure

    configure()
    done = set()
    try:
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["pass"], r["kind"], r["model"], r["task"], r["config"]))
    except FileNotFoundError:
        pass
    builders = {(m, n, c): b for m, n, c, _, b in synthetic_blocks_meta()}
    corpora, models = {}, {}
    for t in targets():
        if args.only_model and t["model"] != args.only_model:
            continue
        key = (args.which, t["kind"], t["model"], t["task"], t["config"])
        if key in done:
            continue
        row = {"pass": args.which, **t, "written": time.strftime("%H:%M:%S")}
        try:
            if t["kind"] == "corpus":
                step, row["payload"] = make_corpus_step(t["model"], t["task"], t["config"], corpora, models)
            else:
                step = make_synthetic_step(builders[(t["model"], t["task"], t["config"])])
            if args.which == "time":
                med, lo, hi = wall_ms(step, args.warmup, args.iters)
                row.update(wall_ms=med, wall_min=lo, wall_max=hi)
            else:
                row.update(kernel_split(step, args.warmup))
        except Exception as exc:  # record and go on with the next row
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        step = None
        torch.cuda.empty_cache()
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("pass", "model", "task", "config", "wall_ms",
                                                   "kernel_ms", "lib_ms", "nonlib_ms", "error")}),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
