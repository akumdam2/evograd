import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _norm_stats_kernel_fn(q, k, invq, invk,
                          B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                          sqb: tl.constexpr, sqt: tl.constexpr, sqh: tl.constexpr, sqd: tl.constexpr,
                          skb: tl.constexpr, skt: tl.constexpr, skh: tl.constexpr, skd: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tt = (pid // H) % T
    b = pid // (H * T)
    offs = tl.arange(0, BD)
    m = offs < D
    qv = tl.load(q + b * sqb + tt * sqt + h * sqh + offs * sqd, mask=m, other=0.0).to(tl.float32)
    kv = tl.load(k + b * skb + tt * skt + h * skh + offs * skd, mask=m, other=0.0).to(tl.float32)
    tl.store(invq + (b * T + tt) * H + h, tl.rsqrt(tl.sum(qv * qv, axis=0) + 0.000001))
    tl.store(invk + (b * T + tt) * H + h, tl.rsqrt(tl.sum(kv * kv, axis=0) + 0.000001))

_norm_stats_kernel = triton.jit(_norm_stats_kernel_fn)

def _forward_seq_kernel_fn(q, k, v, g, beta, invq, invk, out, schunk,
                           B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                           C: tl.constexpr, NC: tl.constexpr,
                           sqb: tl.constexpr, sqt: tl.constexpr, sqh: tl.constexpr, sqd: tl.constexpr,
                           skb: tl.constexpr, skt: tl.constexpr, skh: tl.constexpr, skd: tl.constexpr,
                           svb: tl.constexpr, svt: tl.constexpr, svh: tl.constexpr, svd: tl.constexpr,
                           sgb: tl.constexpr, sgt: tl.constexpr, sgh: tl.constexpr,
                           sbb: tl.constexpr, sbt: tl.constexpr, sbh: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    offs = tl.arange(0, BD)
    oi = tl.arange(0, BD)
    oj = tl.arange(0, BD)
    mi = oi < D
    mj = oj < D
    m2 = mi[:, None] & mj[None, :]
    S = tl.zeros((BD, BD), tl.float32)
    scale = tl.rsqrt((D + 0.0))
    base_s = ((b * H + h) * (NC + 1)) * D * D
    tl.store(schunk + base_s + oi[:, None] * D + oj[None, :], S, mask=m2)
    for cc in range(0, NC):
        for ii in range(0, C):
            t = cc * C + ii
            valid = t < T
            qv = tl.load(q + b * sqb + t * sqt + h * sqh + offs * sqd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            kv0 = tl.load(k + b * skb + t * skt + h * skh + offs * skd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            vv = tl.load(v + b * svb + t * svt + h * svh + offs * svd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            iq = tl.load(invq + (b * T + t) * H + h, mask=valid, other=0.0).to(tl.float32)
            ik = tl.load(invk + (b * T + t) * H + h, mask=valid, other=0.0).to(tl.float32)
            gv = tl.load(g + b * sgb + t * sgt + h * sgh, mask=valid, other=0.0).to(tl.float32)
            bt = tl.load(beta + b * sbb + t * sbt + h * sbh, mask=valid, other=0.0).to(tl.float32)
            Q = qv * iq * scale
            K = kv0 * ik
            r = tl.exp(gv)
            A = S * r
            w = tl.sum(A * K[:, None], axis=0)
            e = vv - w
            z = bt * e
            S = A + K[:, None] * z[None, :]
            ov = tl.sum(S * Q[:, None], axis=0)
            op = out + ((b * T + t) * H + h) * D + offs
            tl.store(op, ov, mask=(offs < D) & valid)
        base_sn = ((b * H + h) * (NC + 1) + cc + 1) * D * D
        tl.store(schunk + base_sn + oi[:, None] * D + oj[None, :], S, mask=m2)

_forward_seq_kernel = triton.jit(_forward_seq_kernel_fn)

def _backward_seq_kernel_fn(q, k, v, g, beta, invq, invk, schunk, do,
                            dq, dk, dv, dgf, dbeta,
                            B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                            C: tl.constexpr, NC: tl.constexpr,
                            sqb: tl.constexpr, sqt: tl.constexpr, sqh: tl.constexpr, sqd: tl.constexpr,
                            skb: tl.constexpr, skt: tl.constexpr, skh: tl.constexpr, skd: tl.constexpr,
                            svb: tl.constexpr, svt: tl.constexpr, svh: tl.constexpr, svd: tl.constexpr,
                            sgb: tl.constexpr, sgt: tl.constexpr, sgh: tl.constexpr,
                            sbb: tl.constexpr, sbt: tl.constexpr, sbh: tl.constexpr,
                            sdb: tl.constexpr, sdt: tl.constexpr, sdh: tl.constexpr, sdd: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H
    offs = tl.arange(0, BD)
    oi = tl.arange(0, BD)
    oj = tl.arange(0, BD)
    m2 = (oi[:, None] < D) & (oj[None, :] < D)
    scale = tl.rsqrt((D + 0.0))
    P = tl.zeros((BD, BD), tl.float32)
    for rc in range(0, NC):
        cc = NC - 1 - rc
        base_s = ((b * H + h) * (NC + 1) + cc) * D * D
        Sin = tl.load(schunk + base_s + oi[:, None] * D + oj[None, :], mask=m2, other=0.0).to(tl.float32)
        for rr in range(0, C):
            ii = C - 1 - rr
            t = cc * C + ii
            valid = t < T
            Sprev = Sin
            for uu in range(0, C):
                if uu < ii:
                    tu = cc * C + uu
                    uvalid = tu < T
                    ku0 = tl.load(k + b * skb + tu * skt + h * skh + offs * skd, mask=(offs < D) & uvalid, other=0.0).to(tl.float32)
                    vu = tl.load(v + b * svb + tu * svt + h * svh + offs * svd, mask=(offs < D) & uvalid, other=0.0).to(tl.float32)
                    iku = tl.load(invk + (b * T + tu) * H + h, mask=uvalid, other=0.0).to(tl.float32)
                    gu = tl.load(g + b * sgb + tu * sgt + h * sgh, mask=uvalid, other=0.0).to(tl.float32)
                    bu = tl.load(beta + b * sbb + tu * sbt + h * sbh, mask=uvalid, other=0.0).to(tl.float32)
                    Ku = ku0 * iku
                    ru = tl.exp(gu)
                    Au = Sprev * ru
                    wu = tl.sum(Au * Ku[:, None], axis=0)
                    eu = vu - wu
                    zu = bu * eu
                    Sprev = Au + Ku[:, None] * zu[None, :]
            qv = tl.load(q + b * sqb + t * sqt + h * sqh + offs * sqd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            kv0 = tl.load(k + b * skb + t * skt + h * skh + offs * skd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            vv = tl.load(v + b * svb + t * svt + h * svh + offs * svd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            iq = tl.load(invq + (b * T + t) * H + h, mask=valid, other=0.0).to(tl.float32)
            ik = tl.load(invk + (b * T + t) * H + h, mask=valid, other=0.0).to(tl.float32)
            gv = tl.load(g + b * sgb + t * sgt + h * sgh, mask=valid, other=0.0).to(tl.float32)
            bt = tl.load(beta + b * sbb + t * sbt + h * sbh, mask=valid, other=0.0).to(tl.float32)
            dov = tl.load(do + b * sdb + t * sdt + h * sdh + offs * sdd, mask=(offs < D) & valid, other=0.0).to(tl.float32)
            Q = qv * iq * scale
            K = kv0 * ik
            r = tl.exp(gv)
            A = Sprev * r
            w = tl.sum(A * K[:, None], axis=0)
            e = vv - w
            z = bt * e
            Scur = A + K[:, None] * z[None, :]
            G = P + Q[:, None] * dov[None, :]
            dQv = tl.sum(Scur * dov[None, :], axis=1)
            dz = tl.sum(G * K[:, None], axis=0)
            dK1 = tl.sum(G * z[None, :], axis=1)
            dbv = tl.sum(dz * e, axis=0)
            de = bt * dz
            dVv = de
            dK2 = -tl.sum(A * de[None, :], axis=1)
            dKv = dK1 + dK2
            dA = G - K[:, None] * de[None, :]
            dgv = r * tl.sum(tl.sum(dA * Sprev, axis=0), axis=0)
            P = r * dA
            # Apply L2-normalization backward immediately, avoiding three
            # activation-sized fp32 temporaries plus the extra normalization/cast
            # kernel launches.  q/k/v/beta grad tensors are allocated with the
            # same strides as their corresponding inputs, so input strides are
            # also valid output strides here.
            dqdot = tl.sum(dQv * qv, axis=0)
            dkdot = tl.sum(dKv * kv0, axis=0)
            iq3 = iq * iq * iq
            ik3 = ik * ik * ik
            dqv = scale * (iq * dQv - qv * iq3 * dqdot)
            dkv = ik * dKv - kv0 * ik3 * dkdot
            tl.store(dq + b * sqb + t * sqt + h * sqh + offs * sqd, dqv, mask=(offs < D) & valid)
            tl.store(dk + b * skb + t * skt + h * skh + offs * skd, dkv, mask=(offs < D) & valid)
            tl.store(dv + b * svb + t * svt + h * svh + offs * svd, dVv, mask=(offs < D) & valid)
            tl.store(dgf + (b * T + t) * H + h, dgv, mask=valid)
            tl.store(dbeta + b * sbb + t * sbt + h * sbh, dbv, mask=valid)

_backward_seq_kernel = triton.jit(_backward_seq_kernel_fn)



def qwen3_5_gated_delta_rule_forward_with_saved(q, k, v, g, beta):
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not g.is_cuda or not beta.is_cuda:
        raise RuntimeError("all inputs must be CUDA tensors")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g.dim() != 3 or beta.dim() != 3:
        raise RuntimeError("invalid rank")
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("q, k, v shapes must match")
    if g.shape != (B, T, H) or beta.shape != (B, T, H):
        raise RuntimeError("g and beta shapes must be [B,T,H]")
    if D > 128 or D < 1:
        raise RuntimeError("unsupported D")
    if g.dtype is not torch.float32:
        raise RuntimeError("g must be float32")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported q dtype")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("q, k, v dtypes must match")
    if beta.dtype != q.dtype:
        raise RuntimeError("beta dtype must match q dtype")
    BD = triton.next_power_of_2(D)
    if BD < 16:
        BD = 16
    # Smaller checkpoint interval trades more saved chunk states for much less
    # quadratic in-chunk recomputation in backward.  For T=2048,D=128 this cuts
    # the expensive inner replay work by ~8x versus C=64; the memory penalty is
    # modest compared with the latency win in this benchmark.
    C = 8
    NC = triton.cdiv(T, C)
    invq = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    invk = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    o = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
    schunk = torch.empty((B, H, NC + 1, D, D), device=q.device, dtype=torch.float32)
    grid_stats = (B * T * H,)
    _norm_stats_kernel[grid_stats](q, k, invq, invk,
                                   B, T, H, D, BD,
                                   q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                   k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                   num_warps=4)
    grid_fwd = (B * H,)
    _forward_seq_kernel[grid_fwd](q, k, v, g, beta, invq, invk, o, schunk,
                                  B, T, H, D, BD, C, NC,
                                  q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                  k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                  v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                  g.stride(0), g.stride(1), g.stride(2),
                                  beta.stride(0), beta.stride(1), beta.stride(2),
                                  num_warps=8)
    saved_tensors = (q, k, v, g, beta, invq, invk, schunk, D, C, NC)
    return o.contiguous(), saved_tensors

def qwen3_5_gated_delta_rule_backward_from_saved(do, saved_tensors):
    q, k, v, g, beta, invq, invk, schunk, D_saved, C_saved, NC_saved = saved_tensors
    if not do.is_cuda:
        raise RuntimeError("do must be CUDA")
    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    if D != D_saved:
        raise RuntimeError("saved D mismatch")
    if do.shape != q.shape:
        raise RuntimeError("do shape mismatch")
    if do.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported do dtype")
    BD = triton.next_power_of_2(D)
    if BD < 16:
        BD = 16
    C = C_saved
    NC = NC_saved
    dgf = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
    dq = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    dk = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=k.dtype)
    dv = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=v.dtype)
    dbeta = torch.empty_strided(beta.shape, beta.stride(), device=beta.device, dtype=beta.dtype)
    grid_bh = (B * H,)
    _backward_seq_kernel[grid_bh](q, k, v, g, beta, invq, invk, schunk, do,
                                  dq, dk, dv, dgf, dbeta,
                                  B, T, H, D, BD, C, NC,
                                  q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                  k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                  v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                                  g.stride(0), g.stride(1), g.stride(2),
                                  beta.stride(0), beta.stride(1), beta.stride(2),
                                  do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                                  num_warps=8)
    return dq, dk, dv, dgf, dbeta
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_gated_delta_rule', 'arguments': ['q', 'k', 'v', 'g', 'beta'], 'outputs': ['o'], 'upstream_grads': ['do'], 'input_grads': ['dq', 'dk', 'dv', 'dg', 'dbeta'], 'gradient_return_order': ['dq', 'dk', 'dv', 'dg', 'dbeta'], 'scalars': [], 'deployment_entry': 'qwen3_5_gated_delta_rule_deployment', 'allowed_primitives': []}


class Qwen35GatedDeltaRuleFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_gated_delta_rule`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        o, saved = qwen3_5_gated_delta_rule_forward_with_saved(q, k, v, g, beta)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return o

    @staticmethod
    def backward(ctx, do):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        do = do.contiguous() if (do is not None and not do.is_contiguous()) else do
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dq, dk, dv, dg, dbeta = qwen3_5_gated_delta_rule_backward_from_saved(
            do, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, dg, dbeta


def qwen3_5_gated_delta_rule_deployment(q, k, v, g, beta):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if q.dim() > 4:
        _leading = q.shape[:-1]
        q = q.reshape(-1, q.shape[-1])
        k = k.reshape(-1, k.shape[-1])
        v = v.reshape(-1, v.shape[-1])
    o = Qwen35GatedDeltaRuleFunction.apply(q, k, v, g, beta)
    if _leading is not None:
        o = o.view(*_leading, o.shape[-1])
    return o


class Qwen35GatedDeltaRuleModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v, g, beta):
        return qwen3_5_gated_delta_rule_deployment(q, k, v, g, beta)


DEPLOYMENT_ENTRY = "qwen3_5_gated_delta_rule_deployment"
