"""The backward formulas the norm-family statements give, checked against autograd.

Agents implement the backward from ``backward_semantics`` text, so a wrong
formula there is a wrong task. Each test writes one statement's formulas as
code, line for line, and compares them with ``torch.autograd`` of the task's
declared forward on CPU in float64 at a small shape. Weights are drawn far from
one: a learned scale near 1 (what the generated inputs use) hides a misplaced
weight factor, which is how the Qwen3 qkv statement's error went unnoticed.

The forwards upcast to float32 internally (``.float()``), so agreement is at
float32 rounding, not float64; every formula error caught here is O(1).
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark import get_task
from evograd.opdecl.importing import resolve_callable

F64 = torch.float64
RTOL, ATOL = 1e-5, 1e-6


def _randn(*shape):
    return torch.randn(*shape, dtype=F64)


def _autograd(task: str, inputs: dict, upstream: dict) -> dict:
    """``{grad name: gradient}`` of ``task``'s declared forward."""
    op = get_task(task)
    leaves = {a.name: inputs[a.name].clone().requires_grad_(True) for a in op.active_args()}
    args = [leaves.get(a.name, inputs.get(a.name, getattr(a, "default", None))) for a in op.args]
    outputs = resolve_callable(op.forward)(*args)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    grads = torch.autograd.grad(outputs, list(leaves.values()),
                                [upstream[o.grad_name] for o in op.outputs])
    by_arg = {a.grad_name: g for a, g in zip(op.active_args(), grads)}
    return {name: by_arg[name] for name in op.grad_names()}


def _upstream(task: str, inputs: dict) -> dict:
    """One random upstream gradient per output, at the output's shape."""
    op = get_task(task)
    args = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
    with torch.no_grad():
        outputs = resolve_callable(op.forward)(*args)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    return {o.grad_name: _randn(y.shape) for o, y in zip(op.outputs, outputs)}


def _rope_tables(tokens: int, width: int):
    """Real rotary tables, ``[1, T, width]``, halves duplicated."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, width, 2, dtype=F64) / width))
    angles = torch.arange(tokens, dtype=F64)[:, None] * inv[None, :]
    table = torch.cat((angles, angles), -1)[None]
    return table.cos(), table.sin()


def _rotate_half(t):
    half = t.shape[-1] // 2
    return torch.cat((-t[..., half:], t[..., :half]), -1)


def _unrotate(d, cos, sin):
    """``d * cos - rotate_half(d * sin)``, cos/sin unsqueezed at dim 1."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return d * c - _rotate_half(d * s)


def _silu_grad(z):
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


class NormBackwardStatements(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def assertGrads(self, task, inputs, upstream, formula):
        expected = _autograd(task, inputs, upstream)
        self.assertEqual(sorted(formula), sorted(expected))
        for name, value in formula.items():
            with self.subTest(task=task, grad=name):
                torch.testing.assert_close(value, expected[name], rtol=RTOL, atol=ATOL)

    # ── Level 1 ──────────────────────────────────────────────────────────────

    def test_rmsnorm(self):
        x, w = _randn(6, 16), _randn(16)
        eps = get_task("rmsnorm").args[2].default
        up = _upstream("rmsnorm", {"x": x, "weight": w})
        dy = up["dy"]
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        xhat, g = x * rrms, dy * w
        self.assertGrads("rmsnorm", {"x": x, "weight": w}, up, {
            "dx": (g - xhat * (g * xhat).mean(-1, keepdim=True)) * rrms,
            "dweight": (dy * xhat).sum(0)})

    def test_gemma_rmsnorm(self):
        x, w = _randn(6, 16), _randn(16)
        eps = get_task("gemma_rmsnorm").args[2].default
        up = _upstream("gemma_rmsnorm", {"x": x, "weight": w})
        dy = up["dy"]
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        xhat, g = x * r, dy * (1 + w)
        self.assertGrads("gemma_rmsnorm", {"x": x, "weight": w}, up, {
            "dx": r * (g - xhat * (g * xhat).mean(-1, keepdim=True)),
            "dweight": (dy * xhat).sum(0)})

    def test_gated_rmsnorm(self):
        x, z, w = _randn(6, 16), _randn(6, 16), _randn(16)
        eps = get_task("gated_rmsnorm").args[3].default
        inputs = {"x": x, "z": z, "weight": w}
        up = _upstream("gated_rmsnorm", inputs)
        dy = up["dy"]
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        xhat, s = x * r, torch.nn.functional.silu(z)
        n, g = w * xhat, dy * torch.nn.functional.silu(z) * w
        self.assertGrads("gated_rmsnorm", inputs, up, {
            "dx": r * (g - xhat * (g * xhat).mean(-1, keepdim=True)),
            "dz": dy * n * _silu_grad(z),
            "dweight": (dy * s * xhat).sum(0)})

    def test_l2norm(self):
        x = _randn(6, 16)
        eps = get_task("l2norm").args[1].default
        up = _upstream("l2norm", {"x": x})
        dy = up["dy"]
        r = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)
        y = x * r
        self.assertGrads("l2norm", {"x": x}, up, {
            "dx": r * (dy - y * (dy * y).sum(-1, keepdim=True))})

    def test_rope(self):
        x = _randn(2, 3, 5, 8)
        cos, sin = (t[0] for t in _rope_tables(5, 8))
        inputs = {"x": x, "cos": cos, "sin": sin}
        up = _upstream("rope", inputs)
        dy = up["dy"]
        dy1, dy2 = dy[..., :4], dy[..., 4:]
        self.assertGrads("rope", inputs, up, {
            "dx": torch.cat((dy1, dy2), -1) * cos + torch.cat((dy2, -dy1), -1) * sin})

    def test_partial_rope(self):
        x = _randn(2, 3, 5, 8)
        cos, sin = _rope_tables(5, 4)
        inputs = {"x": x, "cos": cos, "sin": sin}
        up = _upstream("partial_rope", inputs)
        dy = up["dy"]
        self.assertGrads("partial_rope", inputs, up, {
            "dx": torch.cat((_unrotate(dy[..., :4], cos, sin), dy[..., 4:]), -1)})

    # ── residual add + norm ─────────────────────────────────────────────────

    def _residual_rmsnorm_rows(self, task):
        """fused_add_rms_norm and llama3_residual_rmsnorm share one statement."""
        x, r, w = _randn(6, 16), _randn(6, 16), _randn(16)
        eps = get_task(task).args[3].default
        inputs = {"x": x, "r": r, "weight": w}
        up = _upstream(task, inputs)
        dout, dsummed = up["dout"], up["dsummed"]
        s = x + r
        rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
        dnorm = dout * w
        dtotal = dsummed + rstd * (dnorm - s * (dnorm * s).mean(-1, keepdim=True) * rstd**2)
        self.assertGrads(task, inputs, up, {
            "dx": dtotal, "dr": dtotal, "dweight": (dout * s * rstd).sum(0)})

    def test_qwen3_fused_add_rms_norm(self):
        self._residual_rmsnorm_rows("fused_add_rms_norm")

    def test_llama3_residual_rmsnorm(self):
        self._residual_rmsnorm_rows("llama3_residual_rmsnorm")

    def _residual_rmsnorm_branch(self, task, scale):
        branch, residual, w = _randn(2, 3, 16), _randn(2, 3, 16), _randn(16)
        eps = get_task(task).args[3].default
        inputs = {"branch": branch, "residual": residual, "weight": w}
        up = _upstream(task, inputs)
        dnormed, dsummed = up["dnormed"], up["dsummed"]
        s = residual + branch
        r = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
        xhat, g = s * r, dnormed * scale(w)
        ds = dsummed + r * (g - xhat * (g * xhat).mean(-1, keepdim=True))
        self.assertGrads(task, inputs, up, {
            "dbranch": ds, "dresidual": ds, "dweight": (dnormed * xhat).sum((0, 1))})

    def test_qwen3_5_residual_rmsnorm(self):
        self._residual_rmsnorm_branch("qwen3_5_residual_rmsnorm", lambda w: 1 + w)

    def test_olmoe_residual_rmsnorm(self):
        self._residual_rmsnorm_branch("olmoe_residual_rmsnorm", lambda w: w)

    def test_qwen3_5_gdn_norm_out(self):
        B, T, HV, D, H = 2, 3, 2, 8, 12
        o, z = _randn(B, T, HV, D), _randn(B, T, HV, D)
        w_norm, w_out = _randn(D), _randn(H, HV * D)
        eps = get_task("qwen3_5_gdn_norm_out").args[4].default
        inputs = {"o": o, "z": z, "w_norm": w_norm, "w_out": w_out}
        up = _upstream("qwen3_5_gdn_norm_out", inputs)
        dy = up["dy"]
        r = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
        xhat, s = o * r, torch.nn.functional.silu(z)
        h = (w_norm * xhat * s).reshape(B, T, HV * D)
        dh = (dy @ w_out).view(B, T, HV, D)
        gn = dh * s * w_norm
        self.assertGrads("qwen3_5_gdn_norm_out", inputs, up, {
            "do": r * (gn - xhat * (gn * xhat).mean(-1, keepdim=True)),
            "dz": dh * w_norm * xhat * _silu_grad(z),
            "dw_norm": (dh * s * xhat).sum((0, 1, 2)),
            "dw_out": torch.einsum("bth,btv->hv", dy, h)})

    # ── q/k/v prefixes ───────────────────────────────────────────────────────

    def test_qwen3_qkv_norm_rope(self):
        B, T, H, HQ, HK, D = 2, 5, 12, 4, 2, 8
        x = _randn(B, T, H)
        wq, wk, wv = _randn(HQ * D, H), _randn(HK * D, H), _randn(HK * D, H)
        nq, nk = _randn(D), _randn(D)
        cos, sin = _rope_tables(T, D)
        eps = get_task("qwen3_qkv_norm_rope").args[-1].default
        inputs = {"x": x, "q_weight": wq, "k_weight": wk, "v_weight": wv,
                  "q_norm_weight": nq, "k_norm_weight": nk, "cos": cos, "sin": sin}
        up = _upstream("qwen3_qkv_norm_rope", inputs)

        def head_norm(d, weight, norm_weight, heads, *, stated):
            h = (x @ weight.T).view(B, T, heads, D)
            r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
            dnorm = _unrotate(d, cos, sin).transpose(1, 2)
            xhat = h * r
            if stated == "corrected":
                g = dnorm * norm_weight
                dh = r * (g - xhat * (g * xhat).mean(-1, keepdim=True))
            else:  # the text this replaced: an extra norm_weight on the projection term
                dh = norm_weight * r * (dnorm - h * (dnorm * norm_weight * h).mean(
                    -1, keepdim=True) * r**2)
            return dh.reshape(B, T, heads * D), (dnorm * xhat).sum((0, 1, 2))

        def formula(stated):
            dq_flat, dnq = head_norm(up["dq"], wq, nq, HQ, stated=stated)
            dk_flat, dnk = head_norm(up["dk"], wk, nk, HK, stated=stated)
            dv_flat = up["dv"].transpose(1, 2).reshape(B, T, HK * D)
            outer = lambda d: torch.einsum("bto,bth->oh", d, x)  # noqa: E731
            return {"dx": dq_flat @ wq + dk_flat @ wk + dv_flat @ wv,
                    "dq_weight": outer(dq_flat), "dk_weight": outer(dk_flat),
                    "dv_weight": outer(dv_flat), "dq_norm_weight": dnq, "dk_norm_weight": dnk}

        self.assertGrads("qwen3_qkv_norm_rope", inputs, up, formula("corrected"))
        # The replaced statement is caught, not merely different in the last bits.
        old, expected = formula("replaced"), _autograd("qwen3_qkv_norm_rope", inputs, up)
        self.assertFalse(torch.allclose(old["dx"], expected["dx"], rtol=1e-2, atol=1e-2))
        text = get_task("qwen3_qkv_norm_rope").backward_semantics
        self.assertIn("sum over B, T and the heads of dnorm * xhat", text)
        self.assertIn("dh = r * (g - xhat * mean(g * xhat, -1))", text)

    def test_llama3_qkv_rope(self):
        B, T, H, HQ, HK, D = 2, 5, 12, 4, 2, 8
        x = _randn(B, T, H)
        wq, wk, wv = _randn(HQ * D, H), _randn(HK * D, H), _randn(HK * D, H)
        cos, sin = _rope_tables(T, D)
        inputs = {"x": x, "q_weight": wq, "k_weight": wk, "v_weight": wv, "cos": cos, "sin": sin}
        up = _upstream("llama3_qkv_rope", inputs)
        flat = lambda d, heads: d.transpose(1, 2).reshape(B, T, heads * D)  # noqa: E731
        dq_flat = flat(_unrotate(up["dq"], cos, sin), HQ)
        dk_flat = flat(_unrotate(up["dk"], cos, sin), HK)
        dv_flat = flat(up["dv"], HK)
        outer = lambda d: torch.einsum("bto,bth->oh", d, x)  # noqa: E731
        self.assertGrads("llama3_qkv_rope", inputs, up, {
            "dx": dq_flat @ wq + dk_flat @ wk + dv_flat @ wv,
            "dq_weight": outer(dq_flat), "dk_weight": outer(dk_flat), "dv_weight": outer(dv_flat)})

    def test_olmoe_qkv_norm_rope(self):
        B, T, H, HQ, HK, D = 2, 5, 12, 4, 2, 8
        x = _randn(B, T, H)
        wq, wk, wv = _randn(HQ * D, H), _randn(HK * D, H), _randn(HK * D, H)
        nq, nk = _randn(HQ * D), _randn(HK * D)
        cos, sin = _rope_tables(T, D)
        eps = get_task("olmoe_qkv_norm_rope").args[-1].default
        inputs = {"x": x, "q_weight": wq, "k_weight": wk, "v_weight": wv,
                  "q_norm_weight": nq, "k_norm_weight": nk, "cos": cos, "sin": sin}
        up = _upstream("olmoe_qkv_norm_rope", inputs)

        def full_width_norm(d, weight, w, heads):
            h = x @ weight.T
            r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
            dnorm = _unrotate(d, cos, sin).transpose(1, 2).reshape(B, T, heads * D)
            xhat, g = h * r, dnorm * w
            return (r * (g - xhat * (g * xhat).mean(-1, keepdim=True)),
                    (dnorm * h * r).sum((0, 1)))

        dqp, dnq = full_width_norm(up["dq"], wq, nq, HQ)
        dkp, dnk = full_width_norm(up["dk"], wk, nk, HK)
        dvp = up["dv"].transpose(1, 2).reshape(B, T, HK * D)
        outer = lambda d: torch.einsum("bto,bth->oh", d, x)  # noqa: E731
        self.assertGrads("olmoe_qkv_norm_rope", inputs, up, {
            "dx": dqp @ wq + dkp @ wk + dvp @ wv, "dq_weight": outer(dqp),
            "dk_weight": outer(dkp), "dv_weight": outer(dvp),
            "dq_norm_weight": dnq, "dk_norm_weight": dnk})

    def test_qwen3_5_qkv_norm_partial_rope(self):
        B, T, H, HQ, HK, D, R = 2, 5, 12, 4, 2, 8, 4
        x = _randn(B, T, H)
        wq, wk, wv = _randn(HQ * 2 * D, H), _randn(HK * D, H), _randn(HK * D, H)
        nq, nk = _randn(D), _randn(D)
        cos, sin = _rope_tables(T, R)
        eps = get_task("qwen3_5_qkv_norm_partial_rope").args[-1].default
        inputs = {"x": x, "q_weight": wq, "k_weight": wk, "v_weight": wv,
                  "q_norm_weight": nq, "k_norm_weight": nk, "cos": cos, "sin": sin}
        up = _upstream("qwen3_5_qkv_norm_partial_rope", inputs)

        def unrotate_first_r(d):
            return torch.cat((_unrotate(d[..., :R], cos, sin), d[..., R:]), -1)

        def head_norm(h, d, w):
            r = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
            dnorm = unrotate_first_r(d).transpose(1, 2)
            xhat, g = h * r, dnorm * (1 + w)
            return r * (g - xhat * (g * xhat).mean(-1, keepdim=True)), (dnorm * xhat).sum((0, 1, 2))

        query = (x @ wq.T).view(B, T, HQ, 2 * D)[..., :D]
        dquery, dnq = head_norm(query, up["dq"], nq)
        dk_heads, dnk = head_norm((x @ wk.T).view(B, T, HK, D), up["dk"], nk)
        # Re-interleave per head: [query | gate] in the [B, T, HQ, 2D] layout.
        dqg = torch.cat((dquery, up["dgate"].view(B, T, HQ, D)), -1).reshape(B, T, HQ * 2 * D)
        dk_flat = dk_heads.reshape(B, T, HK * D)
        dv_flat = up["dv"].transpose(1, 2).reshape(B, T, HK * D)
        outer = lambda d: torch.einsum("bto,bth->oh", d, x)  # noqa: E731
        self.assertGrads("qwen3_5_qkv_norm_partial_rope", inputs, up, {
            "dx": dqg @ wq + dk_flat @ wk + dv_flat @ wv, "dq_weight": outer(dqg),
            "dk_weight": outer(dk_flat), "dv_weight": outer(dv_flat),
            "dq_norm_weight": dnq, "dk_norm_weight": dnk})


if __name__ == "__main__":
    unittest.main()
