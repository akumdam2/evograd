"""evograd.sdpa_policy: every process is put under one SDPA policy, and the guard
changes no value -- only which layout SDPA's backward sees."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from evograd import sdpa_policy

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _child(code: str, policy: str | None = None) -> dict:
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop(sdpa_policy.POLICY_ENV, None)
    if policy is not None:
        env[sdpa_policy.POLICY_ENV] = policy
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=300)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


REPORT = ("import json, torch; from evograd import sdpa_policy as p; "
          "print(json.dumps({**p.describe(), 'cudnn_enabled': "
          "torch.backends.cuda.cudnn_sdp_enabled()}))")


class InstallTest(unittest.TestCase):
    def test_evograd_first_guards_when_torch_arrives(self):
        out = _child("import sys, evograd; assert 'torch' not in sys.modules; " + REPORT)
        self.assertEqual(out["policy"], "cudnn_guarded")
        self.assertTrue(out["guard"])
        self.assertTrue(out["cudnn_enabled"])

    def test_torch_first_is_guarded_at_evograd_import(self):
        out = _child("import torch, evograd; " + REPORT)
        self.assertTrue(out["guard"])

    def test_no_cudnn_policy(self):
        out = _child("import evograd; " + REPORT, policy="no_cudnn")
        self.assertEqual(out["policy"], "no_cudnn")
        self.assertFalse(out["guard"])
        self.assertFalse(out["cudnn_enabled"])

    def test_torch_default_touches_nothing(self):
        out = _child("import evograd; " + REPORT, policy="torch_default")
        self.assertFalse(out["guard"])
        self.assertTrue(out["cudnn_enabled"])

    def test_unknown_policy_is_refused(self):
        with self.assertRaises(AssertionError) as ctx:
            _child("import evograd, torch", policy="fastest")
        self.assertIn("EVOGRAD_SDPA_POLICY", str(ctx.exception))

    def test_import_evograd_still_imports_no_torch(self):
        out = _child("import sys, json, evograd; print(json.dumps({'torch': 'torch' in sys.modules}))")
        self.assertFalse(out["torch"])


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(sdpa_policy.guard_installed(), "the test process imported evograd")
        gen = torch.Generator().manual_seed(0)
        self.q = torch.randn(1, 4, 33, 16, generator=gen)
        self.k = torch.randn(1, 2, 33, 16, generator=gen)
        self.v = torch.randn(1, 33, 2, 16, generator=gen).transpose(1, 2)
        self.dout = torch.randn(1, 33, 4, 16, generator=gen).transpose(1, 2)  # not contiguous

    def _run(self, fn, dout):
        q, k, v = (t.clone().requires_grad_(True) for t in (self.q, self.k, self.v))
        out = fn(q, k, v, is_causal=True, enable_gqa=True)
        out.backward(dout)
        return out.detach(), q.grad, k.grad, v.grad

    def test_values_unchanged_and_backward_sees_contiguous_dout(self):
        original = F.scaled_dot_product_attention.__wrapped__
        guarded = self._run(F.scaled_dot_product_attention, self.dout)
        plain = self._run(original, self.dout.contiguous())
        for a, b in zip(guarded, plain):
            self.assertTrue(torch.equal(a, b))

    def test_no_grad_returns_sdpa_output_itself(self):
        with torch.no_grad():
            out = F.scaled_dot_product_attention(self.q, self.k, self.v, enable_gqa=True)
        self.assertIsNone(out.grad_fn)

    def test_compiles_without_a_graph_break(self):
        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
        eager = self._run(lambda q, k, v, **_: fn(q, k, v), self.dout)
        traced = self._run(lambda q, k, v, **_: compiled(q, k, v), self.dout)
        for a, b in zip(eager, traced):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_apply_is_idempotent(self):
        before = F.scaled_dot_product_attention
        sdpa_policy.apply(torch)
        self.assertIs(F.scaled_dot_product_attention, before)


class FingerprintTest(unittest.TestCase):
    def test_tier3_fingerprint_binds_policy_and_cudnn(self):
        from evograd.evaluation.tier3.gate import numerics

        backends = numerics._sdpa_backends()
        self.assertIn("cudnn", backends)
        self.assertEqual(backends["policy"], "cudnn_guarded")
        self.assertTrue(backends["guard"])
        self.assertIn("sdpa_backends", numerics.BINDING_FIELDS)


if __name__ == "__main__":
    unittest.main()
