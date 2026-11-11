"""The three OLMoE Level-2 tasks, on CPU.

Three claims:

* the declaration is coherent: its production spelling traces under compile, and
  the experts site's kernels agree with its oracle on CPU-sized cases;
* its canonical case is the configuration's (``hf_config`` provenance that
  re-derives) at B2 x T2048;
* the attention block's production spelling is ``OlmoeAttention.forward``'s
  arithmetic: on a tiny model's own attention module, input and rotary tables
  it reproduces the module's output, and the gradients autograd takes through
  it, **bitwise** in float32 and bfloat16. Every site's spelling against every
  call of a live step, the residual site's three wirings included, is the
  real-payload capture's test (``tests/verification/test_integration_olmoe.py``).
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest
from evograd.opdecl.models import OLMOE_1B_7B, rederive_dims
from evograd.opdecl.oracle import resolve_runtime_forward

from tests.olmoe_1b_7b._tiny import BATCH, LAYER, SEQ, TINY_ARCH
from tests._small_cases import SMALL


class TestDeclarations(unittest.TestCase):
    def test_three_sites_and_their_calls_per_step(self):
        self.assertEqual(manifest.SITES, ("attention_block", "residual_rmsnorm", "moe_experts"))
        self.assertEqual(manifest.SITE_TASKS["attention_block"], "olmoe_attention_block")
        self.assertEqual(manifest.SITE_TASKS["moe_experts"], "olmoe_moe_experts")
        self.assertEqual({s: manifest.expected_model_frequency(s) for s in manifest.SITES},
                         {"attention_block": 16, "residual_rmsnorm": 32, "moe_experts": 16})
        self.assertIn("router", manifest.DEFERRED)
        self.assertEqual(sorted(manifest.DEFERRED), ["lm_head_ce", "router"])

    def test_canonical_cases_are_config_derived_at_b2(self):
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            with self.subTest(site=site):
                self.assertEqual(len(op.benchmark), 1)
                case = op.benchmark[0]
                self.assertEqual(case.dtype, "bfloat16")
                self.assertEqual(case.provenance.model, "olmoe_1b_7b")
                self.assertEqual(case.provenance.free, {"batch": 2, "seq": 2048})
                self.assertEqual(case.dims, rederive_dims(case.provenance))
                self.assertEqual(case.dims, manifest.dims_for(site, OLMOE_1B_7B, batch=2,
                                                              seq=2048))
                self.assertEqual(op.coverage, op.benchmark)
                self.assertIn("olmoe_1b_7b_observed", op.benchmark_suites)

    def test_the_experts_contract(self):
        op = TASKS["olmoe_moe_experts"]
        self.assertEqual(op.output_names, ("out",))
        self.assertEqual(op.grad_names(), ("dx", "dgate_up_proj", "ddown_proj", "dtop_k_weights"))
        index = next(a for a in op.args if a.name == "top_k_index")
        self.assertEqual(index.dtype, "int64")
        self.assertEqual(op.benchmark[0].dims,
                         {"T": 4096, "H": 2048, "E": 64, "G": 2048, "I": 1024, "K": 8})

    def test_the_attention_block_contract_is_the_whole_module(self):
        op = TASKS["olmoe_attention_block"]
        shapes = {a.name: a.shape for a in op.args}
        self.assertEqual(shapes["q_norm_weight"], "[QO]")
        self.assertEqual(shapes["k_norm_weight"], "[KVO]")
        self.assertEqual(shapes["o_weight"], "[H, QO]")
        self.assertEqual(op.output_names, ("out",))
        self.assertEqual(op.grad_names(), ("dx", "dq_weight", "dk_weight", "dv_weight",
                                           "dq_norm_weight", "dk_norm_weight", "do_weight"))
        self.assertEqual(op.parameter_args, ("q_weight", "k_weight", "v_weight",
                                             "q_norm_weight", "k_norm_weight", "o_weight"))
        self.assertEqual(op.benchmark[0].dims, {"B": 2, "T": 2048, "H": 2048, "HQ": 16, "HK": 16,
                                                "D": 128, "QO": 2048, "KVO": 2048})
        self.assertNotIn("olmoe_qkv_norm_rope", TASKS)
        self.assertNotIn("olmoe_attention", TASKS)

    def test_the_experts_production_kernels_agree(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.moe_experts.reference import (
            moe_experts_forward_production, moe_experts_forward_ref)
        from evograd.opdecl.inputs import make_case_inputs

        op = TASKS["olmoe_moe_experts"]
        for case in SMALL["olmoe_moe_experts"]:
            values = make_case_inputs(op, case, device="cpu")
            args = [values[a.name] for a in op.args]
            with self.subTest(dims=case.dims, dtype=case.dtype):
                grouped = moe_experts_forward_production(*args)
                eager = moe_experts_forward_production(*args, experts_implementation="eager")
                oracle = moe_experts_forward_ref(*args)
                tol = 1e-5 if case.dtype == "float32" else 5e-2
                self.assertTrue(torch.allclose(grouped.float(), eager.float(), atol=tol, rtol=tol))
                self.assertTrue(torch.allclose(grouped.float(), oracle.float(), atol=tol, rtol=tol))


def _tiny_model(dtype: str):
    """A tiny OLMoE built as the workload builds it, q/k norm weights moved off one."""
    from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.model import build_model
    from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.spec import CANONICAL

    model = build_model(CANONICAL.replace(arch=TINY_ARCH, device="cpu", seq_len=SEQ,
                                          batch_size=BATCH, dtype=dtype))
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=generator))
    return model


class TestAttentionBlockIsTheModule(unittest.TestCase):
    ARGS = ("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight", "o_weight")

    def _call(self, dtype):
        model = _tiny_model(dtype)
        attn = model.model.layers[LAYER].self_attn
        generator = torch.Generator().manual_seed(5)
        x = torch.randn((BATCH, SEQ, TINY_ARCH["hidden_size"]), generator=generator).to(
            getattr(torch, dtype))
        cos, sin = model.model.rotary_emb(x, torch.arange(SEQ)[None])
        weights = {"q_weight": attn.q_proj.weight, "k_weight": attn.k_proj.weight,
                   "v_weight": attn.v_proj.weight, "q_norm_weight": attn.q_norm.weight,
                   "k_norm_weight": attn.k_norm.weight, "o_weight": attn.o_proj.weight}
        return attn, x, cos, sin, weights

    def _grads(self, out, x, weights, dout):
        return torch.autograd.grad(out, [x, *weights.values()], dout)

    def test_production_reproduces_the_module_bitwise(self):
        production = resolve_runtime_forward(TASKS["olmoe_attention_block"])
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                attn, x, cos, sin, weights = self._call(dtype)
                self.assertEqual(tuple(cos.shape), (1, SEQ, TINY_ARCH["hidden_size"] // 4))
                x = x.requires_grad_(True)
                native, _ = attn(hidden_states=x, position_embeddings=(cos, sin),
                                 attention_mask=None)
                spelled = production(x, *weights.values(), cos, sin,
                                     float(attn.q_norm.variance_epsilon))
                self.assertTrue(torch.equal(spelled, native))
                self.assertEqual(spelled.stride(), native.stride())
                dout = torch.randn(native.shape, generator=torch.Generator().manual_seed(6)).to(
                    native.dtype)
                for name, a, b in zip(("dx", *self.ARGS), self._grads(native, x, weights, dout),
                                      self._grads(spelled, x, weights, dout)):
                    self.assertTrue(torch.equal(a, b), name)

    def test_a_grouped_call_is_refused(self):
        production = resolve_runtime_forward(TASKS["olmoe_attention_block"])
        _attn, x, cos, sin, weights = self._call("float32")
        half = weights["k_weight"].shape[0] // 2
        grouped = dict(weights, k_weight=weights["k_weight"][:half],
                       v_weight=weights["v_weight"][:half],
                       k_norm_weight=weights["k_norm_weight"][:half])
        with self.assertRaisesRegex(ValueError, "multi-head"):
            production(x, *grouped.values(), cos, sin)

    def test_the_oracle_agrees_in_float32(self):
        op = TASKS["olmoe_attention_block"]
        attn, x, cos, sin, weights = self._call("float32")
        from evograd.opdecl.importing import resolve_callable

        oracle = resolve_callable(op.forward)(x, *weights.values(), cos, sin)
        native, _ = attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=None)
        self.assertTrue(torch.allclose(oracle, native, atol=1e-5, rtol=1e-4))


class TestProductionCompiles(unittest.TestCase):
    """Every production spelling traces under ``torch.compile`` (Dynamo's
    ``eager`` backend, no toolchain needed): the Tier-2 compile baseline, the
    Tier-3 per-site compile provider and the trusted-compile control all
    compile them. The experts site traces whole too: its grouped GEMM goes
    through Transformers' custom-op fallback under Dynamo, and the shim the
    production spelling carries is a plain class Dynamo can read."""

    def test_compiles(self):
        from evograd.opdecl.inputs import as_output_tuple, make_case_inputs

        for task in manifest.TASKS:
            with self.subTest(task=task):
                op = TASKS[task]
                workload = SMALL[task][0]
                values = make_case_inputs(op, workload, device="cpu")
                args = [values.get(a.name, getattr(a, "default", None)) for a in op.args]
                production = resolve_runtime_forward(op)
                torch._dynamo.reset()
                compiled = torch.compile(production, backend="eager", fullgraph=True,
                                         dynamic=False)
                for got, want in zip(as_output_tuple(op, compiled(*args)),
                                     as_output_tuple(op, production(*args))):
                    self.assertTrue(torch.allclose(got.float(), want.float(), atol=1e-6,
                                                   rtol=1e-6))
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
