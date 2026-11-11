"""The six Qwen3.5 Level-2 tasks, on CPU.

* the sites and their calls per step, by block kind;
* every canonical case is the configuration's (``hf_config`` provenance that
  re-derives) at B1 x T2048;
* every production spelling traces whole under compile on a CPU-sized case;
* the attention block's runtime spelling is ``Qwen3_5Attention.forward`` bit
  for bit, at a batch of two, from the module's own inputs and one shared row
  of the rotary tables. The whole-model capture
  (``tests/verification/test_integration_qwen3_5.py``) and the Tier-3 adapters
  (``test_tier3_model.py``) hold every task to the live model the same way.
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest
from evograd.opdecl.models import QWEN3_5_4B, rederive_dims
from evograd.opdecl.oracle import resolve_runtime_forward

from tests.qwen3_5_4b._tiny import FULL_LAYER, SEQ, TINY_ARCH
from tests._small_cases import SMALL


class TestDeclarations(unittest.TestCase):
    def test_six_sites_in_two_block_kinds(self):
        self.assertEqual(len(manifest.SITES), 6)
        self.assertEqual(set(manifest.SITES_BY_BLOCK["linear_attention"]),
                         {"gdn_in_proj", "gated_delta_rule", "gdn_norm_out",
                          "residual_rmsnorm", "swiglu_mlp"})
        self.assertEqual(set(manifest.SITES_BY_BLOCK["full_attention"]),
                         {"attention_block", "residual_rmsnorm", "swiglu_mlp"})
        # Two residual fusions per layer: the attention add and the closing MLP add.
        self.assertEqual({s: manifest.expected_model_frequency(s) for s in manifest.SITES},
                         {"gdn_in_proj": 24, "gated_delta_rule": 24, "gdn_norm_out": 24,
                          "attention_block": 8, "residual_rmsnorm": 64, "swiglu_mlp": 32})

    def test_canonical_cases_are_config_derived_at_b1(self):
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            with self.subTest(site=site):
                self.assertEqual(len(op.benchmark), 1)
                case = op.benchmark[0]
                self.assertEqual(case.dtype, "bfloat16")
                self.assertEqual(case.provenance.model, "qwen3_5_4b")
                self.assertEqual(case.provenance.free, {"batch": 1, "seq": 2048})
                self.assertEqual(case.dims, rederive_dims(case.provenance))
                self.assertEqual(case.dims, manifest.dims_for(site, QWEN3_5_4B, batch=1,
                                                              seq=2048))
                self.assertEqual(op.coverage, op.benchmark)
                self.assertIn("qwen3_5_4b_observed", op.benchmark_suites)

    def test_the_attention_block_case(self):
        self.assertEqual(TASKS["qwen3_5_attention_block"].benchmark[0].dims,
                         {"B": 1, "T": 2048, "H": 2560, "HQ": 16, "HK": 4, "D": 256, "R": 64,
                          "QG": 8192, "KVO": 1024, "QO": 4096})

    def test_library_baselines_are_declared(self):
        self.assertIn("fla", TASKS["qwen3_5_gated_delta_rule"].performance_baselines)
        self.assertIn("fla", TASKS["qwen3_5_gdn_norm_out"].performance_baselines)
        self.assertFalse(TASKS["qwen3_5_attention_block"].performance_baselines)
        self.assertFalse(TASKS["qwen3_5_residual_rmsnorm"].performance_baselines)
        self.assertFalse(TASKS["qwen3_5_swiglu_mlp"].performance_baselines)


class TestAttentionBlockIsTheModule(unittest.TestCase):
    def test_runtime_spelling_is_bitwise_the_attention_module(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import build_model
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL

        production = resolve_runtime_forward(TASKS["qwen3_5_attention_block"])
        for dtype in ("float32", "bfloat16"):
            torch.manual_seed(0)
            model = build_model(CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=2,
                                                  seq_len=SEQ, dtype=dtype))
            with torch.no_grad():                    # a trained spread, not the zero init
                for name, p in model.named_parameters():
                    if name.endswith("norm.weight"):
                        p.copy_(0.1 * torch.randn(p.shape))
            attn = model.model.layers[FULL_LAYER].self_attn
            seen = {}

            def keep(module, args, kwargs, output):
                seen["x"] = kwargs["hidden_states"]
                seen["cos"], seen["sin"] = kwargs["position_embeddings"]
                seen["y"] = output[0]

            handle = attn.register_forward_hook(keep, with_kwargs=True)
            try:
                with torch.no_grad():
                    model(input_ids=torch.randint(0, TINY_ARCH["vocab_size"], (2, SEQ)),
                          use_cache=False)
            finally:
                handle.remove()
            cos, sin = seen["cos"], seen["sin"]
            self.assertEqual(tuple(cos.shape), (2, SEQ, 8))
            self.assertTrue(torch.equal(cos[0], cos[1]))
            with self.subTest(dtype=dtype), torch.no_grad():
                y = production(seen["x"], attn.q_proj.weight, attn.k_proj.weight,
                               attn.v_proj.weight, attn.q_norm.weight, attn.k_norm.weight,
                               attn.o_proj.weight, cos[:1], sin[:1], attn.q_norm.eps)
                self.assertTrue(torch.equal(y, seen["y"]))
                self.assertEqual(y.stride(), seen["y"].stride())


class TestProductionCompiles(unittest.TestCase):
    """Every production spelling traces whole under ``torch.compile(fullgraph=True)``
    (Dynamo's ``eager`` backend, no toolchain needed): the Tier-2 compile baseline,
    the Tier-3 per-site compile provider and the trusted-compile control all
    compile them."""

    def test_fullgraph(self):
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
                    self.assertTrue(torch.equal(got, want))
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
