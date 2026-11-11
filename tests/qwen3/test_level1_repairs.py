"""Semantic-faithfulness repairs to the Level-1 mapping.

Two claims that were wrong, and the tests that keep them fixed:

* a zero-valued bias is not a biasless projection;
* the RoPE oracle upcasts to float32 and Transformers does not, so timing the
  eager baseline through the oracle charges it for casts the model never runs.
"""

from __future__ import annotations

import unittest

import torch

from evograd.opdecl import Workload
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle, resolve_forward, resolve_runtime_forward
from evograd.benchmark import TASKS, get_task
from tests._small_cases import SMALL

SMALL_LINEAR = SMALL["linear_no_bias"][0]

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_TRANSFORMERS = False


class TestBiaslessLinear(unittest.TestCase):
    def test_the_task_exists_and_has_no_bias_anywhere(self):
        op = get_task("linear_no_bias")
        self.assertEqual((op.level, op.family), (1, "gemm"))
        self.assertEqual([a.name for a in op.args], ["x", "weight"])
        self.assertEqual(op.output_names, ("y",))
        self.assertEqual(op.grad_names(), ("dx", "dweight"))
        for text in (op.forward_semantics, op.backward_semantics, op.extra_constraints):
            self.assertNotIn("dbias", text.replace("no dbias", ""))

    def test_the_backward_contract_says_two_gradients(self):
        op = get_task("linear_no_bias")
        self.assertIn("(dx, dweight)", op.backward_semantics)
        self.assertIn("no dbias", op.backward_semantics)

    def test_inputs_carry_no_bias_and_the_weight_is_out_by_in(self):
        op = get_task("linear_no_bias")
        for workload in (SMALL_LINEAR, *op.benchmark_workloads("qwen3_0_6b_observed")[:1]):
            with self.subTest(dims=workload.dims):
                values = make_case_inputs(op, workload, device="cpu")
                self.assertNotIn("bias", values)
                self.assertEqual(sorted(values), ["dy", "weight", "x"])
                n, k = workload.dims["N"], workload.dims["K"]
                self.assertEqual(list(values["weight"].shape), [n, k])
                self.assertTrue(values["weight"].is_contiguous())

    def test_the_weight_interface_is_nn_linears(self):
        """``nn.Linear`` stores ``[N, K]``. A ``[K, N]`` weight interface would
        benchmark a transpose the model does not perform."""
        no_bias = get_task("linear_no_bias")
        self.assertEqual([a.shape for a in no_bias.args], ["[M, K]", "[N, K]"])

    def test_runtime_forward_passes_none_rather_than_zeros(self):
        import inspect

        from evograd.ops.level1.linear_no_bias import forward_ref

        op = get_task("linear_no_bias")
        self.assertIs(resolve_runtime_forward(op), forward_ref.linear_no_bias_runtime_ref)
        source = inspect.getsource(forward_ref.linear_no_bias_runtime_ref)
        # Body only: the docstring mentions the zero-bias spelling to say why it
        # is wrong, which would otherwise trip a naive substring check.
        body = source.split('"""')[-1]
        self.assertIn("F.linear(x, weight, None)", body)
        self.assertNotIn("zeros", body)

    def test_the_oracle_is_more_accurate_than_the_timed_spelling(self):
        from evograd.ops.level1.linear_no_bias.forward_ref import (
            linear_no_bias_forward_ref,
            linear_no_bias_runtime_ref,
        )

        torch.manual_seed(0)
        x = torch.randn(64, 128, dtype=torch.bfloat16)
        weight = (torch.randn(96, 128) * 128**-0.5).to(torch.bfloat16)
        exact = (x.double() @ weight.double().t()).float()
        oracle_out = linear_no_bias_forward_ref(x, weight).float()
        timed = linear_no_bias_runtime_ref(x, weight).float()
        scale = float(exact.abs().max())
        self.assertLessEqual(
            float((oracle_out - exact).abs().max()) / scale,
            float((timed - exact).abs().max()) / scale,
        )

    def test_backward_returns_two_gradients(self):
        op = get_task("linear_no_bias")
        values = make_case_inputs(op, SMALL_LINEAR, device="cpu")
        y, grads = oracle(op, values)
        self.assertEqual(sorted(grads), ["dweight", "dx"])
        self.assertEqual(list(y.shape), [64, 64])
        self.assertNotIn("dbias", grads)

    def test_mismatched_shapes_are_rejected(self):
        from evograd.ops.level1.linear_no_bias.forward_ref import linear_no_bias_forward_ref

        with self.assertRaises(ValueError):
            linear_no_bias_forward_ref(torch.randn(4, 8), torch.randn(6, 9))
        with self.assertRaises(ValueError):
            linear_no_bias_forward_ref(torch.randn(4, 8, 2), torch.randn(6, 8))


class TestNoQwenBiasContract(unittest.TestCase):
    """Negative tests: nothing Qwen-mapped may carry a bias contract."""

    def test_no_qwen_workload_belongs_to_a_task_with_a_bias_argument(self):
        for name, op in TASKS.items():
            observed = [
                w
                for w in op.coverage
                if w.provenance is not None and w.provenance.model == "qwen3_0_6b"
            ]
            if not observed:
                continue
            with self.subTest(op=name):
                self.assertNotIn("bias", [a.name for a in op.args], name)
                self.assertNotIn("dbias", op.grad_names(), name)

    def test_the_biased_linear_task_has_no_qwen_workloads_at_all(self):
        op = get_task("linear")
        for group in (op.coverage, op.benchmark):
            for workload in group:
                if workload.provenance is not None:
                    self.assertNotEqual(workload.provenance.model, "qwen3_0_6b")
        self.assertNotIn("qwen3_0_6b_observed", op.benchmark_suites)

class TestRopeRuntimeBaseline(unittest.TestCase):
    def test_the_task_declares_both_spellings(self):
        from evograd.ops.level1.rope import forward_ref

        op = get_task("rope")
        self.assertIs(resolve_forward(op), forward_ref.rope_forward_ref)
        self.assertIs(resolve_runtime_forward(op), forward_ref.rope_runtime_ref)

    def test_the_timed_spelling_has_no_float32_cast(self):
        import inspect

        from evograd.ops.level1.rope import forward_ref

        timed = inspect.getsource(forward_ref.rope_runtime_ref)
        oracle_source = inspect.getsource(forward_ref.rope_forward_ref)
        self.assertNotIn(".float()", timed)
        self.assertIn(".float()", oracle_source)

    @unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
    def test_the_timed_spelling_is_bitwise_transformers(self):
        from evograd.ops.level1.rope.forward_ref import rope_runtime_ref

        torch.manual_seed(0)
        tokens, head_dim = 16, 32
        x = torch.randn(2, tokens, 4, head_dim, dtype=torch.bfloat16).transpose(1, 2)
        angles = torch.arange(tokens, dtype=torch.float32)[:, None] * (
            1e6 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )[None, :]
        table = torch.cat((angles, angles), dim=-1)
        cos, sin = table.cos().to(torch.bfloat16), table.sin().to(torch.bfloat16)
        expected, _k = apply_rotary_pos_emb(x, x, cos[None], sin[None])
        got = rope_runtime_ref(x, cos, sin)
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(got.stride(), expected.stride())

    def test_the_oracle_and_the_timed_spelling_really_differ_at_bfloat16(self):
        """If they agreed, the runtime_forward would be pointless."""
        from evograd.ops.level1.rope.forward_ref import rope_forward_ref, rope_runtime_ref

        op = get_task("rope")
        workload = Workload(dims=dict(B=1, n_heads=4, T=16, head_dim=64), dtype="bfloat16")
        values = make_case_inputs(op, workload, device="cpu")
        a = rope_forward_ref(values["x"], values["cos"], values["sin"]).float()
        b = rope_runtime_ref(values["x"], values["cos"], values["sin"]).float()
        self.assertFalse(torch.equal(a, b))
        self.assertLess(float((a - b).abs().max()) / float(a.abs().max()), 5e-2)

    def test_both_layouts_survive_the_timed_spelling(self):
        from evograd.ops.level1.rope.forward_ref import rope_runtime_ref

        op = get_task("rope")
        contiguous = Workload(dims=dict(B=2, n_heads=8, T=128, head_dim=128), dtype="bfloat16")
        for workload in (contiguous, op.benchmark_workloads("qwen3_0_6b_observed")[0]):
            values = make_case_inputs(op, workload, device="cpu")
            out = rope_runtime_ref(values["x"], values["cos"], values["sin"])
            with self.subTest(dims=workload.dims):
                self.assertEqual(out.stride(), values["x"].stride())


class TestComposition(unittest.TestCase):
    def test_the_biasless_projections_still_compose_into_level_two(self):
        from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import load

        configs = {
            tuple(c["roles"]): c["dims"]
            for c in load()["level1"]["linear_no_bias"]["configurations"]
        }
        attention = get_task("qwen3_attention_block").benchmark[0].dims
        mlp = get_task("qwen3_swiglu_mlp").benchmark[0].dims
        head = get_task("qwen3_lm_head_ce").benchmark[0].dims
        self.assertEqual(configs[("q_proj",)]["N"], attention["QO"])
        self.assertEqual(configs[("k_proj", "v_proj")]["N"], attention["KVO"])
        self.assertEqual(configs[("gate_proj", "up_proj")]["N"], mlp["I"])
        self.assertEqual(configs[("down_proj",)]["K"], mlp["I"])
        self.assertEqual(configs[("o_proj",)]["K"], attention["QO"])
        self.assertEqual(configs[("lm_head",)]["K"], head["hidden"])
        self.assertEqual(configs[("lm_head",)]["N"], head["vocab"])


if __name__ == "__main__":
    unittest.main()
