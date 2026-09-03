"""Semantic-faithfulness repairs to the Level-1 mapping.

Three claims that were wrong or unproven, and the tests that keep them fixed:

* a zero-valued bias is not a biasless projection;
* the RoPE oracle upcasts to float32 and Transformers does not, so timing the
  eager baseline through the oracle charges it for casts the model never runs;
* a loss magnitude near ``ln(vocab)`` is a sanity check, not evidence that the
  Level-1 cross entropy is the model's.
"""

from __future__ import annotations

import unittest

import torch

from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle, resolve_forward, resolve_runtime_forward
from evograd.ops import OPS, get_op

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_TRANSFORMERS = False


class TestBiaslessLinear(unittest.TestCase):
    def test_the_task_exists_and_has_no_bias_anywhere(self):
        op = get_op("linear_no_bias")
        self.assertEqual((op.level, op.family), (1, "gemm"))
        self.assertEqual([a.name for a in op.args], ["x", "weight"])
        self.assertEqual(op.output_names, ("y",))
        self.assertEqual(op.grad_names(), ("dx", "dweight"))
        for text in (op.forward_semantics, op.backward_semantics, op.extra_constraints):
            self.assertNotIn("dbias", text.replace("no dbias", ""))

    def test_the_backward_contract_says_two_gradients(self):
        op = get_op("linear_no_bias")
        self.assertIn("(dx, dweight)", op.backward_semantics)
        self.assertIn("no dbias", op.backward_semantics)

    def test_inputs_carry_no_bias_and_the_weight_is_out_by_in(self):
        op = get_op("linear_no_bias")
        for workload in op.correctness + op.benchmark_workloads("qwen3_0_6b_observed")[:1]:
            with self.subTest(dims=workload.dims):
                values = make_case_inputs(op, workload, device="cpu")
                self.assertNotIn("bias", values)
                self.assertEqual(sorted(values), ["dy", "weight", "x"])
                n, k = workload.dims["N"], workload.dims["K"]
                self.assertEqual(list(values["weight"].shape), [n, k])
                self.assertTrue(values["weight"].is_contiguous())

    def test_the_weight_interface_is_not_matmuls(self):
        """``matmul`` takes ``[K, N]``; ``nn.Linear`` stores ``[N, K]``. Mapping
        the observed projections onto ``matmul`` would benchmark a transpose the
        model does not perform."""
        no_bias = get_op("linear_no_bias")
        matmul = get_op("matmul")
        self.assertEqual([a.shape for a in no_bias.args], ["[M, K]", "[N, K]"])
        self.assertIn("[K, N]", [a.shape for a in matmul.args])
        self.assertTrue(
            all(
                w.provenance is None or w.provenance.model != "qwen3_0_6b"
                for w in matmul.coverage
            )
        )

    def test_runtime_forward_passes_none_rather_than_zeros(self):
        import inspect

        from evograd.ops.level1.linear_no_bias import forward_ref

        op = get_op("linear_no_bias")
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
        op = get_op("linear_no_bias")
        values = make_case_inputs(op, op.correctness[0], device="cpu")
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
        for name, op in OPS.items():
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
        op = get_op("linear")
        for group in (op.correctness, op.coverage, op.benchmark):
            for workload in group:
                if workload.provenance is not None:
                    self.assertNotEqual(workload.provenance.model, "qwen3_0_6b")
        self.assertNotIn("qwen3_0_6b_observed", op.benchmark_suites)

    def test_the_biased_grid_no_longer_claims_to_model_llama_faithfully(self):
        op = get_op("linear")
        for workload in op.benchmark:
            self.assertTrue(workload.provenance.scaled)
            self.assertIn("no projection biases", workload.provenance.note)
            self.assertIn("linear_no_bias", workload.provenance.note)


class TestRopeRuntimeBaseline(unittest.TestCase):
    def test_the_task_declares_both_spellings(self):
        from evograd.ops.level1.rope import forward_ref

        op = get_op("rope")
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

        op = get_op("rope")
        workload = next(w for w in op.correctness if w.dtype == "bfloat16")
        values = make_case_inputs(op, workload, device="cpu")
        a = rope_forward_ref(values["x"], values["cos"], values["sin"]).float()
        b = rope_runtime_ref(values["x"], values["cos"], values["sin"]).float()
        self.assertFalse(torch.equal(a, b))
        self.assertLess(float((a - b).abs().max()) / float(a.abs().max()), 5e-2)

    def test_both_layouts_survive_the_timed_spelling(self):
        from evograd.ops.level1.rope.forward_ref import rope_runtime_ref

        op = get_op("rope")
        for workload in (op.benchmark[0], op.benchmark_workloads("qwen3_0_6b_observed")[0]):
            values = make_case_inputs(op, workload, device="cpu")
            out = rope_runtime_ref(values["x"], values["cos"], values["sin"])
            with self.subTest(dims=workload.dims):
                self.assertEqual(out.stride(), values["x"].stride())

    def test_the_runtime_gate_passes(self):
        from evograd.opdecl import baselines

        baselines._RUNTIME_FORWARD_VERIFIED.discard("rope")
        baselines.verify_runtime_forward(get_op("rope"), device="cpu")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestCrossEntropyCanonicalCheck(unittest.TestCase):
    """The equivalence proof runs the model and compares live tensors.

    Driven here with the two-layer CPU model so the machinery is tested without
    a GPU; the canonical numbers come from the same function on the real
    workload.
    """

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.qwen3.levels.level1.mapping import run_cross_entropy_check

        from tests.qwen3.test_level4_workload import tiny_spec

        cls.report = run_cross_entropy_check(device="cpu", spec=tiny_spec())

    def test_it_intercepted_the_model_s_own_call(self):
        self.assertEqual(self.report["status"], "pass", self.report["failures"])
        self.assertFalse(self.report["canonical"])
        signature = self.report["signature"]
        self.assertEqual(signature["logits"]["dtype"], "torch.float32")
        self.assertEqual(signature["target"]["dtype"], "torch.int64")
        self.assertEqual(len(signature["logits"]["shape"]), 2)

    def test_the_contract_scalars_are_the_model_s(self):
        self.assertEqual(self.report["attrs"]["ignore_index"], -100)
        self.assertEqual(self.report["attrs"]["reduction"], "mean")
        self.assertEqual(self.report["loss"]["upstream_scalar"], 1.0)

    def test_loss_and_dlogits_are_compared_not_just_the_magnitude(self):
        loss = self.report["loss"]
        self.assertEqual(loss["model"], loss["reference"])
        self.assertEqual(loss["abs_error"], 0.0)
        self.assertEqual(loss["model"], loss["step_loss"])
        dlogits = self.report["dlogits"]
        for field in ("shape_match", "dtype_match", "stride_match", "bitwise_identical"):
            self.assertTrue(dlogits[field], field)
        self.assertEqual(dlogits["max_abs_err"], 0.0)
        self.assertTrue(dlogits["actual_all_finite"] and dlogits["expected_all_finite"])

    def test_nothing_is_persisted(self):
        self.assertFalse(self.report["tensors_written"])

    def test_the_ln_vocab_check_is_a_sanity_test_not_the_proof(self):
        """It would pass for an implementation with the right scale and the
        wrong gradient, which is exactly what the comparison above rules out."""
        from evograd.bench.workloads.qwen3.levels.level1 import mapping as level1

        proof = level1.run_cross_entropy_check.__doc__
        self.assertIn("not the equivalence proof", proof)
        self.assertIn("sanity check", level1.run_loss_check.__doc__)


class TestComposition(unittest.TestCase):
    def test_the_biasless_projections_still_compose_into_level_two(self):
        from evograd.bench.workloads.qwen3.harvest.snapshot import load

        configs = {
            tuple(c["roles"]): c["dims"]
            for c in load()["level1"]["linear_no_bias"]["configurations"]
        }
        qkv = get_op("qwen3_qkv_norm_rope").benchmark[0].dims
        mlp = get_op("qwen3_swiglu_mlp").benchmark[0].dims
        attention = get_op("qwen3_attention").benchmark[0].dims
        self.assertEqual(configs[("q_proj",)]["N"], qkv["QO"])
        self.assertEqual(configs[("k_proj", "v_proj")]["N"], qkv["KVO"])
        self.assertEqual(configs[("gate_proj", "up_proj")]["N"], mlp["I"])
        self.assertEqual(configs[("down_proj",)]["K"], mlp["I"])
        self.assertEqual(configs[("o_proj",)]["K"], attention["QO"])
        self.assertEqual(configs[("lm_head",)]["K"], qkv["H"])


if __name__ == "__main__":
    unittest.main()
