"""``fused_add_rms_norm`` after the move to two outputs.

The single-output contract was wrong about the operator it claimed to describe.
A decoder's residual stream keeps the *un-normalized* sum and hands it to the
next block, so the fusion has two outputs and its backward has two paths into
``x`` and ``residual``. These tests pin the corrected contract, and the
characteristic failure -- a backward that ignores ``dsummed`` -- has a test of
its own, because it leaves ``dweight`` perfectly correct and would slip past any
check that looked only there.
"""

from __future__ import annotations

import unittest

import torch

from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values
from evograd.opdecl.models import QWEN3_0_6B, rederive_dims
from evograd.opdecl.oracle import oracle
from evograd.opdecl.verify import verify
from evograd.ops import get_op
from evograd.ops.level2.fused_add_rms_norm import QWEN3_FUSION_SITES
from evograd.ops.level2.fused_add_rms_norm.forward_ref import (
    fused_add_rms_norm_forward_ref,
    fused_add_rms_norm_runtime_ref,
)

OP = get_op("fused_add_rms_norm")


class _Module:
    def __init__(self, forward, backward):
        setattr(self, OP.forward_fn_name, forward)
        setattr(self, OP.backward_fn_name, backward)


def _reference_pair():
    """A correct candidate pair, written against the declared contract."""

    def forward(x, r, weight, eps=1e-6):
        summed = x + r
        rstd = torch.rsqrt(summed.float().pow(2).mean(-1, keepdim=True) + eps)
        normalized = (summed * rstd.to(summed.dtype)) * weight
        return (normalized, summed), (summed, weight, rstd)

    def backward(output_grads, saved_tensors, eps=1e-6):
        dout, dsummed = output_grads
        summed, weight, rstd = saved_tensors
        wide = summed.float()
        rstd_f = rstd.float()
        dnorm = dout.float() * weight.float()
        dtotal = rstd_f * (
            dnorm - wide * (dnorm * wide).mean(-1, keepdim=True) * rstd_f * rstd_f
        )
        dtotal = dtotal.to(summed.dtype) + dsummed
        dweight = (dout.float() * wide * rstd_f).sum(0).to(weight.dtype)
        return dtotal, dtotal.clone(), dweight

    return forward, backward


def _ignores_dsummed_pair():
    """The characteristic error: correct dweight, wrong dx and dr."""
    forward, backward = _reference_pair()

    def broken(output_grads, saved_tensors, eps=1e-6):
        dout, _dsummed = output_grads
        return backward((dout, torch.zeros_like(_dsummed)), saved_tensors, eps)

    return forward, broken


class TestContract(unittest.TestCase):
    def test_two_named_outputs_with_the_former_one_first(self):
        self.assertTrue(OP.is_multi_output)
        self.assertEqual(OP.output_names, ("out", "summed"))
        self.assertEqual(OP.upstream_grad_names, ("dout", "dsummed"))
        self.assertEqual(OP.forward_returns(), "(out, summed), saved_tensors")

    def test_gradients_are_unchanged(self):
        self.assertEqual(OP.grad_names(), ("dx", "dr", "dweight"))
        self.assertTrue(OP.backward_parameters().startswith("output_grads, saved_tensors"))

    def test_both_references_return_the_pair(self):
        torch.manual_seed(0)
        x, r = torch.randn(8, 16), torch.randn(8, 16)
        weight = torch.randn(16)
        for forward in (fused_add_rms_norm_forward_ref, fused_add_rms_norm_runtime_ref):
            with self.subTest(forward=forward.__name__):
                out, summed = forward(x, r, weight, 1e-6)
                self.assertEqual(out.shape, x.shape)
                self.assertTrue(torch.equal(summed, x + r))

    def test_summed_is_the_sum_itself_not_a_recomputation(self):
        """Both spellings must return the same tensor values the norm consumed."""
        torch.manual_seed(0)
        x, r = torch.randn(4, 8), torch.randn(4, 8)
        weight = torch.randn(8)
        _o1, s1 = fused_add_rms_norm_forward_ref(x, r, weight, 1e-6)
        _o2, s2 = fused_add_rms_norm_runtime_ref(x, r, weight, 1e-6)
        self.assertTrue(torch.equal(s1, s2))

    def test_the_semantics_state_the_combination_rule(self):
        self.assertIn("dsummed", OP.backward_semantics)
        self.assertIn("dtotal", OP.backward_semantics)
        self.assertIn("(out, summed)", OP.forward_semantics)


class TestInputsAndOracle(unittest.TestCase):
    def test_two_independent_upstream_gradients_are_generated(self):
        values = make_case_inputs(OP, OP.correctness[0], device="cpu")
        self.assertIn("dout", values)
        self.assertIn("dsummed", values)
        self.assertFalse(torch.allclose(values["dout"], values["dsummed"]))
        self.assertGreater(float(values["dsummed"].abs().max()), 0.0)
        grads = upstream_grad_values(OP, values)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 2)

    def test_the_oracle_returns_both_outputs(self):
        values = make_case_inputs(OP, OP.correctness[0], device="cpu")
        outputs, grads = oracle(OP, values)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(sorted(grads), ["dr", "dweight", "dx"])

    def test_dsummed_reaches_x_and_r_but_not_weight(self):
        """The whole point of the second output path."""
        values = make_case_inputs(OP, OP.correctness[0], device="cpu")
        _out, full = oracle(OP, values)
        values["dsummed"] = torch.zeros_like(values["dsummed"])
        _out, without = oracle(OP, values)
        self.assertFalse(torch.allclose(full["dx"], without["dx"]))
        self.assertFalse(torch.allclose(full["dr"], without["dr"]))
        self.assertTrue(torch.allclose(full["dweight"], without["dweight"]))

    def test_dx_and_dr_are_the_same_gradient(self):
        values = make_case_inputs(OP, OP.correctness[0], device="cpu")
        _out, grads = oracle(OP, values)
        self.assertTrue(torch.allclose(grads["dx"], grads["dr"], atol=0, rtol=0))


class TestCorrectnessGate(unittest.TestCase):
    def _failed(self, module):
        report = verify(OP, module, device="cpu")
        names = []
        for case in report.cases:
            if case.error:
                names.append("error")
            else:
                names.extend(c.name for c in case.checks if not c.ok)
        return set(names)

    def test_a_correct_pair_passes(self):
        report = verify(OP, _Module(*_reference_pair()), device="cpu")
        for case in report.cases:
            self.assertIsNone(case.error)
            self.assertEqual(
                [c.name for c in case.checks], ["out", "summed", "dx", "dr", "dweight"]
            )
            self.assertTrue(all(c.ok for c in case.checks), [c for c in case.checks if not c.ok])

    def test_a_backward_that_ignores_dsummed_is_rejected(self):
        """The negative control. It gets `dweight` exactly right, so a gate that
        checked only the normalized path would pass it."""
        failed = self._failed(_Module(*_ignores_dsummed_pair()))
        self.assertIn("dx", failed)
        self.assertIn("dr", failed)
        self.assertNotIn("dweight", failed)
        self.assertNotIn("out", failed)
        self.assertNotIn("summed", failed)

    def test_a_forward_returning_only_the_normalized_output_is_rejected(self):
        forward, backward = _reference_pair()

        def single(x, r, weight, eps=1e-6):
            (normalized, _summed), saved = forward(x, r, weight, eps)
            return normalized, saved

        self.assertEqual(self._failed(_Module(single, backward)), {"error"})

    def test_swapped_outputs_are_rejected(self):
        forward, backward = _reference_pair()

        def swapped(x, r, weight, eps=1e-6):
            (normalized, summed), saved = forward(x, r, weight, eps)
            return (summed, normalized), saved

        self.assertTrue(self._failed(_Module(swapped, backward)))

    def test_a_recomputed_summed_that_disagrees_is_rejected(self):
        forward, backward = _reference_pair()

        def drifted(x, r, weight, eps=1e-6):
            (normalized, summed), saved = forward(x, r, weight, eps)
            return (normalized, summed + 1.0), saved

        self.assertEqual(self._failed(_Module(drifted, backward)), {"summed"})


class TestQwenProvenance(unittest.TestCase):
    def test_the_observed_suite_is_the_canonical_configuration(self):
        cases = OP.benchmark_workloads("qwen3_0_6b_observed")
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.dims, {"rows": 4096, "cols": 1024})
        self.assertEqual(case.dtype, "bfloat16")
        self.assertEqual(case.provenance.source, "hf_config")
        self.assertEqual(case.provenance.model, "qwen3_0_6b")
        self.assertEqual(case.dims, rederive_dims(case.provenance))

    def test_the_generic_llama_grid_is_preserved(self):
        """The Qwen suite is added, not substituted."""
        full = OP.benchmark_workloads("full")
        self.assertTrue(full)
        self.assertTrue(all(w.provenance.model == "llama_3_8b" for w in full))
        self.assertEqual(len(OP.benchmark_workloads("legacy")), 16)

    def test_fifty_six_fusion_sites_derived_not_counted(self):
        sites = QWEN3_FUSION_SITES
        self.assertEqual(sites["total"], 56)
        self.assertEqual(sites["attention_residual_then_post_attention_layernorm"], 28)
        self.assertEqual(sites["mlp_residual_then_next_layer_input_layernorm"], 27)
        self.assertEqual(sites["final_mlp_residual_then_model_norm"], 1)
        self.assertEqual(sites["excluded_layer0_input_layernorm"], 1)
        self.assertEqual(sites, QWEN3_0_6B.residual_rmsnorm_fusion_sites())

    def test_the_count_is_one_short_of_the_observed_rms_norms(self):
        """28 input_layernorm + 28 post_attention_layernorm + 1 model.norm = 57
        residual-width RMSNorms; layer 0's input_layernorm has no preceding
        decoder residual add, so 56 are fusion sites."""
        from evograd.bench.workloads.qwen3.harvest.snapshot import load

        entry = load()["tasks"]["fused_add_rms_norm"]
        sites = entry["fusion_sites"]
        self.assertEqual(entry["frequency"], 57)
        self.assertEqual(sites["observed_rms_norm_invocations"], 57)
        self.assertEqual(sites["total"] + sites["excluded_layer0_input_layernorm"], 57)

    def test_the_directly_verified_count_is_distinguished_from_the_frequency(self):
        from evograd.bench.workloads.qwen3.harvest.snapshot import load

        sites = load()["tasks"]["fused_add_rms_norm"]["fusion_sites"]
        self.assertEqual(sites["directly_verified_invocations"], 1)
        self.assertNotEqual(sites["directly_verified_invocations"], sites["total"])
        self.assertIn("layer 14", sites["directly_verified_note"])

    def test_the_declaration_agrees_with_the_snapshot(self):
        from evograd.bench.workloads.qwen3.levels.level2 import residual_rmsnorm as residual_module

        self.assertEqual(residual_module.declaration_problems(), [])


class TestLigerProvider(unittest.TestCase):
    """Liger's kernel already implements the corrected contract; the adapter
    used to throw half of it away."""

    def test_the_adapter_returns_both_outputs_and_consumes_dsummed(self):
        import inspect

        from evograd.ops.level2.fused_add_rms_norm import liger

        source = inspect.getsource(liger.make_liger_fused_add_rms_norm_autograd_pair_fns)
        self.assertIn("return (output, summed)", source)
        self.assertIn("dout, dsummed = output_grads", source)
        self.assertIn("dsummed.contiguous()", source)
        self.assertNotIn("torch.zeros_like", source)

    def test_the_underlying_kernel_accepts_the_summed_gradient(self):
        """Not an assumption: Liger's own signature has the slot, and its own
        autograd Function takes two output gradients."""
        try:
            import liger_kernel.ops.fused_add_rms_norm as module
        except Exception as exc:  # pragma: no cover - depends on the machine
            self.skipTest(f"liger-kernel not importable: {exc}")
        import inspect

        backward = inspect.signature(module.fused_add_rms_norm_backward)
        self.assertIn("dS_out", backward.parameters)
        function_backward = inspect.signature(
            module.LigerFusedAddRMSNormFunction.backward
        )
        self.assertIn("dS_out", function_backward.parameters)

    def test_the_baseline_is_still_declared(self):
        self.assertIn("liger", OP.performance_baselines)


if __name__ == "__main__":
    unittest.main()
