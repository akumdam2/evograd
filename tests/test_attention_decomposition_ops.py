"""The three primitives the causal GQA attention boundary decomposes into.

``gqa_scaled_scores`` -> ``causal_softmax`` -> ``gqa_pv`` are reusable Level-1
contracts; the Qwen3-0.6B case each carries is *derived* from the observed
``qwen3_attention`` boundary, not observed on its own, and the public observed
suites must not change because of them. CPU only: what is checked is the
contract, the dtype policy, the GQA head mapping and reduction, the causal
semantics, and that the three compose into the declared attention oracle.
"""

from __future__ import annotations

import math
import unittest

import torch

from evograd.benchmark import TASKS, get_task
from evograd.benchmark.topdown.common.level2_references import attention_projection_forward_ref
from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values
from evograd.opdecl.models import QWEN3_0_6B, rederive_dims
from evograd.opdecl.oracle import oracle, resolve_forward, resolve_runtime_forward
from evograd.ops import PRIMITIVES
from evograd.pipelines.shared.primitives import PrimitiveViolation, check_source

DECOMPOSITION = ("gqa_scaled_scores", "causal_softmax", "gqa_pv")
SUITE = "qwen3_0_6b_attention_derived"
SMALL = {"B": 2, "HQ": 4, "HK": 2, "T": 32, "D": 16}


class TestContracts(unittest.TestCase):
    def test_registered_as_level_one_primitives(self):
        for name in DECOMPOSITION:
            with self.subTest(op=name):
                self.assertIn(name, PRIMITIVES)
                self.assertEqual(TASKS[name].level, 1)

    def test_dtype_policy_is_explicit(self):
        scores = get_task("gqa_scaled_scores")
        self.assertEqual(scores.output.dtype, "float32")
        self.assertIsNone(scores.args[0].dtype)  # q follows the model dtype
        softmax = get_task("causal_softmax")
        self.assertEqual(softmax.args[0].dtype, "float32")
        self.assertEqual(softmax.output.dtype, "bfloat16")
        self.assertEqual({w.dtype for w in softmax.correctness}, {"bfloat16"})
        pv = get_task("gqa_pv")
        self.assertIsNone(pv.args[0].dtype)
        self.assertIsNone(pv.output.dtype)

    def test_gradient_orders(self):
        self.assertEqual(get_task("gqa_scaled_scores").grad_names(), ("dq", "dk"))
        self.assertEqual(get_task("causal_softmax").grad_names(), ("ds",))
        self.assertEqual(get_task("gqa_pv").grad_names(), ("dp", "dv"))

    def test_every_contract_carries_the_target_size_check(self):
        for name in DECOMPOSITION:
            with self.subTest(op=name):
                big = [w for w in get_task(name).correctness if w.dims["T"] == 2048]
                self.assertEqual(len(big), 1)
                self.assertEqual(big[0].dtype, "bfloat16")
                self.assertEqual(big[0].dims["HQ"], 16)

    def test_inputs_have_the_decoder_layout(self):
        for name, heads in (("gqa_scaled_scores", ("q", "k")), ("gqa_pv", ("v",))):
            op = get_task(name)
            values = make_case_inputs(op, op.correctness[0], device="cpu")
            for arg in heads:
                tensor = values[arg]
                b, h, t, d = tensor.shape
                with self.subTest(op=name, arg=arg):
                    self.assertFalse(tensor.is_contiguous())
                    self.assertEqual(tensor.stride(), (t * h * d, d, h * d, 1))
        scores = get_task("gqa_scaled_scores")
        values = make_case_inputs(scores, scores.correctness[0], device="cpu")
        self.assertEqual(values["ds"].dtype, torch.float32)


class TestDerivedCases(unittest.TestCase):
    def test_the_derived_suite_exists_and_the_observed_suites_are_untouched(self):
        for name in DECOMPOSITION:
            with self.subTest(op=name):
                suites = TASKS[name].benchmark_suites
                self.assertIn(SUITE, suites)
                self.assertNotIn("qwen3_0_6b_observed", suites)
                (case,) = suites[SUITE]
                self.assertEqual(case.provenance.model, "qwen3_0_6b")
                self.assertEqual(case.provenance.source, "hf_config")
                self.assertIn("derived from the attention decomposition", case.provenance.note)
                self.assertIn("not an independently observed operator", case.provenance.note)
                self.assertEqual(case.dims, rederive_dims(case.provenance))
        # The observed Level-1 bindings still list exactly the six harvested tasks.
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import OBSERVED_BINDINGS

        self.assertEqual(
            sorted(b.task for b in OBSERVED_BINDINGS),
            ["causal_gqa_attention", "cross_entropy", "linear_no_bias", "rmsnorm", "rope", "swiglu"],
        )

    def test_the_derived_dims_match_the_observed_attention_boundary(self):
        attention = get_task("qwen3_attention").benchmark[0].dims
        scores = TASKS["gqa_scaled_scores"].benchmark[0].dims
        softmax = TASKS["causal_softmax"].benchmark[0].dims
        pv = TASKS["gqa_pv"].benchmark[0].dims
        for key in ("B", "HQ", "HK", "T", "D"):
            self.assertEqual(scores[key], attention[key])
            self.assertEqual(pv[key], attention[key])
        self.assertEqual(softmax, {"B": attention["B"], "HQ": attention["HQ"], "T": attention["T"]})
        self.assertEqual(QWEN3_0_6B.causal_softmax_dims(batch=2, seq=2048), softmax)


class TestMathematics(unittest.TestCase):
    """The three oracles compose into the declared attention oracle."""

    def _chain(self, dtype):
        torch.manual_seed(0)
        d = SMALL
        q = torch.randn(d["B"], d["T"], d["HQ"], d["D"], dtype=dtype).transpose(1, 2)
        k = torch.randn(d["B"], d["T"], d["HK"], d["D"], dtype=dtype).transpose(1, 2)
        v = torch.randn(d["B"], d["T"], d["HK"], d["D"], dtype=dtype).transpose(1, 2)
        w = torch.randn(24, d["HQ"] * d["D"], dtype=dtype) * 0.1
        scores = resolve_forward(get_task("gqa_scaled_scores"))
        softmax = resolve_forward(get_task("causal_softmax"))
        pv = resolve_forward(get_task("gqa_pv"))
        s = scores(q, k)
        p = softmax(s)
        o = pv(p, v)
        merged = o.transpose(1, 2).contiguous().reshape(d["B"], d["T"], -1)
        out = torch.nn.functional.linear(merged, w)
        return (q, k, v, w), (s, p, o), out

    def test_composition_equals_the_attention_oracle_in_bf16(self):
        (q, k, v, w), (s, p, o), out = self._chain(torch.bfloat16)
        expected = attention_projection_forward_ref(q, k, v, w)
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(p.dtype, torch.bfloat16)
        self.assertEqual(o.dtype, torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_scores_are_scaled_and_use_the_shared_kv_head(self):
        d = SMALL
        torch.manual_seed(1)
        q = torch.randn(d["B"], d["HQ"], d["T"], d["D"])
        k = torch.randn(d["B"], d["HK"], d["T"], d["D"])
        s = resolve_forward(get_task("gqa_scaled_scores"))(q, k)
        groups = d["HQ"] // d["HK"]
        for hq in range(d["HQ"]):
            expected = (q[:, hq] @ k[:, hq // groups].transpose(-2, -1)) / math.sqrt(d["D"])
            torch.testing.assert_close(s[:, hq], expected)

    def test_causal_softmax_masks_strictly_future_keys(self):
        s = torch.randn(1, 2, 8, 8)
        p = resolve_forward(get_task("causal_softmax"))(s)
        future = torch.ones(8, 8, dtype=torch.bool).triu(1)
        self.assertTrue((p[..., future] == 0).all())
        torch.testing.assert_close(p.float().sum(-1), torch.ones(1, 2, 8), atol=1e-2, rtol=1e-2)

    def test_backward_reduces_shared_kv_heads_in_float32(self):
        op = get_task("gqa_scaled_scores")
        workload = op.correctness[0]
        self.assertEqual(workload.dtype, "float32")
        values = make_case_inputs(op, workload, device="cpu")
        _y, grads = oracle(op, values)
        q, k, ds = values["q"], values["k"], values["ds"]
        groups = q.shape[1] // k.shape[1]
        scale = 1.0 / math.sqrt(q.shape[-1])
        dk_exp = torch.matmul(ds.transpose(-2, -1), q.float()) * scale
        b, hk, t, d = k.shape
        dk = dk_exp.view(b, hk, groups, t, d).sum(2).to(k.dtype)
        self.assertEqual(grads["dk"].shape, tuple(k.shape))
        torch.testing.assert_close(grads["dk"], dk)

    def test_runtime_forwards_match_their_oracles_at_the_declared_tolerance(self):
        for name in ("gqa_scaled_scores", "gqa_pv"):
            op = get_task(name)
            for workload in op.correctness[:6]:
                with self.subTest(op=name, dims=workload.dims, dtype=workload.dtype):
                    values = make_case_inputs(op, workload, device="cpu")
                    args = [values[a.name] for a in op.args]
                    if name == "gqa_scaled_scores" and workload.dtype == "bfloat16":
                        # bmm(out_dtype=float32) is CUDA-only; the runtime is
                        # exercised on GPU by verify_runtime_forward.
                        continue
                    expected = resolve_forward(op)(*args)
                    actual = resolve_runtime_forward(op)(*args)
                    atol, rtol = op.tolerance_for(workload, op.output.name)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.stride(), expected.stride())
                    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


class TestOpaqueSoftmaxIsDenied(unittest.TestCase):
    SOURCE = """\
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
def causal_softmax_forward_with_saved(s):
    p = {call}
    return p.to(torch.bfloat16), (p,)


def causal_softmax_backward_from_saved(dp, saved_tensors):
    (p,) = saved_tensors
    return p * (dp.float() - (dp.float() * p).sum(-1, keepdim=True))
# EVOLVE-BLOCK-END
"""

    def test_torch_softmax_and_the_method_form_are_violations(self):
        for call in ("torch.softmax(s, -1)", "s.softmax(-1)", "torch.log_softmax(s, -1)"):
            with self.subTest(call=call):
                with self.assertRaises(PrimitiveViolation):
                    check_source(self.SOURCE.format(call=call))

    def test_an_explicit_spelling_is_not(self):
        source = self.SOURCE.format(call="torch.exp(s - s.amax(-1, keepdim=True))")
        check_source(source)
