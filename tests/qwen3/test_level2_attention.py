"""Level-2 ``qwen3_attention``: the observed causal-GQA-SDPA + output-projection boundary.

CPU-only except where a real derivation is needed. What is checked here is the
contract -- which boundary, which layout, which provenance -- and that the two
declared spellings compute the same thing.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.bench.workloads.qwen3.harvest import snapshot as snapshot_module
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.models import rederive_dims
from evograd.ops import OPS, get_op
from evograd.ops.level2.qwen3_attention import (
    FREQUENCY,
    HARVEST,
    OBSERVED_STRIDES,
    PROVENANCE_CHAIN,
)
from evograd.ops.level2.qwen3_attention.forward_ref import (
    qwen3_attention_forward_production,
    qwen3_attention_forward_ref,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDeclaration(unittest.TestCase):
    def test_registered_at_level_two(self):
        self.assertIn("qwen3_attention", OPS)
        op = get_op("qwen3_attention")
        self.assertEqual((op.level, op.family), (2, "attention"))

    def test_the_canonical_shape_is_the_observed_one(self):
        case = get_op("qwen3_attention").benchmark[0]
        self.assertEqual(
            case.dims,
            {"B": 2, "T": 2048, "HQ": 16, "HK": 8, "D": 128, "QO": 2048, "H": 1024},
        )
        self.assertEqual(case.dtype, "bfloat16")

    def test_provenance_is_recomputable_from_the_published_config(self):
        case = get_op("qwen3_attention").benchmark[0]
        self.assertEqual(case.provenance.source, "hf_config")
        self.assertEqual(case.provenance.model, "qwen3_0_6b")
        self.assertEqual(case.dims, rederive_dims(case.provenance))

    def test_the_declaration_agrees_with_the_snapshot(self):
        from evograd.bench.workloads.qwen3.levels.level2 import attention as attention_module

        self.assertEqual(attention_module.declaration_problems(), [])

    def test_the_observed_sdpa_configuration_is_carried_structurally(self):
        self.assertEqual(FREQUENCY, 28)
        self.assertEqual(HARVEST["config_id"], "9674b971ae24b325")
        self.assertEqual(
            HARVEST["supporting"]["output_projection"]["config_id"], "ea5311a8e1cbba90"
        )
        self.assertEqual(HARVEST["supporting"]["output_projection"]["frequency"], 28)
        attrs = HARVEST["attrs"]
        self.assertTrue(attrs["is_causal"])
        self.assertTrue(attrs["enable_gqa"])
        self.assertEqual(attrs["dropout_p"], 0.0)
        self.assertFalse(attrs["attn_mask_provided"])
        self.assertAlmostEqual(attrs["scale"], 1.0 / math.sqrt(128), places=12)

    def test_the_observed_strides_are_head_major(self):
        self.assertEqual(OBSERVED_STRIDES["q"], (4194304, 128, 2048, 1))
        self.assertEqual(OBSERVED_STRIDES["k"], (2097152, 128, 1024, 1))
        self.assertEqual(OBSERVED_STRIDES["v"], OBSERVED_STRIDES["k"])

    def test_the_boundary_excludes_the_projections_norms_and_rope(self):
        """Stated in the declaration, not left to be inferred from the arg list."""
        op = get_op("qwen3_attention")
        self.assertEqual([a.name for a in op.args], ["q", "k", "v", "o_weight"])
        text = op.extra_constraints + op.forward_semantics
        for excluded in ("q_proj", "k_proj", "v_proj", "RMSNorm", "rotary"):
            self.assertIn(excluded, text)
        self.assertIn("separate task", op.extra_constraints)

    def test_gradient_order(self):
        self.assertEqual(
            get_op("qwen3_attention").grad_names(), ("dq", "dk", "dv", "do_weight")
        )

    def test_runtime_forward_is_the_sdpa_spelling(self):
        op = get_op("qwen3_attention")
        self.assertTrue(op.runtime_forward.endswith("qwen3_attention_forward_production"))
        self.assertTrue(op.forward.endswith("qwen3_attention_forward_ref"))

    def test_the_provenance_chain_is_complete(self):
        joined = " | ".join(PROVENANCE_CHAIN)
        for link in (
            "qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad",
            "3ab24571",
            "model.layers.14.self_attn",
            "9674b971ae24b325",
            "ea5311a8e1cbba90",
            "qwen3_attention",
        ):
            self.assertIn(link, joined)

    def test_the_declaration_imports_without_transformers_or_results(self):
        script = (
            "import sys; sys.modules['transformers'] = None;"
            " from evograd.ops import get_op;"
            " print(get_op('qwen3_attention').benchmark[0].dims['HK'])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "8")


class TestInputGeneration(unittest.TestCase):
    """The generated inputs must have the layout the model presented, not a
    contiguous substitute that would benchmark a different access pattern."""

    def _inputs(self, index: int):
        op = get_op("qwen3_attention")
        workload = op.correctness[index]
        return workload, make_case_inputs(op, workload, device="cpu")

    def test_q_k_v_are_non_contiguous_head_major(self):
        for index in range(len(get_op("qwen3_attention").correctness)):
            workload, values = self._inputs(index)
            dims = workload.dims
            with self.subTest(dims=dims):
                for name, heads in (("q", dims["HQ"]), ("k", dims["HK"]), ("v", dims["HK"])):
                    tensor = values[name]
                    self.assertEqual(
                        list(tensor.shape), [dims["B"], heads, dims["T"], dims["D"]]
                    )
                    self.assertFalse(tensor.is_contiguous(), name)
                    self.assertEqual(
                        list(tensor.stride()),
                        [
                            dims["T"] * heads * dims["D"],
                            dims["D"],
                            heads * dims["D"],
                            1,
                        ],
                        name,
                    )

    def test_the_generated_layout_matches_the_observed_pattern(self):
        """Same stride *pattern* as the canonical capture, at the smaller size."""
        workload, values = self._inputs(1)
        dims = workload.dims
        observed_q = OBSERVED_STRIDES["q"]
        # stride[1] is the head dimension and stride[3] is 1 in both.
        self.assertEqual(values["q"].stride()[1], dims["D"])
        self.assertEqual(values["q"].stride()[3], 1)
        self.assertEqual(observed_q[1], 128)
        self.assertEqual(observed_q[3], 1)

    def test_every_correctness_case_preserves_grouped_query_attention(self):
        for workload in get_op("qwen3_attention").correctness:
            dims = workload.dims
            with self.subTest(dims=dims):
                self.assertGreater(dims["HQ"], dims["HK"])
                self.assertEqual(dims["HQ"] % dims["HK"], 0)
                self.assertEqual(dims["QO"], dims["HQ"] * dims["D"])

    def test_both_group_ratios_are_covered(self):
        ratios = {
            w.dims["HQ"] // w.dims["HK"] for w in get_op("qwen3_attention").correctness
        }
        self.assertEqual(ratios, {2, 4})

    def test_both_dtypes_are_covered(self):
        dtypes = {w.dtype for w in get_op("qwen3_attention").correctness}
        self.assertEqual(dtypes, {"float32", "bfloat16"})


class TestForwardReference(unittest.TestCase):
    def _case(self, dtype=torch.float32):
        torch.manual_seed(0)
        B, T, HQ, HK, D, H = 2, 16, 4, 2, 8, 24
        q = torch.randn(B, T, HQ, D, dtype=dtype).transpose(1, 2)
        k = torch.randn(B, T, HK, D, dtype=dtype).transpose(1, 2)
        v = torch.randn(B, T, HK, D, dtype=dtype).transpose(1, 2)
        o_weight = (torch.randn(H, HQ * D) * (HQ * D) ** -0.5).to(dtype)
        return q, k, v, o_weight

    def test_the_two_spellings_agree(self):
        q, k, v, o_weight = self._case()
        dense = qwen3_attention_forward_ref(q, k, v, o_weight)
        production = qwen3_attention_forward_production(q, k, v, o_weight)
        self.assertEqual(dense.shape, production.shape)
        scale = float(dense.abs().max())
        self.assertLess(float((dense - production).abs().max()) / scale, 1e-6)

    def test_attention_is_causal(self):
        """Perturbing a future key must not change an earlier output row."""
        q, k, v, o_weight = self._case()
        k2 = k.clone()
        k2[:, :, -1, :] += 100.0
        for forward in (qwen3_attention_forward_ref, qwen3_attention_forward_production):
            with self.subTest(forward=forward.__name__):
                a = forward(q, k, v, o_weight)
                b = forward(q, k2, v, o_weight)
                self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5))
                self.assertFalse(torch.allclose(a[:, -1], b[:, -1], atol=1e-5))

    def test_grouped_query_broadcast_matches_an_explicit_repeat(self):
        q, k, v, o_weight = self._case()
        groups = q.shape[1] // k.shape[1]
        expanded = qwen3_attention_forward_production(
            q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1), o_weight
        )
        grouped = qwen3_attention_forward_production(q, k, v, o_weight)
        self.assertLess(
            float((expanded - grouped).abs().max()) / float(grouped.abs().max()), 1e-6
        )

    def test_backward_produces_all_four_gradients_in_order(self):
        q, k, v, o_weight = self._case()
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, o_weight)]
        out = qwen3_attention_forward_production(*leaves)
        out.backward(torch.randn_like(out))
        for name, tensor in zip(("dq", "dk", "dv", "do_weight"), leaves):
            with self.subTest(gradient=name):
                self.assertIsNotNone(tensor.grad)
                self.assertEqual(tensor.grad.shape, tensor.shape)
                self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_mismatched_inputs_are_rejected(self):
        q, k, v, o_weight = self._case()
        odd = torch.randn(q.shape[0], 3, q.shape[2], q.shape[3])
        with self.assertRaises(ValueError):  # HQ=4 not divisible by HK=3
            qwen3_attention_forward_ref(q, odd, odd, o_weight)
        with self.assertRaises(ValueError):  # o_weight fan-in wrong
            qwen3_attention_forward_ref(q, k, v, o_weight[:, :-1])
        with self.assertRaises(ValueError):  # k and v disagree
            qwen3_attention_forward_ref(q, k, v[:, :1], o_weight)

    def test_the_declared_reference_uses_the_declared_scale(self):
        """A reference that used a different scale would be self-consistent and
        wrong, so it is checked against the harvested scalar."""
        q, k, v, o_weight = self._case()
        production = qwen3_attention_forward_production(q, k, v, o_weight)
        rescaled = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=1.0 / math.sqrt(q.shape[-1]), enable_gqa=True,
        )
        merged = rescaled.transpose(1, 2).contiguous().reshape(q.shape[0], q.shape[2], -1)
        expected = torch.nn.functional.linear(merged, o_weight)
        self.assertTrue(torch.equal(production, expected))


class TestTimedBaselineAndGate(unittest.TestCase):
    def test_runtime_forward_resolves_to_the_sdpa_spelling(self):
        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
        from evograd.ops.level2.qwen3_attention import forward_ref

        op = get_op("qwen3_attention")
        self.assertIs(
            resolve_runtime_forward(op), forward_ref.qwen3_attention_forward_production
        )
        self.assertIs(resolve_forward(op), forward_ref.qwen3_attention_forward_ref)

    def test_the_eager_baseline_never_materializes_a_dense_score_matrix(self):
        """The declared oracle does; the timed spelling must not."""
        import inspect

        from evograd.ops.level2.qwen3_attention import forward_ref

        dense = inspect.getsource(forward_ref.qwen3_attention_forward_ref)
        timed = inspect.getsource(forward_ref.qwen3_attention_forward_production)
        self.assertIn("masked_fill", dense)
        self.assertIn("softmax", dense)
        self.assertNotIn("masked_fill", timed)
        self.assertNotIn("softmax", timed)
        self.assertIn("scaled_dot_product_attention", timed)

    def test_the_declared_tolerances_are_the_calibrated_ones(self):
        op = get_op("qwen3_attention")
        self.assertEqual(op.tolerances["bfloat16"], (1e-2, 1e-2))
        self.assertEqual(op.tolerances["float32"], (2e-5, 2e-5))
        # do_weight went 5.4 -> 6.5 when the observed shape was measured; see
        # the calibration artifact for the 1.58x it was short by.
        self.assertEqual(op.tolerance_multipliers, {"do_weight": (6.5, 1.0)})

    def test_the_hook_barely_moves_the_correctness_grid(self):
        # The dimension-aware term widens workloads *above* the anchor. One
        # correctness case has a do_weight of 4096 elements against the anchor's
        # 3072, so it picks up 1.04x of the element-count term -- measured, and
        # bounded here so a future anchor change cannot quietly loosen the grid.
        # Its measured margin at that case is 1.74x, so 1.04x costs nothing.
        op = get_op("qwen3_attention")
        for workload in op.correctness:
            for name in (*op.output_names, *op.grad_names()):
                with self.subTest(dims=workload.dims, result=name):
                    base = op.tolerances[workload.dtype]
                    ma, mr = op.tolerance_multipliers.get(name, (1.0, 1.0))
                    atol, rtol = op.tolerance_for(workload, name)
                    self.assertGreaterEqual(atol, base[0] * ma)
                    self.assertLessEqual(atol, base[0] * ma * 1.05)
                    self.assertEqual(rtol, base[1] * mr)

    def test_the_observed_shape_gets_the_element_count_term_only(self):
        # Attention declares no reduction term: the measured exponent of its
        # required atol against reduction length is 0.032, because a softmax
        # -weighted average does not grow with the number of terms.
        from evograd.ops.level2.qwen3_attention import _REDUCTION_SCALED

        op = get_op("qwen3_attention")
        self.assertEqual(_REDUCTION_SCALED.reduction_dims, {})
        observed = op.benchmark_workloads(suite="qwen3_0_6b_observed")[0]
        atol, rtol = op.tolerance_for(observed, "do_weight")
        self.assertGreater(atol, 1e-2 * 6.5)          # widened
        self.assertLess(atol, 1e-2 * 6.5 * 3.0)       # and only modestly
        self.assertEqual(rtol, 1e-2)                  # rtol never moves

    def test_the_real_pair_passes_verify_runtime_forward(self):
        from evograd.opdecl import baselines

        baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_attention")
        baselines.verify_runtime_forward(get_op("qwen3_attention"), device="cpu")

    def test_a_materially_perturbed_implementation_is_rejected(self):
        from evograd.opdecl import baselines
        from evograd.ops.level2.qwen3_attention import forward_ref

        original = forward_ref.qwen3_attention_forward_production

        def perturbed(q, k, v, o_weight):
            # Bound to the original, not to the module attribute: the patch
            # below replaces that attribute.
            return original(q, k, v, o_weight) * 1.02

        forward_ref.qwen3_attention_forward_production = perturbed
        try:
            baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_attention")
            with self.assertRaises(RuntimeError) as ctx:
                baselines.verify_runtime_forward(get_op("qwen3_attention"), device="cpu")
            self.assertIn("disagrees with forward", str(ctx.exception))
        finally:
            forward_ref.qwen3_attention_forward_production = original
            baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_attention")


class TestSnapshotEntry(unittest.TestCase):
    def test_the_attention_entry_is_derived_not_hand_written(self):
        payload = snapshot_module.load()
        self.assertIn("qwen3_attention", payload["tasks"])
        entry = payload["tasks"]["qwen3_attention"]
        self.assertEqual(entry["harvested_task"], "sdpa")
        self.assertEqual(len(entry["input_shapes"]), 3)
        self.assertEqual(entry["frequency"], 28)
        self.assertEqual(sorted(entry["supporting"]), ["enclosing_attention", "output_projection"])

    def test_the_snapshot_hash_still_verifies(self):
        payload = snapshot_module.load()
        self.assertEqual(payload["snapshot_hash"], snapshot_module.snapshot_hash(payload))


if __name__ == "__main__":
    unittest.main()
