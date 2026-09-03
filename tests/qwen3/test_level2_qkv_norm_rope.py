"""Level-2 ``qwen3_qkv_norm_rope``: the observed projection + head-norm + RoPE prefix.

The first registered operator with structured outputs, so these tests check the
contract twice over: that the boundary is the one that was observed, and that
the three outputs are treated as three things rather than one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.bench.workloads.qwen3.harvest import snapshot as snapshot_module
from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values
from evograd.opdecl.models import rederive_dims
from evograd.opdecl.oracle import oracle
from evograd.ops import OPS, get_op
from evograd.ops.level2.qwen3_qkv_norm_rope import (
    FREQUENCY,
    HARVEST,
    OBSERVED_STRIDES,
    PROVENANCE_CHAIN,
)
from evograd.ops.level2.qwen3_qkv_norm_rope.forward_ref import (
    qwen3_qkv_norm_rope_forward_production,
    qwen3_qkv_norm_rope_forward_ref,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OP = get_op("qwen3_qkv_norm_rope")


class TestDeclaration(unittest.TestCase):
    def test_registered_at_level_two_with_three_outputs(self):
        self.assertIn("qwen3_qkv_norm_rope", OPS)
        self.assertEqual((OP.level, OP.family), (2, "attention"))
        self.assertTrue(OP.is_multi_output)
        self.assertEqual(OP.output_names, ("q", "k", "v"))
        self.assertEqual(OP.upstream_grad_names, ("dq", "dk", "dv"))

    def test_the_canonical_contract(self):
        case = OP.benchmark[0]
        self.assertEqual(
            case.dims,
            {"B": 2, "T": 2048, "H": 1024, "HQ": 16, "HK": 8, "D": 128, "QO": 2048, "KVO": 1024},
        )
        self.assertEqual(case.dtype, "bfloat16")
        shapes = {o.name: o.shape for o in OP.outputs}
        self.assertEqual(shapes["q"], "[B, HQ, T, D]")
        self.assertEqual(shapes["k"], "[B, HK, T, D]")
        self.assertEqual(shapes["v"], "[B, HK, T, D]")

    def test_gradient_order(self):
        self.assertEqual(
            OP.grad_names(),
            ("dx", "dq_weight", "dk_weight", "dv_weight", "dq_norm_weight", "dk_norm_weight"),
        )

    def test_cos_and_sin_are_inactive_and_get_no_gradient(self):
        by_name = {a.name: a for a in OP.args}
        for name in ("cos", "sin"):
            self.assertEqual(type(by_name[name]).__name__, "Inactive")
            self.assertTrue(by_name[name].is_tensor)
        self.assertNotIn("dcos", OP.grad_names())
        self.assertNotIn("dsin", OP.grad_names())

    def test_eps_is_the_observed_value(self):
        by_name = {a.name: a for a in OP.args}
        self.assertEqual(by_name["eps"].default, 1e-6)
        self.assertEqual(HARVEST["supporting"]["q_norm"]["attrs"]["eps"], 1e-6)

    def test_provenance_is_recomputable_from_the_published_config(self):
        case = OP.benchmark[0]
        self.assertEqual(case.provenance.source, "hf_config")
        self.assertEqual(case.provenance.model, "qwen3_0_6b")
        self.assertEqual(case.dims, rederive_dims(case.provenance))

    def test_the_declaration_agrees_with_the_snapshot(self):
        from evograd.bench.workloads.qwen3.levels.level2 import qkv_norm_rope as qkv_module

        self.assertEqual(qkv_module.declaration_problems(), [])

    def test_the_harvested_records_are_carried_structurally(self):
        self.assertEqual(FREQUENCY, 28)
        self.assertEqual(HARVEST["config_id"], "34378ecf454fc895")
        supporting = HARVEST["supporting"]
        self.assertEqual(supporting["q_projection"]["config_id"], "b9cc24095ee7dc62")
        self.assertEqual(supporting["kv_projection"]["config_id"], "be060ca58cf90863")
        self.assertEqual(supporting["q_norm"]["config_id"], "a872639f0512398e")
        self.assertEqual(supporting["k_norm"]["config_id"], "494b2d469ae06000")
        self.assertEqual(supporting["q_projection"]["frequency"], 28)
        # k_proj and v_proj deduplicate into one configuration, so it ran twice
        # per layer; the roles record which two.
        self.assertEqual(supporting["kv_projection"]["frequency"], 56)
        self.assertEqual(supporting["kv_projection"]["roles"], ["k_proj", "v_proj"])

    def test_the_observed_output_strides(self):
        self.assertEqual(OBSERVED_STRIDES["q"], (4194304, 128, 2048, 1))
        self.assertEqual(OBSERVED_STRIDES["k"], (2097152, 128, 1024, 1))
        self.assertEqual(OBSERVED_STRIDES["v"], OBSERVED_STRIDES["k"])

    def test_the_boundary_excludes_attention_and_the_output_projection(self):
        text = OP.extra_constraints + OP.forward_semantics
        self.assertIn("scaled_dot_product_attention", text)
        self.assertIn("o_proj", OP.extra_constraints)
        self.assertIn("qwen3_attention", OP.extra_constraints)
        self.assertIn("residual RMSNorm", OP.extra_constraints)

    def test_the_two_attention_tasks_meet_exactly(self):
        """This task's outputs are the other task's inputs, shape for shape."""
        other = get_op("qwen3_attention").benchmark[0].dims
        mine = OP.benchmark[0].dims
        for dim in ("B", "T", "HQ", "HK", "D"):
            self.assertEqual(mine[dim], other[dim], dim)
        self.assertEqual(mine["QO"], other["QO"])

    def test_the_declaration_imports_without_transformers_or_results(self):
        script = (
            "import sys; sys.modules['transformers'] = None;"
            " from evograd.ops import get_op;"
            " op = get_op('qwen3_qkv_norm_rope');"
            " print(len(op.outputs), op.benchmark[0].dims['KVO'])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "3 1024")


class TestInputGeneration(unittest.TestCase):
    def test_every_correctness_case_covers_the_contract(self):
        dtypes = {w.dtype for w in OP.correctness}
        ratios = {w.dims["HQ"] // w.dims["HK"] for w in OP.correctness}
        self.assertEqual(dtypes, {"float32", "bfloat16"})
        self.assertEqual(ratios, {2, 4})
        for workload in OP.correctness:
            dims = workload.dims
            with self.subTest(dims=dims):
                self.assertEqual(dims["HQ"] % dims["HK"], 0)
                self.assertEqual(dims["QO"], dims["HQ"] * dims["D"])
                self.assertEqual(dims["KVO"], dims["HK"] * dims["D"])

    def test_one_upstream_gradient_is_generated_per_output(self):
        values = make_case_inputs(OP, OP.correctness[0], device="cpu")
        grads = upstream_grad_values(OP, values)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 3)
        dims = OP.correctness[0].dims
        self.assertEqual(list(grads[0].shape), [dims["B"], dims["HQ"], dims["T"], dims["D"]])
        self.assertEqual(list(grads[1].shape), [dims["B"], dims["HK"], dims["T"], dims["D"]])

    def test_the_rotary_tables_are_real_rotations(self):
        """A random cos/sin pair is not a rotation, and a tolerance calibrated
        against one would describe a computation the model never performs."""
        values = make_case_inputs(OP, OP.correctness[1], device="cpu")
        cos, sin = values["cos"].float(), values["sin"].float()
        self.assertTrue(torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5))
        self.assertTrue(torch.allclose(cos[0, 0], torch.ones_like(cos[0, 0]), atol=1e-5))


class TestForwardReference(unittest.TestCase):
    def _case(self, dtype=torch.float32, heads=4, kv_heads=2, head_dim=8, hidden=32, tokens=16):
        torch.manual_seed(0)
        x = torch.randn(2, tokens, hidden, dtype=dtype)
        def proj(out):
            return (torch.randn(out, hidden) * hidden**-0.5).to(dtype)
        angles = torch.arange(tokens, dtype=torch.float32)[:, None] * torch.arange(
            1, head_dim // 2 + 1, dtype=torch.float32
        )[None, :]
        table = torch.cat((angles, angles), dim=-1)[None]
        return (
            x,
            proj(heads * head_dim),
            proj(kv_heads * head_dim),
            proj(kv_heads * head_dim),
            torch.ones(head_dim, dtype=dtype),
            torch.ones(head_dim, dtype=dtype),
            table.cos().to(dtype),
            table.sin().to(dtype),
        )

    def test_the_two_spellings_agree_and_share_the_layout(self):
        args = self._case()
        ref = qwen3_qkv_norm_rope_forward_ref(*args)
        prod = qwen3_qkv_norm_rope_forward_production(*args)
        self.assertEqual(len(ref), 3)
        for name, a, b in zip("qkv", ref, prod):
            with self.subTest(output=name):
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.stride(), b.stride())
                self.assertLess(float((a - b).abs().max()) / float(b.abs().max()), 1e-6)

    def test_the_outputs_are_non_contiguous_head_major(self):
        q, k, v = qwen3_qkv_norm_rope_forward_production(*self._case())
        for name, tensor, heads in (("q", q, 4), ("k", k, 2), ("v", v, 2)):
            with self.subTest(output=name):
                self.assertFalse(tensor.is_contiguous())
                b, h, t, d = tensor.shape
                self.assertEqual(list(tensor.stride()), [t * heads * d, d, heads * d, 1])

    def test_v_is_neither_normalized_nor_rotated(self):
        """The value path is a plain projection; the reference must not touch it."""
        args = list(self._case())
        _q, _k, v_before = qwen3_qkv_norm_rope_forward_production(*args)
        args[7] = torch.zeros_like(args[7])  # sin = 0, so RoPE becomes identity
        args[5] = args[5] * 3.0  # k_norm scale
        _q2, _k2, v_after = qwen3_qkv_norm_rope_forward_production(*args)
        self.assertTrue(torch.equal(v_before, v_after))

    def test_rope_is_applied_to_q_and_k(self):
        args = list(self._case())
        q_before, k_before, _v = qwen3_qkv_norm_rope_forward_production(*args)
        args[7] = torch.zeros_like(args[7])
        args[6] = torch.ones_like(args[6])
        q_after, k_after, _v = qwen3_qkv_norm_rope_forward_production(*args)
        self.assertFalse(torch.allclose(q_before, q_after))
        self.assertFalse(torch.allclose(k_before, k_after))

    def test_the_head_norm_uses_the_declared_epsilon(self):
        args = self._case()
        a = qwen3_qkv_norm_rope_forward_production(*args, eps=1e-6)
        b = qwen3_qkv_norm_rope_forward_production(*args, eps=1.0)
        self.assertFalse(torch.allclose(a[0], b[0]))
        self.assertTrue(torch.equal(a[2], b[2]))  # v is unaffected

    def test_mismatched_inputs_are_rejected(self):
        args = list(self._case())
        with self.assertRaises(ValueError):  # HQ=4 not divisible by HK=3
            bad = list(args)
            odd = torch.randn(3 * 8, args[0].shape[-1]) * args[0].shape[-1] ** -0.5
            bad[2] = odd
            bad[3] = odd
            qwen3_qkv_norm_rope_forward_ref(*bad)
        with self.assertRaises(ValueError):  # cos width != head_dim
            bad = list(args)
            bad[6] = bad[6][..., :-2]
            bad[7] = bad[7][..., :-2]
            qwen3_qkv_norm_rope_forward_ref(*bad)

    def test_backward_returns_six_gradients_and_uses_all_three(self):
        values = make_case_inputs(OP, OP.correctness[1], device="cpu")
        outputs, grads = oracle(OP, values)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(sorted(grads), sorted(OP.grad_names()))
        for name, grad in grads.items():
            with self.subTest(gradient=name):
                self.assertTrue(torch.isfinite(grad).all())
        # Zeroing dv must change dx but leave dq_weight alone.
        values["dv"] = torch.zeros_like(values["dv"])
        _out, partial = oracle(OP, values)
        self.assertFalse(torch.allclose(grads["dx"], partial["dx"]))
        self.assertTrue(torch.allclose(grads["dq_weight"], partial["dq_weight"]))
        self.assertTrue(torch.equal(partial["dv_weight"], torch.zeros_like(partial["dv_weight"])))


class TestTimedBaselineAndGate(unittest.TestCase):
    def test_runtime_forward_resolves_to_the_hf_spelling(self):
        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
        from evograd.ops.level2.qwen3_qkv_norm_rope import forward_ref

        self.assertIs(
            resolve_runtime_forward(OP), forward_ref.qwen3_qkv_norm_rope_forward_production
        )
        self.assertIs(resolve_forward(OP), forward_ref.qwen3_qkv_norm_rope_forward_ref)

    def test_the_declared_tolerances_are_the_calibrated_ones(self):
        self.assertEqual(OP.tolerances["bfloat16"], (2e-2, 2e-2))
        self.assertEqual(OP.tolerances["float32"], (2e-5, 2e-5))
        self.assertEqual(
            OP.tolerance_multipliers,
            {
                "dx": (1.6, 1.0),
                "dq_weight": (7.1, 1.0),
                "dk_weight": (5.4, 1.0),
                "dq_norm_weight": (7.4, 1.0),
                "dk_norm_weight": (4.2, 1.0),
            },
        )

    def test_the_outputs_carry_no_multiplier(self):
        # q, k and v have no reduction and no measured need for one, so no
        # multiplier. At the observed shape they still pick up the hook's
        # element-count term, which is a property of the workload rather than
        # of the result.
        for name in OP.output_names:
            self.assertNotIn(name, OP.tolerance_multipliers)

    def test_the_hook_is_the_identity_on_the_correctness_grid(self):
        for workload in OP.correctness:
            for name in (*OP.output_names, *OP.grad_names()):
                with self.subTest(dims=workload.dims, result=name):
                    base = OP.tolerances[workload.dtype]
                    ma, mr = OP.tolerance_multipliers.get(name, (1.0, 1.0))
                    self.assertEqual(
                        OP.tolerance_for(workload, name),
                        (base[0] * ma, base[1] * mr),
                    )

    def test_the_observed_shape_widens_the_reduction_gradients_most(self):
        # The four weight gradients contract over 4096 tokens where the grid's
        # longest is 64; q/k/v contract over nothing. The gate has to separate
        # them, or a constant is doing the work of a law.
        observed = OP.benchmark_workloads(suite="qwen3_0_6b_observed")[0]
        base_atol = OP.tolerances["bfloat16"][0]
        widening = {
            name: OP.tolerance_for(observed, name)[0]
            / (base_atol * OP.tolerance_multipliers.get(name, (1.0, 1.0))[0])
            for name in (*OP.output_names, *OP.grad_names())
        }
        for name in ("q", "k", "v", "dx"):
            self.assertLess(widening[name], 2.5, name)
        for name in ("dq_weight", "dk_weight", "dq_norm_weight", "dk_norm_weight"):
            self.assertGreater(widening[name], 10.0, name)
        # rtol is untouched everywhere: relative error is what stays constant.
        for name in (*OP.output_names, *OP.grad_names()):
            self.assertEqual(OP.tolerance_for(observed, name)[1], 2e-2)

    def test_the_real_pair_passes_verify_runtime_forward(self):
        from evograd.opdecl import baselines

        baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)
        baselines.verify_runtime_forward(OP, device="cpu")

    def test_a_perturbed_production_spelling_is_rejected_per_output(self):
        from evograd.opdecl import baselines
        from evograd.ops.level2.qwen3_qkv_norm_rope import forward_ref

        original = forward_ref.qwen3_qkv_norm_rope_forward_production

        def perturbed(*args, **kwargs):
            q, k, v = original(*args, **kwargs)
            return q, k * 1.05, v  # only the middle output is wrong

        forward_ref.qwen3_qkv_norm_rope_forward_production = perturbed
        try:
            baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)
            with self.assertRaises(RuntimeError) as ctx:
                baselines.verify_runtime_forward(OP, device="cpu")
            message = str(ctx.exception)
            self.assertIn("output 'k'", message)
        finally:
            forward_ref.qwen3_qkv_norm_rope_forward_production = original
            baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)


class TestSnapshotEntry(unittest.TestCase):
    def test_the_entry_is_derived_from_five_harvested_records(self):
        entry = snapshot_module.load()["tasks"]["qwen3_qkv_norm_rope"]
        self.assertEqual(entry["harvested_task"], "rope_apply")
        self.assertEqual(len(entry["output_shapes"]), 2)  # rope returns q and k
        self.assertEqual(
            sorted(entry["supporting"]),
            ["consumer", "enclosing_attention", "k_norm", "kv_projection", "q_norm", "q_projection"],
        )

    def test_v_comes_from_where_it_is_consumed(self):
        """v never passes through RoPE, so its layout is sourced from SDPA."""
        entry = snapshot_module.load()["tasks"]["qwen3_qkv_norm_rope"]
        v = entry["supporting"]["consumer"]["input_shapes"][2]
        self.assertEqual(v["shape"], [2, 8, 2048, 128])
        self.assertEqual(v["stride"], [2097152, 128, 1024, 1])

    def test_the_snapshot_hash_still_verifies(self):
        payload = snapshot_module.load()
        self.assertEqual(payload["snapshot_hash"], snapshot_module.snapshot_hash(payload))


if __name__ == "__main__":
    unittest.main()
