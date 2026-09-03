"""The dimension-aware tolerance hook, on its own.

A tolerance that grows with the workload is a claim about numerics, so the
shape of the growth has to be checkable without a GPU. What these pin is the
one property that made it safe to add to already-calibrated declarations: it is
the identity at and below the anchor, so nothing that passed before becomes
easier to pass.
"""

from __future__ import annotations

import math
import unittest

from evograd.opdecl.tolerance import ReductionScaledAtol


class _Workload:
    def __init__(self, **dims):
        self.dims = dims


def _hook(**overrides):
    base = dict(
        anchor_dims={"B": 2, "T": 16, "H": 32, "I": 96},
        reduction_dims={"dw": ("B", "T")},
        result_dims={"y": ("B", "T", "H"), "dw": ("I", "H")},
        gain=2.0,
    )
    base.update(overrides)
    return ReductionScaledAtol(**base)


class TestTheAnchorIsUntouched(unittest.TestCase):
    def test_the_anchor_workload_gets_exactly_the_declared_tolerance(self):
        hook = _hook()
        anchor = _Workload(**hook.anchor_dims)
        for name in ("y", "dw"):
            with self.subTest(result=name):
                self.assertEqual(hook(anchor, name, 0.01, 0.02), (0.01, 0.02))

    def test_a_smaller_workload_is_never_loosened(self):
        hook = _hook()
        smaller = _Workload(B=1, T=8, H=16, I=48)
        for name in ("y", "dw"):
            with self.subTest(result=name):
                atol, _rtol = hook(smaller, name, 0.01, 0.02)
                self.assertEqual(atol, 0.01)

    def test_a_smaller_workload_is_not_tightened_either(self):
        # Tightening would be a stricter gate, but it would also fail cases the
        # declaration was calibrated to accept. The clamp is deliberate.
        hook = _hook()
        self.assertEqual(hook.factor("dw", {"B": 1, "T": 1, "H": 4, "I": 4}), 1.0)


class TestTheGrowthLaw(unittest.TestCase):
    def test_a_longer_reduction_widens_atol(self):
        hook = _hook()
        small = hook.factor("dw", {"B": 2, "T": 16, "H": 32, "I": 96})
        large = hook.factor("dw", {"B": 2, "T": 2048, "H": 1024, "I": 3072})
        self.assertEqual(small, 1.0)
        self.assertGreater(large, 10.0)

    def test_the_random_walk_term_is_the_square_root_of_the_reduction(self):
        # Widths held fixed so only the reduction moves: 16x the tokens must be
        # 4x the raw factor, which is the claim the law makes.
        hook = _hook(result_dims={"dw": ("I", "H")})
        base = {"B": 2, "T": 16, "H": 32, "I": 96}
        wide = {**base, "T": 256}
        self.assertAlmostEqual(hook.raw_factor("dw", wide), 4.0, places=6)

    def test_the_element_count_term_is_the_square_root_of_its_log(self):
        # No reduction declared, so only extreme-value growth is left.
        hook = _hook(reduction_dims={})
        anchor_m = 2 * 16 * 32
        dims = {"B": 2, "T": 2048, "H": 1024, "I": 96}
        expected = math.sqrt(math.log(2 * 2048 * 1024) / math.log(anchor_m))
        self.assertAlmostEqual(hook.raw_factor("y", dims), expected, places=6)

    def test_a_result_with_no_reduction_grows_far_less(self):
        hook = _hook()
        observed = {"B": 2, "T": 2048, "H": 1024, "I": 3072}
        self.assertLess(hook.factor("y", observed), 2.5)
        self.assertGreater(hook.factor("dw", observed), 10.0)

    def test_gain_multiplies_the_excess_not_the_tolerance(self):
        one = _hook(gain=1.0)
        two = _hook(gain=2.0)
        dims = {"B": 2, "T": 2048, "H": 1024, "I": 3072}
        raw = one.raw_factor("dw", dims)
        self.assertAlmostEqual(one.factor("dw", dims), raw)
        self.assertAlmostEqual(two.factor("dw", dims), 1 + 2 * (raw - 1))
        # ... and neither touches the anchor.
        self.assertEqual(one.factor("dw", one.anchor_dims), 1.0)
        self.assertEqual(two.factor("dw", two.anchor_dims), 1.0)


class TestWhatItRefusesToTouch(unittest.TestCase):
    def test_rtol_is_never_scaled(self):
        hook = _hook()
        dims = _Workload(B=2, T=2048, H=1024, I=3072)
        for name in ("y", "dw"):
            with self.subTest(result=name):
                _atol, rtol = hook(dims, name, 0.01, 0.02)
                self.assertEqual(rtol, 0.02)

    def test_an_unnamed_result_is_left_alone(self):
        hook = _hook()
        dims = _Workload(B=2, T=2048, H=1024, I=3072)
        self.assertEqual(hook(dims, None, 0.01, 0.02), (0.01, 0.02))

    def test_a_result_absent_from_the_maps_gets_the_identity(self):
        hook = _hook()
        dims = _Workload(B=2, T=2048, H=1024, I=3072)
        self.assertEqual(hook(dims, "unknown", 0.01, 0.02), (0.01, 0.02))

    def test_a_single_element_result_has_no_extreme_value_term(self):
        # log(1) is zero; the term is dropped rather than dividing by it.
        hook = _hook(reduction_dims={}, result_dims={"loss": ()})
        self.assertEqual(hook.raw_factor("loss", {"B": 2, "T": 2048}), 1.0)


class TestTheQwenDeclarationsUseIt(unittest.TestCase):
    def test_each_declaration_names_its_anchor_and_its_reductions(self):
        try:
            from evograd.ops import get_op
        except Exception:  # pragma: no cover
            self.skipTest("torch not installed")

        expected = {
            "qwen3_swiglu_mlp": {"dgate_weight", "dup_weight", "ddown_weight"},
            "qwen3_qkv_norm_rope": {
                "dq_weight", "dk_weight", "dv_weight",
                "dq_norm_weight", "dk_norm_weight",
            },
            "qwen3_attention": set(),
        }
        for name, reductions in expected.items():
            with self.subTest(op=name):
                op = get_op(name)
                hook = op.tolerance_hook
                self.assertIsInstance(hook, ReductionScaledAtol)
                self.assertEqual(set(hook.reduction_dims), reductions)
                # The anchor must be one of the declaration's own correctness
                # cases, or "the workload the multipliers were measured on" is
                # a claim about a workload that does not exist.
                self.assertIn(
                    hook.anchor_dims, [dict(w.dims) for w in op.correctness]
                )

    def test_every_declared_result_is_covered_by_the_shape_map(self):
        try:
            from evograd.ops import get_op
        except Exception:  # pragma: no cover
            self.skipTest("torch not installed")

        for name in ("qwen3_swiglu_mlp", "qwen3_qkv_norm_rope", "qwen3_attention"):
            with self.subTest(op=name):
                op = get_op(name)
                declared = {*op.output_names, *op.grad_names()}
                self.assertEqual(set(op.tolerance_hook.result_dims), declared)

    def test_the_hook_is_serializable_for_the_artifact(self):
        try:
            from evograd.ops import get_op
        except Exception:  # pragma: no cover
            self.skipTest("torch not installed")

        import json

        described = get_op("qwen3_swiglu_mlp").tolerance_hook.describe()
        self.assertEqual(described["kind"], "reduction_scaled_atol")
        self.assertIn("sqrt(N/N_a)", described["formula"])
        json.dumps(described)  # must round-trip into the calibration artifact


if __name__ == "__main__":
    unittest.main()
