"""Llama-3-8B's SwiGLU task, and the `out` multiplier its width requires.

These assertions came from the upstream Llama-3 work, where they were written
against ``qwen3_swiglu_mlp`` because that task then served Llama's MLP site.
Llama now owns ``llama3_swiglu_mlp``, so they moved here with the multiplier
they describe. The numbers are unchanged: the measured shortfall at the
8192-wide intermediate, the bounds on the multiplier that covers it, and what
that widening costs the correctness grid.

``tests/qwen3/test_level2_swiglu_mlp`` holds the other half of the separation:
that Qwen3's ``out`` gate has *not* been widened to pay for this shape.
"""

import unittest

from evograd.benchmark import get_task
from evograd.opdecl.inputs import make_case_inputs

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class LlamaSwigluTolerances(unittest.TestCase):
    def test_the_declared_tolerances_carry_the_wide_shape(self):
        op = get_task("llama3_swiglu_mlp")
        self.assertEqual(op.tolerances["bfloat16"], (1e-2, 1e-2))
        self.assertEqual(op.tolerances["float32"], (2e-5, 2e-5))
        self.assertEqual(
            op.tolerance_multipliers,
            {
                "out": (2.4, 2.4),
                "dx": (2.3, 1.0),
                "dgate_weight": (3.7, 1.0),
                "dup_weight": (4.9, 1.0),
                "ddown_weight": (6.5, 1.0),
            },
        )

def test_the_out_multiplier_is_the_measured_shortfall_at_the_wide_shape(self):
        """`out`'s multiplier exists for one measurement, and covers it once.

        `out` sums over `I`, and the hook does not model that: its `result_dims`
        count output *elements* and `out` has no `reduction_dims` entry, so all
        the hook supplies at Llama-3-8B's shape is the extreme-value term. The
        multiplier carries the rest. Bounded on both sides, because a multiplier
        that covers its measurement by 10x is a hole rather than a gate.
        """
        op = get_task("llama3_swiglu_mlp")
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2.swiglu_mlp.task import _REDUCTION_SCALED

        self.assertNotIn("out", _REDUCTION_SCALED.reduction_dims)
        observed = op.benchmark_workloads(suite="llama_3_2_1b_observed")[0]
        self.assertEqual(observed.dims["I"], 8192)

        # What the hook alone supplies, before the multiplier: 2.098e-02.
        hook_only = _REDUCTION_SCALED.factor("out", dict(observed.dims))
        self.assertAlmostEqual(hook_only * 1e-2, 2.098e-02, places=5)

        # And what the harvested invocation needed there.
        required = 3.242e-02
        atol, rtol = op.tolerance_for(observed, "out")
        self.assertGreater(atol, required * 1.5)   # the declared safety margin
        self.assertLess(atol, required * 3.0)      # and no more than that
        # `out` is the only result in this operator -- and the only row in
        # `level2.negative_controls` -- whose rtol is not the declared base.
        # The multiplier was measured as an atol shortfall, and the calibration
        # reports it as a single base `t` used as `allclose(atol=ma*t, rtol=t)`,
        # which is how the rtol half got carried along. It is not free: the
        # scaled fault is the one `rtol` exists to catch, and at 2.4e-2 the
        # `out` detection floor is 5.0% against 0.5-2.0% everywhere else in
        # that report. Pinned here so the anomaly is visible in the gate rather
        # than only in the artifact.
        self.assertEqual(rtol, 1e-2 * 2.4)
        for name in op.grad_names():
            self.assertEqual(op.tolerance_for(observed, name)[1], 1e-2)

def test_what_the_out_multiplier_costs_the_correctness_grid(self):
        """The multiplier is global, so the shrunk grid pays for it too.

        `out`'s gate on the correctness cases went 1e-2 -> 2.4e-2 in both halves
        when the wide shape was measured, and that grid is what
        `verify_runtime_forward` checks. What survives that widening is measured
        here rather than assumed: the smallest uniform error in the SwiGLU
        intermediate that the `out` gate alone still rejects. It is bounded, so
        a future widening of this multiplier cannot quietly blind the check --
        `verify_runtime_forward` would still fail on the float32 cases, where
        the two spellings are bit-identical, but `out`'s bfloat16 gate is the
        one this multiplier moves.
        """
        import torch

        from evograd.benchmark.topdown.llama3_2_1b.levels.level2.swiglu_mlp import reference as forward_ref

        op = get_task("llama3_swiglu_mlp")
        case = next(w for w in op.correctness if w.dtype == "bfloat16")
        atol, rtol = op.tolerance_for(case, "out")
        self.assertEqual((atol, rtol), (1e-2 * 2.4, 1e-2 * 2.4))
        # Still tighter than the 8e-2 this operator's gate replaced.
        self.assertLess(max(atol, rtol), 8e-2)

        values = make_case_inputs(op, case, device="cpu")
        args = (
            values["x"],
            values["gate_weight"],
            values["up_weight"],
            values["down_weight"],
        )
        reference = forward_ref.llama3_swiglu_mlp_forward_ref(*args).float()

        def rejected(scale):
            gate = torch.nn.functional.linear(args[0], args[1])
            up = torch.nn.functional.linear(args[0], args[2])
            hidden = (
                torch.nn.functional.silu(gate.float()) * up.float() * scale
            ).to(args[0].dtype)
            wrong = torch.nn.functional.linear(hidden, args[3]).float()
            return not torch.allclose(wrong, reference, atol=atol, rtol=rtol)

        # A plainly broken intermediate is still caught.
        self.assertTrue(rejected(1.10), "a 10% error in the intermediate escapes `out`")
        # And the smallest fault it catches is a fault, not a rounding.
        smallest = next(
            (pct for pct in range(1, 11) if rejected(1.0 + pct / 100.0)), None
        )
        self.assertIsNotNone(smallest)
        self.assertLessEqual(smallest, 10)


if __name__ == "__main__":
    unittest.main()
