"""Structured outputs through the tier-1 and tier-2 protocols.

The coworker's evaluation architecture was written against one output and one
upstream gradient. Every declaration here has to keep working unchanged, and the
three multi-output ones have to work at all -- so the same assertions are made
for both arities side by side, which is the only way to see that the
single-output path was generalized rather than replaced.

Everything runs on CPU: what is under test is the wiring -- which arguments
become module state, which gradients reach which slot, what a report names --
not a kernel's speed. The timing halves of both tiers need a GPU and are
exercised separately.
"""

from __future__ import annotations

import json
import unittest

try:  # torch is absent on some dev boxes; the whole file needs it
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

if HAVE_TORCH:
    from evograd.bench.integrated import activation_and_parameter_args
    from evograd.bench.provider import (
        PairProvider,
        _grad_outputs,
        assert_tensors_unchanged,
        pytorch_autograd_provider,
        renamed_provider,
        snapshot_tensors,
        verify_pair_provider,
    )
    from evograd.bench.report import from_tier2_report
    from evograd.bench.tier2 import (
        OperatorModule,
        build_parameters,
        check_module,
        eager_module,
        identity_control_specs,
    )
    from evograd.opdecl.inputs import upstream_grad_values
    from evograd.ops import get_op

#: (op, declared outputs). One single-output operator of each interesting shape,
#: and every multi-output operator there is.
SINGLE_OUTPUT = ("layernorm", "qwen3_attention", "causal_gqa_attention")
MULTI_OUTPUT = ("qwen3_qkv_norm_rope", "fused_add_rms_norm")


def _case(name):
    """An operator, its smallest declared correctness shape, and CPU inputs."""
    op = get_op(name)
    workload = op.correctness[0]
    return op, workload, build_parameters(op, workload, device="cpu")


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestParameterRoles(unittest.TestCase):
    """Which declared arguments an ``nn.Module`` holds, and which it is called with."""

    def test_attention_takes_qkv_and_owns_only_the_output_projection(self):
        op, _workload, values = _case("qwen3_attention")
        self.assertEqual(op.parameter_args, ("o_weight",))
        module = eager_module(op, values)
        held = dict(module.named_parameters())
        self.assertEqual(sorted(held), ["o_weight"])
        # q, k and v arrive from the previous boundary every step. Holding one
        # of them as module state would measure a layer that keeps its own
        # queries, which is not the operator the declaration describes.
        self.assertEqual(list(module._activation_names), ["q", "k", "v"])

    def test_qkv_weights_are_parameters_and_the_rotary_tables_are_buffers(self):
        op, _workload, values = _case("qwen3_qkv_norm_rope")
        module = eager_module(op, values)
        self.assertEqual(
            sorted(dict(module.named_parameters())),
            ["k_norm_weight", "k_weight", "q_norm_weight", "q_weight", "v_weight"],
        )
        # cos/sin take no gradient and are not optimized; they are precomputed
        # tables the module carries, which is what a buffer is for.
        buffers = dict(module.named_buffers())
        self.assertEqual(sorted(buffers), ["_inactive_cos", "_inactive_sin"])
        for name in ("cos", "sin"):
            self.assertNotIn(name, dict(module.named_parameters()))
        self.assertEqual(list(module._activation_names), ["x"])

    def test_causal_gqa_attention_owns_no_parameters(self):
        op, _workload, values = _case("causal_gqa_attention")
        # `()` is a positive statement, not a missing declaration: the tier can
        # measure it, and the framework cost it charges is paid either way.
        self.assertEqual(op.parameter_args, ())
        module = eager_module(op, values)
        self.assertEqual(list(module.parameters()), [])
        self.assertEqual(list(module._activation_names), ["q", "k", "v"])

    def test_the_residual_stream_is_an_activation(self):
        op, _workload, values = _case("fused_add_rms_norm")
        module = eager_module(op, values)
        self.assertEqual(sorted(dict(module.named_parameters())), ["weight"])
        self.assertEqual(list(module._activation_names), ["x", "r"])

    def test_both_tier_two_and_the_integrated_step_read_the_same_split(self):
        for name in (*SINGLE_OUTPUT, *MULTI_OUTPUT):
            with self.subTest(op=name):
                op, _workload, values = _case(name)
                module = eager_module(op, values)
                activations, parameters = activation_and_parameter_args(op)
                self.assertEqual(
                    [a.name for a in activations], list(module._activation_names)
                )
                self.assertEqual(
                    [p.name for p in parameters], list(module._parameter_names)
                )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestTierOnePairProtocol(unittest.TestCase):
    """The direct pair: providers, per-output gates, identity, mutation."""

    def _verify(self, name):
        op, workload, _values = _case(name)
        provider = pytorch_autograd_provider(op)
        return op, verify_pair_provider(op, provider, (workload,), device="cpu")

    def test_a_single_output_operator_is_gated_exactly_as_before(self):
        op, report = self._verify("layernorm")
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(
            [c.name for c in report.cases[0].checks],
            ["y", "dx", "dweight", "dbias"],
        )
        self.assertFalse(op.is_multi_output)

    def test_three_outputs_are_each_gated_under_their_declared_name(self):
        op, report = self._verify("qwen3_qkv_norm_rope")
        self.assertTrue(report.ok, report.to_dict())
        names = [c.name for c in report.cases[0].checks]
        self.assertEqual(names[:3], ["q", "k", "v"])
        self.assertEqual(names[3:], list(op.grad_names()))

    def test_the_residual_sum_is_gated_alongside_the_normalized_output(self):
        _op, report = self._verify("fused_add_rms_norm")
        self.assertTrue(report.ok, report.to_dict())
        names = [c.name for c in report.cases[0].checks]
        self.assertEqual(names[:2], ["out", "summed"])

    def test_the_upstream_gradients_match_the_declared_outputs(self):
        for name in MULTI_OUTPUT:
            with self.subTest(op=name):
                op, _workload, values = _case(name)
                grads = upstream_grad_values(op, values)
                self.assertIsInstance(grads, tuple)
                self.assertEqual(len(grads), len(op.outputs))
                for out, grad in zip(op.outputs, grads):
                    self.assertIs(grad, values[out.grad_name])
        for name in SINGLE_OUTPUT:
            with self.subTest(op=name):
                op, _workload, values = _case(name)
                # Still a bare Tensor, so nothing written against the old ABI
                # has to learn about tuples.
                self.assertTrue(torch.is_tensor(upstream_grad_values(op, values)))

    def test_a_provider_returning_one_of_three_outputs_is_refused(self):
        op, workload, _values = _case("qwen3_qkv_norm_rope")
        honest = pytorch_autograd_provider(op)

        def forward(values):
            outputs, saved = honest.forward(values)
            return outputs[0], saved  # drops k and v

        truncated = PairProvider(
            name="truncated",
            forward=forward,
            backward=honest.backward,
            source_hash=honest.source_hash,
            adapter_kind=honest.adapter_kind,
        )
        report = verify_pair_provider(op, truncated, (workload,), device="cpu")
        self.assertFalse(report.ok)
        self.assertIn("3 outputs", report.cases[0].error or "")

    def test_a_mismatched_upstream_gradient_count_is_named(self):
        outputs = (torch.zeros(2), torch.zeros(2))
        with self.assertRaises(ValueError) as caught:
            _grad_outputs(torch.zeros(2), outputs)
        self.assertIn("1 upstream gradients for 2 outputs", str(caught.exception))
        self.assertEqual(len(_grad_outputs(outputs, outputs)), 2)

    def test_the_identity_control_reuses_the_exact_callables(self):
        op, workload, _values = _case("qwen3_qkv_norm_rope")
        provider = pytorch_autograd_provider(op)
        control = renamed_provider(provider, "eager_control")
        self.assertIs(control.forward, provider.forward)
        self.assertIs(control.backward, provider.backward)
        self.assertEqual(control.source_hash, provider.source_hash)
        # Same callables on both sides must clear the same gate; whatever a
        # timed run of this pairing reports is the protocol's noise floor.
        for side in (provider, control):
            self.assertTrue(
                verify_pair_provider(op, side, (workload,), device="cpu").ok
            )

    def test_input_mutation_is_caught_for_a_multi_output_operator(self):
        _op, _workload, values = _case("fused_add_rms_norm")
        snapshots = snapshot_tensors(values)
        assert_tensors_unchanged(values, snapshots, provider="candidate")
        with torch.no_grad():
            values["r"].add_(1.0)  # a backward writing over the residual
        with self.assertRaises(RuntimeError) as caught:
            assert_tensors_unchanged(values, snapshots, provider="candidate")
        self.assertIn("'r'", str(caught.exception))

    def test_running_a_provider_leaves_every_input_untouched(self):
        for name in (*SINGLE_OUTPUT, *MULTI_OUTPUT):
            with self.subTest(op=name):
                op, _workload, values = _case(name)
                provider = pytorch_autograd_provider(op)
                snapshots = snapshot_tensors(values)
                _output, saved = provider.forward(values)
                provider.backward(upstream_grad_values(op, values), saved, values)
                assert_tensors_unchanged(values, snapshots, provider=provider.name)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestTierTwoOperatorProtocol(unittest.TestCase):
    """The operator through the autograd engine: routing, naming, serialization."""

    def _check(self, name):
        op, workload, values = _case(name)
        module = eager_module(op, values)
        return op, check_module(op, module, values, workload)

    def test_a_single_output_operator_still_passes(self):
        op, verdict = self._check("layernorm")
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(
            list(verdict["checks"]),
            ["y", "dx", "dweight", "dbias"],
        )
        # The forward check is keyed by the declared output name, not a fixed
        # "y", so a report reads the same whatever the operator calls it.
        self.assertEqual(op.output_names, ("y",))

    def test_qkv_norm_rope_routes_all_three_output_gradients(self):
        op, verdict = self._check("qwen3_qkv_norm_rope")
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(list(verdict["checks"])[:3], ["q", "k", "v"])
        # Every parameter gradient is checked against the oracle, so a backward
        # that dropped dk or dv would fail here rather than merely look small.
        for grad_name in op.grad_names():
            self.assertIn(grad_name, verdict["checks"])

    def test_fused_add_rms_norm_checks_the_sum_as_well_as_the_norm(self):
        _op, verdict = self._check("fused_add_rms_norm")
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(list(verdict["checks"])[:2], ["out", "summed"])

    def test_qwen3_attention_gradients_reach_q_k_v_and_the_projection(self):
        op, workload, values = _case("qwen3_attention")
        module = eager_module(op, values)
        verdict = check_module(op, module, values, workload)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(
            [n for n in verdict["checks"] if n != "out"],
            ["dq", "dk", "dv", "do_weight"],
        )

    def test_causal_gqa_attention_measures_without_parameters(self):
        _op, verdict = self._check("causal_gqa_attention")
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(list(verdict["checks"]), ["o", "dq", "dk", "dv"])

    def test_a_module_dropping_an_output_is_refused_not_averaged(self):
        op, workload, values = _case("qwen3_qkv_norm_rope")
        module = eager_module(op, values)
        original = module._call
        module._call = lambda *args, **kwargs: original(*args, **kwargs)[0]
        with self.assertRaises(ValueError) as caught:
            check_module(op, module, values, workload)
        self.assertIn("3 outputs", str(caught.exception))

    def test_every_check_serializes_per_output(self):
        for name in (*SINGLE_OUTPUT, *MULTI_OUTPUT):
            with self.subTest(op=name):
                op, verdict = self._check(name)
                encoded = json.loads(json.dumps(verdict))
                for out in op.outputs:
                    entry = encoded["checks"][out.name]
                    self.assertTrue(entry["ok"], entry)
                    self.assertTrue(entry["finite"])
                    # Stride is measured and published; see `_compare` for why
                    # it is reported rather than gated at this tier.
                    self.assertIn("stride", entry)
                    self.assertIn("stride_match", entry)

    def test_the_identity_control_is_two_eager_modules(self):
        specs = identity_control_specs()
        self.assertEqual([s.name for s in specs], ["eager", "eager_control"])
        self.assertEqual({s.kind for s in specs}, {"eager"})

    def test_an_identity_pairing_reads_back_as_one_times(self):
        # The same measurement under two names: whatever the driver reports for
        # this pairing is its noise floor, and the reader must not turn equal
        # times into anything but 1.0.
        block = {
            "ok": True,
            "kind": "eager",
            "forward": {"median_ms": 0.4, "q20_ms": 0.4, "q80_ms": 0.4},
            "full_step": {"median_ms": 1.25, "q20_ms": 1.2, "q80_ms": 1.3},
            "peak_memory_bytes": 4096.0,
            "adapter_kind": "pytorch_eager_module",
        }
        report = from_tier2_report(
            "qwen3_qkv_norm_rope",
            {
                "environment": {"gpu_name": "cpu"},
                "cases": [
                    {
                        "dims": {"B": 1, "T": 16},
                        "dtype": "bfloat16",
                        "providers": {"eager": block, "eager_control": dict(block)},
                    }
                ],
            },
            candidate="eager_control",
            baseline="eager",
        )
        self.assertAlmostEqual(report.cases[0].speedup_full, 1.0)
        self.assertIsNone(report.cases[0].speedup_backward)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestTheDeclarationRefusesAmbiguity(unittest.TestCase):
    def test_the_single_output_accessor_refuses_a_multi_output_declaration(self):
        op = get_op("qwen3_qkv_norm_rope")
        with self.assertRaises(ValueError) as caught:
            op.upstream_grad_name
        self.assertIn("upstream_grad_names", str(caught.exception))

    def test_a_repeated_parameter_name_is_rejected(self):
        from evograd.opdecl.activity import Active, declare_op

        with self.assertRaises(ValueError) as caught:
            declare_op(
                name="toy",
                forward="evograd.ops.level1.swiglu.forward_ref:swiglu_forward_ref",
                dims=("rows", "cols"),
                args=(
                    Active("a", "[rows, cols]"),
                    Active("b", "[rows, cols]"),
                    Active("w", "[cols]"),
                ),
                output=Active("c", "[rows, cols]"),
                parameter_args=("w", "w"),
                forward_semantics="s",
                backward_semantics="s",
            )
        self.assertIn("repeats", str(caught.exception))

    def test_a_multi_output_operator_can_still_declare_parameters(self):
        op = get_op("fused_add_rms_norm")
        self.assertTrue(op.is_multi_output)
        self.assertEqual(op.parameter_args, ("weight",))


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestIntegratedStepUsesTheSharedAdapter(unittest.TestCase):
    """One operator-through-autograd wrapper, two protocols on top of it.

    ``bench.integrated`` measures a training step -- gradient reset inside the
    timed region, many steps under one event pair -- and tier 2 measures the
    forward and the step with ``do_bench``. Those are different protocols. The
    module they both wrap is not, and keeping two of it is how the two drift.
    """

    def test_the_integrated_wrapper_is_the_tier_two_module(self):
        from evograd.bench import integrated

        op, workload, _values = _case("layernorm")
        _activations, _dy, values = integrated.case_tensors(
            op, workload, device="cpu"
        )
        model = integrated.eager_layer(op, values)
        self.assertIsInstance(model, OperatorModule)

    def test_an_operator_carrying_buffers_can_be_wrapped_at_all(self):
        # rope's cos/sin and cross_entropy's target are tensor `Inactive` args.
        # They are neither activations nor parameters, and a wrapper that knew
        # only about those two raised KeyError here.
        from evograd.bench import integrated

        for name in ("rope", "cross_entropy"):
            with self.subTest(op=name):
                op = get_op(name)
                workload = op.correctness[0]
                activations, dy, values = integrated.case_tensors(
                    op, workload, device="cpu"
                )
                model = integrated.eager_layer(op, values)
                integrated.make_training_step(model, activations, dy)()
                for activation in activations:
                    self.assertIsNotNone(activation.grad)

    def test_the_timed_step_backpropagates_every_output(self):
        from evograd.bench import integrated

        op = get_op("fused_add_rms_norm")
        workload = op.correctness[0]
        activations, dy, values = integrated.case_tensors(op, workload, device="cpu")
        self.assertIsInstance(dy, tuple)
        self.assertEqual(len(dy), 2)
        model = integrated.eager_layer(op, values)
        integrated.make_training_step(model, activations, dy)()
        # `summed` feeds the residual stream; a step that backpropagated only
        # `out` would leave a strictly smaller dr and still look plausible.
        self.assertIsNotNone(model.weight.grad)
        for activation in activations:
            self.assertIsNotNone(activation.grad)

    def test_a_candidate_pair_goes_through_bind(self):
        from evograd.bench import integrated
        from evograd.ops.level1.swiglu import forward_ref  # noqa: F401

        op = get_op("swiglu")
        workload = op.correctness[0]
        activations, dy, values = integrated.case_tensors(op, workload, device="cpu")

        class _Pair:
            @staticmethod
            def swiglu_forward_with_saved(a, b):
                s = torch.sigmoid(a.float())
                c = (a.float() * s * b.float()).to(a.dtype)
                return c, (a, b, s)

            @staticmethod
            def swiglu_backward_from_saved(dc, saved):
                a, b, s = saved
                dc32, a32, b32 = dc.float(), a.float(), b.float()
                silu = a32 * s
                da = dc32 * b32 * (s * (1 + a32 * (1 - s)))
                return da.to(a.dtype), (dc32 * silu).to(b.dtype)

        model = integrated.candidate_module(op, _Pair, values=values)
        integrated.make_training_step(model, activations, dy)()
        for activation in activations:
            self.assertIsNotNone(activation.grad)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestModuleTemplate(unittest.TestCase):
    def test_scalar_configuration_rides_in_the_call_template(self):
        op, _workload, values = _case("fused_add_rms_norm")
        module = eager_module(op, values)
        self.assertIsInstance(module, OperatorModule)
        eps_slot = [a.name for a in op.args].index("eps")
        self.assertEqual(module._template[eps_slot], values["eps"])
        # eps is neither a parameter nor a buffer: it takes no gradient and is
        # not a tensor.
        self.assertNotIn("eps", dict(module.named_parameters()))
        self.assertNotIn("_inactive_eps", dict(module.named_buffers()))


if __name__ == "__main__":
    unittest.main()
