"""One artifact contract, generated identically for every pipeline.

The ABI belongs to EvoGrad: argument order, output arity and order, upstream-
and input-gradient order, inactive-argument handling. A generator picks kernels,
launch strategy and saved state -- never the interface.
"""

from __future__ import annotations

import ast
import types
import unittest

import torch

from evograd.opdecl.activity import Active, Inactive, Workload, declare_op
from evograd.ops import get_op
from evograd.pipelines.shared.artifact import (
    ArtifactContract,
    ArtifactError,
    deployment_entry,
    find_forwarding_aliases,
    render_deployment_layer,
    validate_artifact,
)

SINGLE = "rmsnorm"                 # one output, one upstream gradient
STRUCTURED = "fused_add_rms_norm"  # two outputs, two independent upstream grads


def build(op, *, forward_body=None):
    """Stub pair + the real generated deployment layer, exec'd into a module."""
    args = ", ".join(a.name for a in op.args)
    outs = list(op.output_names)
    ret = f"({', '.join(outs)})" if len(outs) > 1 else outs[0]
    scalars = ", ".join(f"{c.name}={c.default!r}" for c in op.scalar_inactive_args())
    body = forward_body or (
        "    " + ", ".join(outs) + " = " + ", ".join(f"x * {i + 1}" for i in range(len(outs)))
        + f"\n    return {ret}, (x, weight)\n"
    )
    grads = ", ".join(
        "torch.zeros_like(x)" if g != "dweight" else "torch.zeros_like(weight)"
        for g in op.grad_names()
    )
    src = (
        "import torch\n\n"
        f"def {op.forward_fn_name}({args}):\n{body}\n"
        f"def {op.backward_fn_name}(grads, saved, {scalars}):\n"
        "    x, weight = saved\n"
        f"    return {grads}\n"
    ) + render_deployment_layer(op)
    module = types.ModuleType("artifact")
    module.__source__ = src
    exec(compile(src, "<artifact>", "exec"), module.__dict__)
    return module


class TestContractShape(unittest.TestCase):
    def test_a_single_output_declaration_generates_one_upstream_gradient(self):
        op = get_op(SINGLE)
        c = ArtifactContract(op)
        self.assertEqual(c.output_names, ("y",))
        self.assertEqual(c.upstream_names, ("dy",))
        module = build(op)
        self.assertEqual(
            list(module.RmsnormFunction.backward.__code__.co_varnames[:2]),
            ["ctx", "dy"],
        )

    def test_a_structured_declaration_generates_one_gradient_per_output(self):
        op = get_op(STRUCTURED)
        c = ArtifactContract(op)
        self.assertEqual(c.output_names, ("out", "summed"))
        self.assertEqual(c.upstream_names, ("dout", "dsummed"))
        module = build(op)
        names = module.FusedAddRmsNormFunction.backward.__code__.co_varnames[:3]
        self.assertEqual(list(names), ["ctx", "dout", "dsummed"])

    def test_forward_arguments_follow_the_declared_order_exactly(self):
        for name in (SINGLE, STRUCTURED):
            with self.subTest(op=name):
                op = get_op(name)
                module = build(op)
                entry = deployment_entry(module)
                import inspect

                self.assertEqual(
                    list(inspect.signature(entry).parameters),
                    [a.name for a in op.args],
                )

    def test_one_gradient_slot_per_forward_argument_with_none_for_inactive(self):
        op = get_op(STRUCTURED)
        order = ArtifactContract(op).gradient_return_order()
        # x, r, weight are active; eps is a scalar Inactive and takes None.
        self.assertEqual(order, ["dx", "dr", "dweight", None])

    def test_the_metadata_records_every_pinned_order(self):
        meta = ArtifactContract(get_op(STRUCTURED)).metadata()
        self.assertEqual(meta["arguments"], ["x", "r", "weight", "eps"])
        self.assertEqual(meta["outputs"], ["out", "summed"])
        self.assertEqual(meta["upstream_grads"], ["dout", "dsummed"])
        self.assertEqual(meta["input_grads"], ["dx", "dr", "dweight"])
        self.assertEqual(meta["gradient_return_order"], ["dx", "dr", "dweight", "None"])


class TestGeneratedBehaviour(unittest.TestCase):
    def test_both_outputs_reach_backward_independently(self):
        op = get_op(STRUCTURED)
        # A forward whose second output depends on `r` alone isolates dsummed:
        # if the wrapper dropped it, r's gradient would come back zero.
        module = build(op, forward_body=(
            "    out = x * 2\n    summed = r * 3\n"
            "    return (out, summed), (x, weight)\n"
        ))
        module.fused_add_rms_norm_backward_from_saved = (
            lambda grads, saved, eps=1e-6: (grads[0], grads[1], saved[1] * 0)
        )
        x = torch.ones(2, 3, requires_grad=True)
        r = torch.ones(2, 3, requires_grad=True)
        w = torch.ones(3, requires_grad=True)
        out, summed = module.fused_add_rms_norm_deployment(x, r, w, 1e-6)
        torch.autograd.backward((out, summed),
                                (torch.full_like(out, 5.0), torch.full_like(summed, 7.0)))
        self.assertTrue(torch.allclose(x.grad, torch.full_like(x, 5.0)))
        self.assertTrue(torch.allclose(r.grad, torch.full_like(r, 7.0)))

    def test_the_scalar_argument_is_stored_on_ctx_and_returns_no_gradient(self):
        op = get_op(STRUCTURED)
        module = build(op)
        seen = {}
        original = module.fused_add_rms_norm_backward_from_saved

        def spy(grads, saved, eps=1e-6):
            seen["eps"] = eps
            return original(grads, saved, eps)

        module.fused_add_rms_norm_backward_from_saved = spy
        x = torch.ones(2, 3, requires_grad=True)
        r = torch.ones(2, 3, requires_grad=True)
        w = torch.ones(3, requires_grad=True)
        out, summed = module.fused_add_rms_norm_deployment(x, r, w, 4.25e-3)
        torch.autograd.backward((out, summed),
                                (torch.ones_like(out), torch.ones_like(summed)))
        self.assertEqual(seen["eps"], 4.25e-3)     # carried on ctx, not saved

    def test_saved_tensors_go_through_save_for_backward(self):
        op = get_op(STRUCTURED)
        module = build(op)
        source = module.__source__
        self.assertIn("ctx.save_for_backward(*saved)", source)
        self.assertIn("ctx.saved_tensors", source)


class TestNoForwardingAliases(unittest.TestCase):
    def test_a_public_function_that_only_forwards_is_detected(self):
        bad = "def foo_forward(x):\n    return _foo_forward_impl(x)\n"
        self.assertEqual(find_forwarding_aliases(bad), ["foo_forward"])

    def test_a_real_implementation_is_not_flagged(self):
        good = "def foo_forward(x):\n    y = x + 1\n    return y, (x,)\n"
        self.assertEqual(find_forwarding_aliases(good), [])

    def test_neither_pipeline_emits_a_forwarding_alias(self):
        from evograd.pipelines.b_dispatch.wrapper_codegen import (
            render_autograd_pair_wrapper,
        )

        for name in (SINGLE, STRUCTURED):
            with self.subTest(op=name):
                source = render_autograd_pair_wrapper("pkg.mod:fn", get_op(name))
                self.assertEqual(find_forwarding_aliases(source), [])
                self.assertNotIn("_impl", source)


class TestPipelineParity(unittest.TestCase):
    """A and B must not drift into two deployment conventions."""

    def test_both_pipelines_emit_the_identical_deployment_layer(self):
        from evograd.pipelines.b_dispatch.wrapper_codegen import (
            render_autograd_pair_wrapper,
        )

        for name in (SINGLE, STRUCTURED):
            with self.subTest(op=name):
                op = get_op(name)
                layer = render_deployment_layer(op)          # what A appends
                b_source = render_autograd_pair_wrapper("pkg.mod:fn", op)
                # Containment, not suffix: both pipelines embed this text
                # verbatim, but each assembles its file differently -- A appends
                # it last, B's CLI writes the graph-program body after it.
                # Ordering is an assembly detail; identity of the layer is the
                # invariant that keeps the two from drifting.
                self.assertIn(layer.strip(), b_source)

    def test_b_output_satisfies_the_same_contract_a_is_validated_against(self):
        from evograd.pipelines.b_dispatch.wrapper_codegen import (
            render_autograd_pair_wrapper,
        )

        for name in (SINGLE, STRUCTURED):
            with self.subTest(op=name):
                op = get_op(name)
                source = render_autograd_pair_wrapper("pkg.mod:fn", op)
                ast.parse(source)
                for symbol in ArtifactContract(op).required_symbols():
                    self.assertIn(symbol, source)


class TestValidationRejectsMalformed(unittest.TestCase):
    def test_a_missing_symbol_is_reported_by_name(self):
        op = get_op(STRUCTURED)
        module = build(op)
        del module.FusedAddRmsNormModule
        with self.assertRaises(ArtifactError) as caught:
            validate_artifact(op, module)
        self.assertIn("FusedAddRmsNormModule", str(caught.exception))

    def test_a_wrong_argument_order_is_rejected(self):
        op = get_op(STRUCTURED)
        module = build(op)
        module.fused_add_rms_norm_deployment = lambda weight, x, r, eps: None
        with self.assertRaises(ArtifactError) as caught:
            validate_artifact(op, module)
        self.assertIn("argument order", str(caught.exception))

    def test_a_non_string_deployment_entry_is_rejected(self):
        op = get_op(STRUCTURED)
        module = build(op)
        module.DEPLOYMENT_ENTRY = lambda: None
        with self.assertRaises(ArtifactError):
            validate_artifact(op, module)

    def test_a_forwarding_alias_in_the_source_is_rejected(self):
        op = get_op(STRUCTURED)
        module = build(op)
        bad = module.__source__ + (
            "\ndef public_thing(x):\n    return _public_thing_impl(x)\n"
        )
        with self.assertRaises(ArtifactError) as caught:
            validate_artifact(op, module, source=bad)
        self.assertIn("_impl", str(caught.exception))


class TestNoBindInTheDirectStack(unittest.TestCase):
    def test_the_generated_deployment_layer_never_mentions_the_binder(self):
        for name in (SINGLE, STRUCTURED):
            with self.subTest(op=name):
                layer = render_deployment_layer(get_op(name))
                for token in ("opdecl", "bind(", "lookup_pair", "OperatorModule"):
                    self.assertNotIn(token, layer)

    def test_tier2_routes_a_direct_artifact_away_from_the_binder(self):
        import inspect

        from evograd.bench.tier2 import candidate_module

        source = inspect.getsource(candidate_module)
        self.assertIn("deployment_entry", source)
        self.assertIn("validate_artifact", source)
        # legacy remains reachable, and labelled
        self.assertIn("legacy_bind_pair_module", source)


if __name__ == "__main__":
    unittest.main()


class TestEvolvableRegionCoversPair(unittest.TestCase):
    """The pair bodies are the implementation, so they must be evolvable."""

    def _source(self, start_line, end_line):
        op = get_op(STRUCTURED)
        body = [
            "import torch",
            "# EVOLVE-BLOCK-START" if start_line else "# nothing",
            "def _kernel(): pass",
            "# EVOLVE-BLOCK-END" if end_line == "early" else "# filler",
            f"def {op.forward_fn_name}(x, r, weight, eps=1e-6):",
            "    return (x, r), (x,)",
            f"def {op.backward_fn_name}(g, s, eps=1e-6):",
            "    return g[0], g[1], s[0]",
            "# EVOLVE-BLOCK-END" if end_line == "late" else "# end",
        ]
        return "\n".join(body)

    def test_a_block_that_stops_before_the_pair_is_rejected(self):
        from evograd.pipelines.shared.artifact import evolvable_region_covers_pair

        op = get_op(STRUCTURED)
        self.assertFalse(
            evolvable_region_covers_pair(op, self._source(True, "early"))
        )

    def test_a_block_spanning_kernels_and_pair_is_accepted(self):
        from evograd.pipelines.shared.artifact import evolvable_region_covers_pair

        op = get_op(STRUCTURED)
        self.assertTrue(evolvable_region_covers_pair(op, self._source(True, "late")))

    def test_missing_markers_are_rejected(self):
        from evograd.pipelines.shared.artifact import evolvable_region_covers_pair

        op = get_op(STRUCTURED)
        self.assertFalse(evolvable_region_covers_pair(op, self._source(False, "late")))

    def test_validate_artifact_enforces_it_when_given_source(self):
        op = get_op(STRUCTURED)
        module = build(op)
        with self.assertRaises(ArtifactError) as caught:
            validate_artifact(op, module, source=self._source(True, "early"))
        self.assertIn("EVOLVE-BLOCK", str(caught.exception))

    def test_the_prompt_asks_for_the_wider_block(self):
        from evograd.pipelines.a_atenir_llm.prompts import render_pair_rules

        text = render_pair_rules(get_op(STRUCTURED))
        self.assertIn("EVOLVE-BLOCK-END", text)
        self.assertIn("after the two public pair functions", text)


class TestRankAdaptationCases(unittest.TestCase):
    """Which declarations get leading-dimension adaptation, and which cannot."""

    def _layer(self, name):
        return render_deployment_layer(get_op(name))

    def test_a_batched_declaration_restores_leading_dimensions(self):
        # rmsnorm: x [rows, cols] -> y [rows, cols]. Inputs and outputs share
        # the top rank, so flatten-and-restore is unambiguous.
        layer = self._layer(SINGLE)
        self.assertIn("_leading = None", layer)
        self.assertIn("y = y.view(*_leading, y.shape[-1])", layer)

    def test_a_structured_declaration_restores_every_output(self):
        layer = self._layer(STRUCTURED)
        for name in ("out", "summed"):
            self.assertIn(f"{name} = {name}.view(*_leading, {name}.shape[-1])", layer)

    def test_a_scalar_output_declaration_gets_no_adaptation(self):
        # tvd: p, q are [rows, cols] but the output is a scalar []. There is no
        # leading shape to restore onto a scalar, so the entry calls at the
        # declared rank instead of inventing one.
        layer = self._layer("tvd")
        self.assertNotIn("_leading", layer)
        self.assertIn("return TvdFunction.apply(p, q)", layer)

    def test_a_mixed_rank_declaration_gets_no_adaptation(self):
        # qwen3_attention takes [B, HQ, T, D] and returns [B, T, H]: inputs and
        # outputs disagree on rank, so no single leading shape exists. Guessing
        # one would silently reshape a result.
        from evograd.pipelines.shared.artifact import batched_names

        op = get_op("qwen3_attention")
        args, outs, rank = batched_names(op)
        self.assertEqual(rank, 4)
        self.assertEqual(outs, ())          # no output shares the argument rank
        self.assertNotIn("_leading", self._layer("qwen3_attention"))

    def test_every_declaration_generates_parseable_code(self):
        from evograd.ops import OPS

        adapting = []
        for name in sorted(OPS):
            with self.subTest(op=name):
                layer = render_deployment_layer(get_op(name))
                ast.parse("import torch\n" + layer)
                if "_leading" in layer:
                    adapting.append(name)
        # A regression guard on the split itself: silently widening or
        # narrowing which declarations adapt would change generated behaviour
        # for operators nobody re-checked.
        self.assertIn("fused_add_rms_norm", adapting)
        self.assertIn("rmsnorm", adapting)
        self.assertNotIn("tvd", adapting)
        self.assertNotIn("qwen3_attention", adapting)
