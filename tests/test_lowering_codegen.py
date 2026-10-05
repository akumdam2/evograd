"""Regression tests for exact ATen argument binding and generated forward graphs."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from evograd.atenir.extract import _serialise
from evograd.atenir.primitive_triton.dispatch import make_kernel
from evograd.pipelines.b_dispatch.program_codegen import generate_dispatch_program
from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder, _ordered_arg_exprs


def graph_for(fn, *args, symbolic=False):
    def wrapped(*values):
        result = fn(*values)
        return result if isinstance(result, (tuple, list)) else (result,)

    return _serialise(make_fx(wrapped, tracing_mode="symbolic" if symbolic else "real")(*args))


class TestExactLoweringArguments(unittest.TestCase):
    def test_arange_keyword_flags_do_not_become_a_positional_step(self):
        graph = graph_for(
            lambda x: torch.arange(1, x.shape[0] + 1, dtype=torch.int64, pin_memory=False),
            torch.randn(5), symbolic=True,
        )
        node = next(n for n in graph["nodes"] if n.get("target") == "aten.arange.start")
        self.assertEqual(len(node["positional_args"]), 2)
        self.assertEqual(node["kwargs_ordered"]["pin_memory"], {"kind": "scalar", "value": False})
        with patch("evograd.atenir.primitive_triton.elementwise.arange") as arange:
            make_kernel(node)(10)
            arange.assert_called_once_with(1, 10, 1, dtype=torch.int64, device="cuda")

    def test_keyword_order_does_not_change_addmm_scaling(self):
        graph = graph_for(lambda bias, a, b: torch.addmm(bias, a, b, alpha=3, beta=2),
                          torch.randn(2, 4), torch.randn(2, 3), torch.randn(3, 4))
        node = next(n for n in graph["nodes"] if n.get("target") == "aten.addmm.default")
        bias, a, b = object(), object(), object()
        with patch("evograd.atenir.primitive_triton.gemm.addmm") as addmm:
            make_kernel(node)(bias, a, b)
            addmm.assert_called_once_with(bias, a, b, beta=2, alpha=3)

    def test_scalar_add_and_sub_keep_alpha_and_scalar_types(self):
        for target, helper in (("aten.add.Scalar", "add_scalar"), ("aten.sub.Scalar", "sub_scalar"),
                               ("aten.rsub.Scalar", "rsub_scalar")):
            node = {"target": target, "args_ordered": [
                {"kind": "node", "name": "x"}, {"kind": "scalar", "value": 2},
                {"kind": "scalar", "value": 3, "keyword": "alpha"}],
                "kwargs_ordered": {"alpha": {"kind": "scalar", "value": 3}}}
            tensor = object()
            with self.subTest(target=target), patch(
                "evograd.atenir.primitive_triton.elementwise." + helper
            ) as primitive:
                make_kernel(node)(tensor)
                primitive.assert_called_once_with(tensor, 2, 3)

    def test_symbolic_slice_binds_runtime_endpoint_in_its_original_slot(self):
        node = {"target": "aten.slice.Tensor", "args_ordered": [
            {"kind": "node", "name": "x"}, {"kind": "scalar", "value": 1},
            {"kind": "scalar", "value": 1}, {"kind": "sym_node", "name": "end"},
            {"kind": "scalar", "value": 2}]}
        x = torch.arange(30).reshape(3, 10)
        torch.testing.assert_close(make_kernel(node)(x, 8), x[:, 1:8:2])

    def test_amax_axes_and_keepdim_survive_serialization(self):
        graph = graph_for(lambda x: torch.amax(x, dim=(0, 2), keepdim=True), torch.randn(2, 3, 4))
        node = next(n for n in graph["nodes"] if n.get("target") == "aten.amax.default")
        self.assertEqual(node["reduction_dims"], [0, 2])
        self.assertTrue(node["keepdim"])

    def test_selected_gelu_helper_does_not_receive_its_baked_keyword(self):
        graph = graph_for(lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
                          torch.randn(3, 5))
        node = next(n for n in graph["nodes"] if n.get("target") == "aten.gelu.default")
        expression = _ProgramBuilder().call_expr_for_node(
            node, make_kernel(node), _ordered_arg_exprs(node["args_ordered"]))
        self.assertEqual(expression, f"gelu_tanh(t_{node['args_ordered'][0]['name']})")


class TestGeneratedForwardCone(unittest.TestCase):
    def test_symbolic_sizes_alias_and_reshape_are_emitted_in_forward(self):
        def reference(x):
            alias = torch.ops.aten.alias.default(x)
            result = alias.reshape(x.shape[0] * x.shape[1])
            return result, result

        gm = make_fx(reference, tracing_mode="symbolic")(torch.randn(2, 3))
        gm._evograd_num_forward_outputs = 1
        gm._evograd_num_grad_placeholders = 0
        source = generate_dispatch_program(_serialise(gm))
        namespace = {}
        exec(compile(source, "<layout-only>", "exec"), namespace)
        x = torch.randn(5, 7)
        torch.testing.assert_close(namespace["run_forward_program"](x)[0], x.reshape(35))

    def test_empty_shape_reshape_emits_a_tuple_argument(self):
        gm = make_fx(lambda x: (x.reshape(()), x.reshape(())))(torch.randn(1))
        gm._evograd_num_forward_outputs = 1
        source = generate_dispatch_program(_serialise(gm))
        namespace = {}
        exec(compile(source, "<scalar-view>", "exec"), namespace)
        self.assertEqual(namespace["run_forward_program"](torch.ones(1))[0].shape, ())

    def test_symbolic_slice_generated_closure_keeps_baked_plan_keyword_only(self):
        graph = graph_for(lambda x: x[:, : x.shape[1] // 2], torch.randn(3, 8), symbolic=True)
        source = generate_dispatch_program(graph)
        namespace = {}
        exec(compile(source, "<symbolic-slice>", "exec"), namespace)
        x = torch.randn(5, 10)
        torch.testing.assert_close(namespace["run_graph_program"](x)[0], x[:, :5])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for emitted primitive kernels")
class TestGeneratedPrimitiveCuda(unittest.TestCase):
    def test_symbolic_arange_generated_program_executes_with_new_size(self):
        from evograd.opdecl.verify_cli import load_candidate

        graph = graph_for(
            lambda x: torch.arange(1, x.shape[0] + 1, dtype=torch.int64, pin_memory=False),
            torch.randn(5), symbolic=True,
        )
        source = generate_dispatch_program(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.py"
            path.write_text(source)
            module = load_candidate(path)
            actual = module.run_graph_program(torch.randn(11, device="cuda"))[0]
            torch.testing.assert_close(actual, torch.arange(1, 12, device="cuda"))


if __name__ == "__main__":
    unittest.main()
