"""Pipeline codegen/prompt derivations — all pure string work, no torch.

Covers the pieces that replaced the old string conventions:
wrapper codegen (grad_reorder / "d"+name matching), example-input derivation
(hand-typed README strings), prompt rendering, and config templating."""

import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evograd.opdecl.activity import example_input_spec
from evograd.atenir.primitive_triton.dispatch import make_kernel
from evograd.benchmark import get_task
from evograd.pipelines.b_dispatch.program_codegen import (
    _ProgramBuilder,
    _forward_subgraph,
    _node_refs,
    _ordered_arg_exprs,
)
from evograd.pipelines.b_dispatch.wrapper_codegen import (
    grad_indices,
    render_autograd_pair_wrapper,
)


class TestWrapperCodegen(unittest.TestCase):
    def test_layernorm_identity_order_returns_directly(self):
        op = get_task("layernorm")
        self.assertEqual(grad_indices(op), [0, 1, 2])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn("def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):", wrapper)
        self.assertIn("def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):", wrapper)
        # Grads are cast to their source input's dtype (autograd contract).
        self.assertIn(
            "return (_grads[0].to(x.dtype), _grads[1].to(weight.dtype), "
            "_grads[2].to(bias.dtype))",
            wrapper,
        )

    def test_single_tensor_op_unpacks_rather_than_binding_the_tuple(self):
        # `x = saved_tensors[:1]` is a plain assignment, so a one-tensor op
        # would bind the whole tuple and fail on the first `.contiguous()`.
        for name in ("softmax", "relu_squared", "sparsemax"):
            op = get_task(name)
            wrapper = render_autograd_pair_wrapper("m:f", op)
            self.assertIn("x, = saved_tensors[:1]", wrapper, name)

    def test_generated_backward_wrapper_executes(self):
        import torch

        for name in ("softmax", "layernorm"):
            op = get_task(name)
            source = (
                "def run_graph_program(*a):\n"
                "    return tuple(torch.zeros_like(t) for t in a[1:])\n"
                + render_autograd_pair_wrapper("m:f", op)
            )
            namespace = {"torch": torch}
            exec(compile(source, "<generated>", "exec"), namespace)
            n_tensors = len(
                [a for a in op.args if getattr(a, "shape", None) is not None]
            )
            saved = tuple(torch.zeros(2, 3) for _ in range(n_tensors))
            grads = namespace[op.backward_fn_name](torch.zeros(2, 3), saved)
            self.assertEqual(len(grads), len(op.grad_names()), name)

    def test_evoattention_drops_inactive_gradient(self):
        op = get_task("evoattention")
        # res_mask is tensor arg index 3; its graph gradient is skipped.
        self.assertEqual(grad_indices(op), [0, 1, 2, 4])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn(
            "return (_grads[0].to(q.dtype), _grads[1].to(k.dtype), "
            "_grads[2].to(v.dtype), _grads[4].to(pair_bias.dtype))",
            wrapper,
        )
        self.assertIn("def evoattention_backward_from_saved(do, saved_tensors):", wrapper)
        self.assertNotIn("eps", wrapper)  # no scalar consts on this op

    def test_layernorm_linear_honors_grad_order(self):
        op = get_task("layernorm_linear")
        # contract order: dx, dlinear_weight, dweight, dbias
        self.assertEqual(grad_indices(op), [0, 3, 1, 2])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn(
            "return (_grads[0].to(x.dtype), _grads[3].to(linear_weight.dtype), "
            "_grads[1].to(weight.dtype), _grads[2].to(bias.dtype))",
            wrapper,
        )

    def test_inactive_tensor_does_not_create_graph_gradient_slot(self):
        op = get_task("fused_moe_swiglu")
        self.assertEqual(grad_indices(op), [0, 1, 2, 3])
        wrapper = render_autograd_pair_wrapper(
            "evograd.benchmark.operator_suite.tasks.level2.fused_moe_swiglu.forward_ref:"
            "fused_moe_swiglu_forward_ref",
            op,
        )
        self.assertNotIn("_grads[4]", wrapper)

    def test_wrapper_compiles_for_every_single_output_op(self):
        from evograd.benchmark import TASKS

        for name, op in TASKS.items():
            if op.is_multi_output:
                continue
            with self.subTest(op=name):
                wrapper = render_autograd_pair_wrapper("m:f", op)
                compile("def run_graph_program(*a): pass\n" + wrapper, f"<{name}>", "exec")

    def test_a_multi_output_op_is_generated_with_one_grad_per_output(self):
        """Pipeline B used to refuse structured outputs; now it emits them.

        The wrapper takes one upstream gradient per declared output and returns
        one input gradient per active argument, in declared order.
        """
        import ast

        from evograd.benchmark import get_task
        from evograd.pipelines.b_dispatch.wrapper_codegen import (
            render_autograd_pair_wrapper,
        )

        for name in ("fused_add_rms_norm", "qwen3_qkv_norm_rope"):
            with self.subTest(op=name):
                op = get_task(name)
                source = render_autograd_pair_wrapper("pkg.mod:fn", op)
                ast.parse(source)
                self.assertIn(op.forward_fn_name, source)
                self.assertIn(op.backward_fn_name, source)
                # one upstream gradient name per output, unpacked in backward
                for upstream in op.upstream_grad_names:
                    self.assertIn(upstream, source)
                self.assertIn("DEPLOYMENT_ENTRY", source)

class TestDispatchProgramCodegen(unittest.TestCase):
    def test_ordered_args_preserve_scalar_constants(self):
        self.assertEqual(
            _ordered_arg_exprs(
                [
                    {"kind": "node", "name": "mean"},
                    {"kind": "scalar", "value": 1e-5},
                    {
                        "kind": "shape_list",
                        "items": [
                            {"kind": "sym_node", "name": "rows"},
                            {"kind": "scalar", "value": 64},
                        ],
                    },
                ]
            ),
            ["t_mean", "1e-05", "(t_rows, 64)"],
        )

    def test_dispatch_closure_drops_already_baked_scalar_argument(self):
        def factory(scalar):
            def op(a):
                return a + scalar

            return op

        builder = _ProgramBuilder()
        expression = builder.call_expr_for_node(
            {
                "target": "aten.add.Scalar",
                "args_ordered": [
                    {"kind": "node", "name": "input"},
                    {"kind": "scalar", "value": 2.0},
                ],
            },
            factory(2.0),
            ["t_input", "2.0"],
        )
        self.assertEqual(expression, "_k0(t_input)")

        scalar_first = builder.call_expr_for_node(
            {
                "target": "aten.sub.Tensor",
                "args_ordered": [
                    {"kind": "scalar", "value": 1},
                    {"kind": "node", "name": "input"},
                ],
            },
            factory(1.0),
            ["1", "t_input"],
        )
        self.assertEqual(scalar_first, "_k1(t_input)")

    def test_symbolic_conv_and_moe_nodes_have_handwritten_routes(self):
        cases = (
            {
                "name": "conv_bwd",
                "target": "aten.convolution_backward.default",
                "args_ordered": [
                    {"kind": "node", "name": "dy"},
                    {"kind": "node", "name": "x"},
                    {"kind": "node", "name": "weight"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "sym_node", "name": "channels"}],
                    },
                    {"kind": "scalar", "value": [1, 1]},
                    {"kind": "scalar", "value": [0, 0]},
                    {"kind": "scalar", "value": [1, 1]},
                    {"kind": "scalar", "value": False},
                    {"kind": "scalar", "value": [0, 0]},
                    {"kind": "scalar", "value": 1},
                    {"kind": "scalar", "value": [True, True, True]},
                ],
            },
            {
                "name": "expert_index",
                "target": "aten.index.Tensor",
                "args_ordered": [
                    {"kind": "node", "name": "experts"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "node", "name": "routing"}],
                    },
                ],
            },
            {
                "name": "expert_index_put",
                "target": "aten.index_put.default",
                "args_ordered": [
                    {"kind": "node", "name": "base"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "node", "name": "routing"}],
                    },
                    {"kind": "node", "name": "values"},
                    {"kind": "scalar", "value": True},
                ],
            },
        )
        for node in cases:
            with self.subTest(target=node["target"]):
                kernel = make_kernel(node)
                self.assertNotIn("fallback", getattr(kernel, "_dispatch_tag", ""))


class TestExampleInputSpec(unittest.TestCase):
    def test_layernorm_matches_legacy_readme_string(self):
        self.assertEqual(
            example_input_spec(get_task("layernorm")),
            "[(8,64) f32, (64) f32, (64) f32]",
        )

    def test_rmsnorm_matches_legacy_readme_string(self):
        self.assertEqual(
            example_input_spec(get_task("rmsnorm")),
            "[(8,64) f32, (64) f32]",
        )

    def test_evoattention_includes_inactive_mask_with_fixed_dtype(self):
        # First correctness workload with every dim >= 2 (so no symbolic dim
        # specializes to 1 during extraction): B=2 S=2 H=4 N=128 D=64, bfloat16.
        self.assertEqual(
            example_input_spec(get_task("evoattention")),
            "[(2,2,128,4,64) bf16, (2,2,128,4,64) bf16, (2,2,128,4,64) bf16, "
            "(2,2,1,1,128) f32, (2,1,4,128,128) f32]",
        )

    def test_cross_entropy_includes_integer_labels_and_scalar_output(self):
        self.assertEqual(
            example_input_spec(get_task("cross_entropy")),
            "[(8,512) f32, (8) i64]",
        )


class TestPromptRendering(unittest.TestCase):
    def test_pipeline_a_rules_include_contract_and_no_grad(self):
        from evograd.pipelines.a_atenir_llm.prompts import render_pair_rules

        rules = render_pair_rules(get_task("evoattention"))
        self.assertIn("def evoattention_forward_with_saved(q, k, v, res_mask, pair_bias):", rules)
        self.assertIn("return dq, dk, dv, d_pair_bias", rules)
        self.assertIn("`res_mask`", rules)  # no-grad warning present

    def test_pipeline_c_rules_include_contract(self):
        from evograd.pipelines.c_forward_only.prompts import render_pair_rules

        rules = render_pair_rules(get_task("layernorm_linear"))
        self.assertIn("return dx, dlinear_weight, dweight, dbias", rules)

    def test_pipeline_a_grant_text_follows_the_grant(self):
        from evograd.pipelines.a_atenir_llm.prompts import render_pair_rules
        from evograd.pipelines.shared.primitives import REGISTRY

        op = get_task("rmsnorm")
        none = render_pair_rules(op)
        self.assertIn("None granted", none)
        self.assertNotIn("vendor_", none)
        gemm = render_pair_rules(op, ("vendor_gemm",))
        self.assertIn(REGISTRY["vendor_gemm"].source.strip(), gemm)
        self.assertNotIn("vendor_sdpa", gemm)
        both = render_pair_rules(op, ("vendor_sdpa", "vendor_gemm"))
        for spec in REGISTRY.values():
            self.assertIn(spec.source.strip(), both)
        for text in (none, gemm, both):  # nothing granted is called forbidden
            self.assertNotIn("forbidden", text)

    def test_pipeline_a_prompts_carry_the_reference_source_and_the_graph(self):
        from evograd.opdecl.importing import reference_closure
        from evograd.pipelines.a_atenir_llm.prompts import (
            render_codegen_prompt,
            render_context,
            render_plan_prompt,
            render_repair_prompt,
        )

        op = get_task("rmsnorm")
        reference, _ = reference_closure(op.forward)
        context = render_context(reference=reference, graph_summary="GRAPH-SUMMARY", op=op)
        for prompt in (render_plan_prompt(context), render_codegen_prompt(context, "PLAN"),
                       render_repair_prompt(context, "PLAN", "CODE", "{}")):
            self.assertIn(reference.rstrip(), prompt)
            self.assertIn("GRAPH-SUMMARY", prompt)
            self.assertIn(f"def {op.forward_fn_name}(", prompt)

    def test_the_graph_summary_shows_edges_and_scalars_of_a_real_extraction(self):
        import torch

        from evograd.atenir.extract import _serialise, extract_autograd
        from evograd.pipelines.shared.graph_summary import summarize_graph

        gm = extract_autograd(f"{__name__}:_rms", [torch.randn(4, 8), torch.randn(8)],
                              dynamic=True)
        summary = summarize_graph(_serialise(gm))
        self.assertIn("inputs=['arg1_1']", summary)  # x, after the upstream gradient
        self.assertNotIn("inputs=[]", summary)
        self.assertIn("scalars=[1e-06]", summary)
        cat = next(line for line in summary.splitlines() if "aten.cat" in line)
        self.assertRegex(cat, r"inputs=\['\w+', '\w+'\] scalars=\[-1\]")

    def test_pipeline_a_timeout_defaults_leave_c_alone(self):
        from evograd.pipelines.a_atenir_llm.cli import parse_args as parse_a
        from evograd.pipelines.c_forward_only.cli import parse_args as parse_c

        argv = ["--op", "rmsnorm", "--output-dir", "out"]
        a, c = parse_a(argv), parse_c(argv)
        self.assertEqual((a.timeout, a.eval_timeout, a.max_attempts), (600, 300, 3))
        self.assertEqual((c.timeout, c.eval_timeout), (180, 120))


def _rms(x, weight):
    import torch

    half = x.shape[-1] // 2  # symbolic size arithmetic under dynamic tracing
    h = torch.cat((-x[..., half:], x[..., :half]), dim=-1)  # a list operand
    return h * (h.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * weight


class TestPipelineASynthesis(unittest.TestCase):
    """Plan -> codegen -> contract check with a stub LLM; no oracle (no CUDA)."""

    MODULE = (
        "import torch\n\n"
        "# EVOLVE-BLOCK-START\n"
        "def rmsnorm_forward_with_saved(x, weight, eps=1e-5):\n"
        "    rrms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)\n"
        "    return (x.float() * rrms * weight).to(x.dtype), (x, weight, rrms)\n\n"
        "def rmsnorm_backward_from_saved(dy, saved_tensors, eps=1e-5):\n"
        "    x, weight, rrms = saved_tensors\n"
        "    return dy * weight * rrms, (dy * x * rrms).sum(0)\n"
        "# EVOLVE-BLOCK-END\n"
    )

    def _synthesize(self, module, grant=()):
        import json
        import sys
        import types

        from evograd.pipelines.a_atenir_llm.synthesize import (
            AutogradPairConfig,
            synthesize_autograd_pair,
        )

        replies, clients = ["PLAN", f"```python\n{module}```"], []

        def reply(**_):
            message = types.SimpleNamespace(content=replies.pop(0))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        def client(**kwargs):
            clients.append(kwargs)
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=reply)))

        def extract(*, out_path, **_):
            nodes = [{"op": "placeholder", "name": "arg0_1", "shape": [8, 64]},
                     {"op": "output", "args": [["arg0_1"]]}]
            out_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
            return out_path

        op = get_task("rmsnorm")
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = AutogradPairConfig(
            op=op, forward=op.forward, example_input="", output_dir=Path(tmp.name),
            api_base="http://metering-proxy.invalid/v1", model="stub", api_key="k",
            max_attempts=1, max_tokens=64, temperature=None, timeout=5,
            python=sys.executable, allowed_primitives=grant, skip_verify=True)
        with mock.patch("openai.OpenAI", client), mock.patch(
                "evograd.pipelines.a_atenir_llm.synthesize.extract_graph", extract):
            rc = synthesize_autograd_pair(config)
        # Plan and codegen, each through the configured endpoint and key only.
        self.assertEqual([(c["base_url"], c["api_key"]) for c in clients],
                         [(config.api_base, "k")] * 2)
        return rc, Path(tmp.name)

    def test_the_seed_passes_the_protocol_rules_under_its_grant(self):
        import importlib.util

        from evograd.evaluation.protocol import check_rules

        op = get_task("rmsnorm")
        for grant in ((), ("vendor_gemm",)):
            with self.subTest(grant=grant):
                rc, out = self._synthesize(self.MODULE, grant)
                self.assertEqual(rc, 0)
                path = out / "best" / "initial_program_autograd_pair.py"
                spec = importlib.util.spec_from_file_location("a_seed", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.ARTIFACT_CONTRACT["allowed_primitives"], list(grant))
                self.assertIsNone(check_rules(op, module, path))
                self.assertIn("Autograd-pair rules", (out / "autograd_pair_plan_prompt.md").read_text())

    def test_model_code_outside_the_block_is_rejected(self):
        import json

        for stray in ("\nif __name__ == '__main__':\n    print(1)\n",       # after the block
                      "from torch import mm\n"):                            # aliasing import
            with self.subTest(stray=stray):
                module = (self.MODULE + stray if stray.startswith("\n")
                          else stray + self.MODULE)
                rc, out = self._synthesize(module)
                self.assertEqual(rc, 1)
                report = json.loads((out / "attempt_001" / "verification_report.json").read_text())
                self.assertIn("outside the EVOLVE-BLOCK", report["artifact_contract"]["error"])
                self.assertFalse((out / "best").exists())


class TestEvolveConfigRendering(unittest.TestCase):
    def test_all_placeholders_replaced(self):
        from evograd.evolve.run import render_config

        for name in ("layernorm", "evoattention"):
            with self.subTest(op=name):
                config = render_config(get_task(name), iterations=7)
                self.assertNotIn("__", config)
                self.assertIn("max_iterations: 7", config)
                self.assertIn(get_task(name).forward_fn_name, config)
                self.assertIn("models:", config)
                self.assertNotIn("primary_model:", config)

    def test_default_map_elites_feature_dimensions(self):
        from evograd.evolve.run import render_config

        config = render_config(get_task("layernorm"))
        self.assertIn(
            'feature_dimensions: ["complexity", "saved_memory_ratio"]', config
        )
        self.assertIn("feature_bins: 10", config)

    def test_custom_feature_dimensions_render_and_validate(self):
        from evograd.evolve.run import render_config

        config = render_config(
            get_task("layernorm"),
            feature_dimensions=("diversity", "speedup"),
            feature_bins=6,
        )
        self.assertIn('feature_dimensions: ["diversity", "speedup"]', config)
        self.assertIn("feature_bins: 6", config)
        with self.assertRaises(ValueError):
            render_config(get_task("layernorm"), feature_dimensions=("not_a_metric",))
        with self.assertRaises(ValueError):
            render_config(get_task("layernorm"), feature_dimensions=())
        with self.assertRaises(ValueError):
            render_config(get_task("layernorm"), feature_bins=0)

    def test_grid_density_controls_render_and_validate(self):
        from evograd.evolve.run import render_config

        config = render_config(get_task("layernorm"))
        self.assertIn("num_islands: 2", config)
        self.assertIn("archive_size: 20", config)
        dense = render_config(
            get_task("layernorm"),
            feature_dimensions=("saved_memory_ratio", "shape_specialization"),
            feature_bins=3,
            num_islands=1,
            archive_size=8,
        )
        self.assertIn("num_islands: 1", dense)
        self.assertIn("archive_size: 8", dense)
        self.assertIn("feature_bins: 3", dense)
        # OpenEvolve clamps int bins up to ceil(archive^(1/dims)): an archive
        # bigger than the grid would silently undo the coarse-bin request.
        with self.assertRaises(ValueError):
            render_config(
                get_task("layernorm"),
                feature_dimensions=("saved_memory_ratio", "shape_specialization"),
                feature_bins=3,
                archive_size=10,
            )

    def test_evaluator_failure_results_carry_feature_metrics(self):
        # OpenEvolve raises if a configured custom feature dimension is missing
        # from any program's metrics, so failure paths must include them too.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from evograd.evolve.evaluator import build_evaluate
        from evograd.evolve.scoring import CUSTOM_FEATURE_DIMENSION_DEFAULTS, get_policy

        evaluate = build_evaluate(get_task("layernorm"), get_policy("speed_memory"))
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = evaluate("/nonexistent/candidate.py")
        for dim, default in CUSTOM_FEATURE_DIMENSION_DEFAULTS.items():
            self.assertEqual(result.metrics.get(dim), default)

    def test_run_wrapper_propagates_declaration_native_benchmark_selection(self):
        from evograd.evolve.run import run_evolve

        with TemporaryDirectory() as td:
            root = Path(td)
            seed = root / "seed.py"
            seed.write_text("# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END\n")
            observed_env = {}

            def fake_run_evolution(**kwargs):
                observed_env.update(os.environ)
                return mock.Mock(best_code="")

            with (
                mock.patch("openevolve.run_evolution", side_effect=fake_run_evolution),
                mock.patch(
                    "evograd.opdecl.baselines.resolve_performance_baseline",
                    return_value="cublas_pair",
                ),
            ):
                code = run_evolve(
                    get_task("layernorm"),
                    seed_path=seed,
                    output_dir=root / "out",
                    benchmark_suite="tb_i12",
                    benchmark_dtypes=("bfloat16",),
                    performance_baseline="cublas_pair",
                )
            self.assertEqual(code, 1)
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_SUITE"], "tb_i12")
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_DTYPES"], "bfloat16")
            self.assertEqual(observed_env["EVOGRAD_PERFORMANCE_BASELINE"], "cublas_pair")


if __name__ == "__main__":
    unittest.main()


class TestForwardSubgraphSplit(unittest.TestCase):
    """Splitting the joint trace into its forward cone.

    The extractor returns forward outputs ahead of the gradients; the codegen
    has to emit a forward entry point over the forward placeholders alone.
    """

    # x -> sq -> mean -> y   (forward)      dy, mean -> gsq -> gx   (backward)
    NODES = [
        {"name": "sq", "args_ordered": [{"kind": "node", "name": "x"}]},
        {"name": "mean", "args_ordered": [
            {"kind": "node", "name": "sq"},
            {"kind": "shape_list", "items": [{"kind": "scalar", "value": -1}]},
        ]},
        {"name": "y", "args_ordered": [
            {"kind": "node", "name": "x"}, {"kind": "node", "name": "mean"}]},
        {"name": "gsq", "args_ordered": [
            {"kind": "node", "name": "dy"}, {"kind": "node", "name": "mean"}]},
        {"name": "gx", "args_ordered": [
            {"kind": "node", "name": "gsq"}, {"kind": "node", "name": "x"}]},
    ]

    def test_the_cone_excludes_backward_nodes_and_stays_topological(self):
        self.assertEqual(
            _forward_subgraph(self.NODES, ["y"], {"dy"}), ["sq", "mean", "y"]
        )

    def test_a_cone_reaching_an_upstream_gradient_refuses(self):
        # Not separable: emitting a forward from it would silently require a
        # gradient the forward is never given. Refuse rather than emit.
        self.assertIsNone(_forward_subgraph(self.NODES, ["gx"], {"dy"}))

    def test_node_refs_reads_shape_list_members(self):
        self.assertEqual(_node_refs(self.NODES[1]), ["sq"])

    def test_node_refs_falls_back_to_the_pre_args_ordered_schema(self):
        self.assertEqual(
            _node_refs({"name": "z", "input_nodes": [{"name": "a"}, {"name": "b"}]}),
            ["a", "b"],
        )


class TestGeneratedForwardWrapper(unittest.TestCase):
    """Pipeline B's forward must come from the graph, not the eager reference."""

    def test_the_forward_calls_the_generated_program(self):
        op = get_task("rmsnorm")
        wrapper = render_autograd_pair_wrapper("m:f", op, has_generated_forward=True)
        self.assertIn("_outs = run_forward_program(x, weight)", wrapper)
        self.assertNotIn("_load_forward_callable", wrapper)

    def test_without_a_generated_forward_it_still_falls_back_to_eager(self):
        op = get_task("rmsnorm")
        wrapper = render_autograd_pair_wrapper("m:f", op, has_generated_forward=False)
        self.assertIn("_load_forward_callable", wrapper)
        self.assertNotIn("run_forward_program", wrapper)

    def test_the_generated_forward_wrapper_executes(self):
        import torch

        op = get_task("rmsnorm")
        source = (
            "def run_forward_program(*a):\n"
            "    return (a[0] * 2.0,)\n"
            "def run_graph_program(*a):\n"
            "    return tuple(torch.zeros_like(t) for t in a[1:])\n"
            + render_autograd_pair_wrapper("m:f", op, has_generated_forward=True)
        )
        namespace = {"torch": torch}
        exec(compile(source, "<generated>", "exec"), namespace)
        x, weight = torch.randn(4, 8), torch.randn(8)
        y, saved = namespace[op.forward_fn_name](x, weight)
        self.assertTrue(torch.equal(y, x * 2.0))
        self.assertEqual(len(saved), 2)

    def test_traced_scalars_are_enforced_rather_than_ignored(self):
        import torch

        op = get_task("rmsnorm")
        wrapper = render_autograd_pair_wrapper(
            "m:f", op, has_generated_forward=True, scalars={"eps": 1e-6})
        self.assertNotIn("_ = eps", wrapper)
        source = (
            "def run_forward_program(*a):\n"
            "    return (a[0] * 2.0,)\n"
            "def run_graph_program(*a):\n"
            "    return tuple(torch.zeros_like(t) for t in a[1:])\n"
            + wrapper
        )
        namespace = {"torch": torch}
        exec(compile(source, "<generated>", "exec"), namespace)
        x, weight = torch.randn(4, 8), torch.randn(8)
        namespace[op.forward_fn_name](x, weight, 1e-6)
        for call in (lambda: namespace[op.forward_fn_name](x, weight, 1e-5),
                     lambda: namespace[op.backward_fn_name](x, (x, weight), 1e-5)):
            with self.assertRaisesRegex(ValueError, "eps=1e-06"):
                call()

    def test_the_pair_closes_the_block_and_the_granted_layer_follows(self):
        op = get_task("rmsnorm")
        wrapper = render_autograd_pair_wrapper(
            "m:f", op, has_generated_forward=True, allowed_primitives=("vendor_gemm",))
        end = wrapper.index("# EVOLVE-BLOCK-END")
        self.assertLess(wrapper.index(f"def {op.backward_fn_name}("), end)
        self.assertLess(end, wrapper.index("def vendor_gemm("))
        self.assertLess(end, wrapper.index("ARTIFACT_CONTRACT = "))
        self.assertIn("'allowed_primitives': ['vendor_gemm']", wrapper)
