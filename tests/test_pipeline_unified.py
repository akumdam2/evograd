"""Compiler-guided primitive synthesis, fixed providers, and the training ABI.

The CPU tests check preservation and graph serialization. CUDA tests exercise
real compiler capture and generated Triton against an independent autograd
reference; they skip on login nodes without a GPU.
"""

from __future__ import annotations

import importlib.util
import ast
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from evograd.benchmark import get_task
from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.importing import resolve_callable
from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
from evograd.pipelines.unified.contract import build_contract, validate_source
from evograd.pipelines.shared.artifact import deployment_entry, validate_artifact

from tests._small_cases import SMALL


_CONTRACT_SOURCE = '''
UNIFIED_EXTERNAL_REGIONS = ({'wrapper': '_external_mm'},)

def _external_mm(a, b):
    return vendor.mm(a, b)

# EVOLVE-BLOCK-START
def forward(a, b):
    y = _external_mm(a, b)
    return y * 2
# EVOLVE-BLOCK-END

DEPLOYMENT_ENTRY = 'forward'
'''


class TestPreservedRegions(unittest.TestCase):
    def setUp(self):
        self.contract = build_contract(_CONTRACT_SOURCE)

    def test_surrounding_math_can_change_but_provider_and_abi_cannot(self):
        validate_source(_CONTRACT_SOURCE.replace('return y * 2', 'return y + y'), self.contract)
        for before, after in (
            ('vendor.mm(a, b)', 'vendor.bmm(a, b)'),
            ("DEPLOYMENT_ENTRY = 'forward'", "DEPLOYMENT_ENTRY = 'other'"),
        ):
            with self.subTest(change=before), self.assertRaises(ValueError):
                validate_source(_CONTRACT_SOURCE.replace(before, after), self.contract)

    def test_external_call_cannot_be_removed_duplicated_or_redirected(self):
        for replacement in (
            'y = a',
            'y = _external_mm(a, b) + _external_mm(a, b)',
            '_external_mm = replacement\n    y = _external_mm(a, b)',
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_source(
                    _CONTRACT_SOURCE.replace('y = _external_mm(a, b)', replacement),
                    self.contract,
                )

    def test_formatting_changes_do_not_change_the_fixed_semantics(self):
        validate_source(
            _CONTRACT_SOURCE.replace('return vendor.mm(a, b)', 'return vendor.mm( a, b )  # fixed'),
            self.contract,
        )

    def test_without_regions_only_the_fixed_code_is_pinned(self):
        source = (_CONTRACT_SOURCE.split('# EVOLVE-BLOCK-START')[0].split('\n', 2)[2]
                  .replace('_external_mm', 'vendor_gemm')
                  + '# EVOLVE-BLOCK-START\ndef forward(a, b):\n    return vendor_gemm(a, b) * 2\n'
                  + '# EVOLVE-BLOCK-END\n\nDEPLOYMENT_ENTRY = \'forward\'\n')
        contract = build_contract(source)
        self.assertEqual((contract['external_regions'], contract['external_call_sites']), ([], {}))
        # The block may drop, repeat or restructure its library calls ...
        validate_source(source.replace('return vendor_gemm(a, b) * 2',
                                       'return vendor_gemm(a, b) + vendor_gemm(a, b)'), contract)
        # ... but code outside it, e.g. a helper the block could call, stays the seed's.
        with self.assertRaises(ValueError):
            validate_source(source.replace('# EVOLVE-BLOCK-START',
                                           'def helper(a, b):\n    return a.mm(b)\n\n# EVOLVE-BLOCK-START'),
                            contract)

    def test_region_requires_a_fixed_provider_and_one_seed_call(self):
        with self.assertRaises(ValueError):
            build_contract(_CONTRACT_SOURCE.replace('def _external_mm(a, b):', 'def other(a, b):'))
        with self.assertRaises(ValueError):
            build_contract(_CONTRACT_SOURCE.replace('y = _external_mm(a, b)', 'y = a'))

    def test_evolution_evaluator_enforces_the_external_manifest(self):
        from evograd.evolve.evaluator import _primitive_usage
        from evograd.pipelines.unified.contract import CONTRACT_ENV

        module = SimpleNamespace(ARTIFACT_CONTRACT={'allowed_primitives': []})
        with tempfile.TemporaryDirectory(prefix='evograd-unified-contract-') as directory:
            program = Path(directory) / 'candidate.py'
            manifest = Path(directory) / 'contract.json'
            manifest.write_text(json.dumps(self.contract), encoding='utf-8')
            with patch.dict(os.environ, {CONTRACT_ENV: str(manifest)}, clear=True):
                program.write_text(_CONTRACT_SOURCE, encoding='utf-8')
                usage, failure = _primitive_usage(str(program), module)
                self.assertIsNone(failure)
                self.assertTrue(usage['unified_fixed_code_valid'])
                program.write_text(
                    _CONTRACT_SOURCE.replace('y = _external_mm(a, b)', 'y = a'),
                    encoding='utf-8',
                )
                usage, failure = _primitive_usage(str(program), module)
                self.assertIsNone(usage)
                self.assertEqual(failure['error_type'], 'PrimitiveViolation')
                self.assertIn('unified pipeline preservation', failure['error_message'])


class TestSemanticCapture(unittest.TestCase):
    def test_reduction_parameters_and_transposed_strides_survive_serialization(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        from evograd.pipelines.unified.capture import _serialize_graph

        graph = make_fx(lambda x: torch.sum(x, dim=[0], keepdim=True, dtype=torch.float64))(
            torch.randn(3, 5).t()
        )
        semantic = _serialize_graph(graph)
        placeholder = next(n for n in semantic['nodes'] if n['op'] == 'placeholder')
        reduction = next(n for n in semantic['nodes'] if n.get('target') == 'aten.sum.dim_IntList')
        self.assertEqual(placeholder['stride'], [1, 5])
        self.assertEqual(reduction['positional_args'][1],
                         {'kind': 'shape_list', 'items': [{'kind': 'scalar', 'value': 0}]})
        self.assertEqual(reduction['positional_args'][2], {'kind': 'scalar', 'value': True})
        self.assertEqual(reduction['kwargs_ordered']['dtype'],
                         {'kind': 'torch_constant', 'value': 'torch.float64'})

    def test_strict_library_audit_rejects_inplace_math_and_hidden_assignments(self):
        from evograd.pipelines.unified.codegen import _audit_primitive_library, UnsupportedSemanticNode

        statements = (
            'result = torch.sum(x)',
            'result = x.sum()',
            'x.scatter_add_(0, index, values)',
            'x.add_(values)',
            'x.mul_(values)',
        )
        for statement in statements:
            source = f'def helper(x, index, values):\n    {statement}\n    return x\n'
            with self.subTest(statement=statement), self.assertRaises(UnsupportedSemanticNode):
                _audit_primitive_library(ast.parse(source))


class TestPreservedAttention(unittest.TestCase):
    """cuDNN's fused attention is preserved the way cuBLAS GEMMs are."""

    def test_the_cudnn_attention_externs_render_as_fixed_wrappers(self):
        from evograd.pipelines.unified.capture import _SUPPORTED_EXTERNALS
        from evograd.pipelines.unified.codegen import _external_wrapper

        forward = 'torch.ops.aten._scaled_dot_product_cudnn_attention.default'
        backward = 'torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default'
        self.assertEqual(_SUPPORTED_EXTERNALS[forward], 'aten._scaled_dot_product_cudnn_attention.default')
        self.assertEqual(_SUPPORTED_EXTERNALS[backward],
                         'aten._scaled_dot_product_cudnn_attention_backward.default')
        node = {
            'name': 'attn', 'external_kernel': forward, 'target': _SUPPORTED_EXTERNALS[forward],
            'positional_args': [{'kind': 'node', 'name': 'q'}, {'kind': 'node', 'name': 'k'},
                                {'kind': 'node', 'name': 'v'}, {'kind': 'scalar', 'value': None},
                                {'kind': 'scalar', 'value': True}, {'kind': 'scalar', 'value': 0.0},
                                {'kind': 'scalar', 'value': True}, {'kind': 'scalar', 'value': False}],
            'kwargs_ordered': {'scale': {'kind': 'scalar', 'value': None}},
        }
        source, call, metadata = _external_wrapper('forward', node)
        from evograd.pipelines.unified.codegen import _pyname

        wrapper = f'_unified_external_forward_{_pyname("attn")}'
        self.assertIn(f'def {wrapper}(arg0, arg1, arg2):', source)
        self.assertIn('with torch.no_grad():', source)
        self.assertIn('return _unified_sdpa_forward(arg0, arg1, arg2, None, True, 0.0, True, False, '
                      'scale=None)', source)
        self.assertEqual(call, f'{wrapper}({_pyname("q")}, {_pyname("k")}, {_pyname("v")})')
        self.assertEqual(metadata['kernel'], forward)
        compile(source, '<wrapper>', 'exec')

    def test_the_attention_kernels_own_saved_state_is_not_pinned_to_one_backend(self):
        from evograd.pipelines.unified.codegen import _sdpa_state_outputs

        forward = 'torch.ops.aten._scaled_dot_product_cudnn_attention.default'
        graph = {'nodes': [
            {'op': 'placeholder', 'name': 'q'},
            {'op': 'call_function', 'name': 'attn', 'target': 'aten._scaled_dot_product_cudnn_attention.default',
             'external_kernel': forward},
            {'op': 'call_function', 'name': 'out', 'target': '<built-in function getitem>',
             'positional_args': [{'kind': 'node', 'name': 'attn'}, {'kind': 'scalar', 'value': 0}]},
            {'op': 'call_function', 'name': 'lse', 'target': '<built-in function getitem>',
             'positional_args': [{'kind': 'node', 'name': 'attn'}, {'kind': 'scalar', 'value': 1}]},
            {'op': 'call_function', 'name': 'other', 'target': 'aten.mul.Tensor',
             'positional_args': [{'kind': 'node', 'name': 'q'}, {'kind': 'node', 'name': 'q'}]},
            {'op': 'output', 'name': 'output', 'args': [['out', 'lse', 'q', 'other']]},
        ]}
        self.assertEqual(_sdpa_state_outputs(graph, 1), {'lse'})

    def test_a_flash_capture_renders_on_the_same_helpers_in_cudnn_order(self):
        from evograd.pipelines.unified.codegen import UNIFIED_SDPA_SOURCE, _sdpa_call

        flash_fwd = 'torch.ops.aten._scaled_dot_product_flash_attention.default'
        flash_bwd = 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default'
        self.assertEqual(_sdpa_call(flash_fwd, ['q', 'k', 'v', '0.0', 'True', 'False'], ['scale=0.1']),
                         '_unified_sdpa_forward(q, k, v, None, True, 0.0, True, False, scale=0.1)')
        # Trailing defaults the capture left out.
        self.assertEqual(_sdpa_call(flash_fwd, ['q', 'k', 'v', '0.0', 'True'], []),
                         '_unified_sdpa_forward(q, k, v, None, True, 0.0, True, False)')
        self.assertEqual(
            _sdpa_call(flash_bwd, ['g', 'q', 'k', 'v', 'o', 'lse', 'cq', 'ck', '2048', '2048', '0.0',
                                   'True', 'seed', 'off'], ['scale=0.1']),
            '_unified_sdpa_backward(g, q, k, v, o, lse, seed, off, None, cq, ck, 2048, 2048, 0.0, True, '
            'scale=0.1)')
        namespace = {'torch': torch}
        exec(compile(UNIFIED_SDPA_SOURCE, '<sdpa>', 'exec'), namespace)
        import inspect
        # The cuDNN capture passes seven positionals (return_debug_mask left to its default).
        forward = inspect.signature(namespace['_unified_sdpa_forward'])
        forward.bind('q', 'k', 'v', None, True, 0.0, True, scale=0.1)
        backward = inspect.signature(namespace['_unified_sdpa_backward'])
        backward.bind(*range(15), scale=0.1)


class TestGrantPrompt(unittest.TestCase):
    """A granted run's evolve system message documents the grant."""

    def test_the_evolve_prompt_documents_a_grant_and_is_unchanged_without_one(self):
        import yaml

        from evograd.benchmark import get_task
        from evograd.evolve.run import render_config

        op = get_task('rmsnorm')
        plain = yaml.safe_load(render_config(op))['prompt']['system_message']
        granted = yaml.safe_load(render_config(op, allowed_primitives=('vendor_gemm',)))
        granted = granted['prompt']['system_message']
        self.assertNotIn('vendor_gemm', plain)
        self.assertTrue(granted.startswith(plain.rstrip('\n')))
        self.assertIn('- `vendor_gemm`: 2-D row-major GEMM', granted)
        self.assertIn('call sites inside the EVOLVE-BLOCK are part of the search', granted)


class TestFixedConfiguration(unittest.TestCase):
    def test_explicit_equal_dimensions_are_preserved_exactly(self):
        from evograd.pipelines.unified.synthesize import select_workload

        workload = select_workload(
            get_task('layernorm'), dtype='bfloat16', dims={'rows': 31, 'hidden': 31},
        )
        self.assertEqual(workload.dims, {'rows': 31, 'hidden': 31})
        self.assertEqual(workload.dtype, 'bfloat16')

    def test_selection_uses_the_requested_dtype_and_rejects_partial_dimensions(self):
        from evograd.pipelines.unified.synthesize import select_workload

        op = replace(get_task('layernorm'), benchmark=SMALL['layernorm'])
        self.assertEqual(select_workload(op, dtype='float16'),
                         next(case for case in op.benchmark if case.dtype == 'float16'))
        for dims in ({'rows': 7}, {'rows': 0, 'hidden': 31}, {'rows': True, 'hidden': 31}):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                select_workload(op, dtype='float32', dims=dims)

    def test_a_declared_configuration_is_selected_even_with_a_zero_dim(self):
        # Padding is a dim: every patchify stem (ConvNeXt V2's, DiT's) is P=0.
        from evograd.pipelines.unified.synthesize import scoped_op, select_workload

        op = replace(get_task('spatial_conv2d'), benchmark=SMALL['spatial_conv2d'])
        declared = next(case for case in op.benchmark if case.dims['P'] == 0)
        self.assertEqual(select_workload(op, dtype='float32', dims=dict(declared.dims)), declared)
        for suite in ('convnextv2_tiny_observed', 'dit_s_2_observed'):
            stem = next(case for case in op.benchmark_suites[suite] if case.dims['P'] == 0)
            scoped = scoped_op(op, stem)
            with self.subTest(suite=suite):
                self.assertEqual(select_workload(scoped, dtype=stem.dtype, dims=dict(stem.dims)),
                                 stem)
        # An undeclared configuration is still held to positive dims.
        padded = next(case for case in op.benchmark if case.dims['P'] > 0)
        with self.assertRaises(ValueError):
            select_workload(op, dtype='float32', dims={**padded.dims, 'B': 0})
        undeclared = {**padded.dims, 'B': 3}
        self.assertEqual(select_workload(op, dtype='float32', dims=undeclared),
                         Workload(dims=undeclared, dtype='float32'))

    def test_scoped_scalars_are_what_the_configuration_passes(self):
        # A convolution's stride/padding/groups come from its dims, a model
        # norm's eps from its provenance: the scoped signature states them.
        from evograd.opdecl.inputs import case_scalars
        from evograd.pipelines.unified.synthesize import scoped_op

        conv = get_task('spatial_conv2d')
        stem = next(case for case in conv.benchmark_suites['convnextv2_tiny_observed']
                    if case.dims['S'] == 4)
        scoped = scoped_op(conv, stem)
        self.assertEqual({a.name: a.default for a in scoped.scalar_inactive_args()},
                         {'stride': 4, 'padding': 0, 'groups': 1})
        self.assertIn('stride=4, padding=0, groups=1', scoped.forward_parameters())
        self.assertEqual(case_scalars(scoped, stem), case_scalars(conv, stem))
        norm = get_task('layernorm')
        observed = next(case for case in norm.benchmark_suites['qwen2_vl_2b_observed']
                        if case.provenance.free.get('eps') == 1e-6)
        self.assertEqual(next(a.default for a in norm.args if a.name == 'eps'), 1e-5)
        self.assertEqual(next(a.default for a in scoped_op(norm, observed).args
                              if a.name == 'eps'), 1e-6)
        # Without a provenance the generator's generic value is the configuration's.
        plain = Workload(dims=dict(observed.dims), dtype=observed.dtype)
        self.assertEqual(case_scalars(norm, plain), {'eps': 1e-5})

    def test_verification_scope_keeps_the_oracle_at_the_selected_configuration(self):
        from evograd.pipelines.unified.synthesize import scoped_op, select_workload

        op = get_task('layernorm')
        workload = select_workload(op, dtype='bfloat16', dims={'rows': 7, 'hidden': 31})
        scoped = scoped_op(op, workload)
        self.assertEqual(scoped.forward, op.forward)
        self.assertEqual(scoped.make_inputs, op.make_inputs)
        for name in ('coverage', 'benchmark'):
            self.assertEqual(getattr(scoped, name), (workload,))

    def test_generated_declaration_roundtrips_the_exact_validation_configuration(self):
        from evograd.benchmark import load_task
        from evograd.pipelines.unified.synthesize import _write_declaration, select_workload

        op = get_task('layernorm')
        workload = select_workload(op, dtype='bfloat16', dims={'rows': 7, 'hidden': 31})
        with tempfile.TemporaryDirectory(prefix='evograd-unified-declaration-') as directory:
            path = Path(directory) / 'declaration.py'
            _write_declaration(path, op, workload)
            loaded = load_task(f'{path}:op')
            self.assertEqual(loaded.benchmark, (workload,))
            self.assertEqual(loaded.coverage, (workload,))
            self.assertEqual(loaded.make_inputs, op.make_inputs)
            self.assertEqual(loaded.forward, op.forward)
            self.assertEqual(loaded.declaration, f'{path}:op')

    def test_generated_declaration_keeps_the_workloads_provenance(self):
        # A recorded layout, a norm's eps and the model a generator reads its
        # tables from live in the provenance; the scoped case keeps all three.
        from evograd.benchmark import load_task
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.pipelines.unified.synthesize import _write_declaration, scoped_op

        op = get_task('rope')
        observed = op.benchmark_suites['qwen3_0_6b_observed'][0]
        self.assertEqual(observed.provenance.layout, 'head_major_view')
        with tempfile.TemporaryDirectory(prefix='evograd-unified-declaration-') as directory:
            path = Path(directory) / 'declaration.py'
            _write_declaration(path, op, observed)
            loaded = load_task(f'{path}:op')
        self.assertEqual(loaded.benchmark, (observed,))
        expected = make_case_inputs(scoped_op(op, observed), observed, device='meta')
        generated = make_case_inputs(loaded, loaded.benchmark[0], device='meta')
        self.assertEqual(generated['x'].stride(), expected['x'].stride())
        self.assertNotEqual(generated['x'].stride(), torch.empty(generated['x'].shape,
                                                                device='meta').stride())

    def test_evolution_entry_honors_the_scoped_declaration_for_a_registered_op(self):
        import evograd.evolve.evaluator as evaluator
        from evograd.pipelines.unified.synthesize import _write_declaration, select_workload

        op = get_task('layernorm')
        workload = select_workload(op, dtype='bfloat16', dims={'rows': 7, 'hidden': 31})
        with tempfile.TemporaryDirectory(prefix='evograd-unified-evaluator-') as directory:
            declaration = Path(directory) / 'declaration.py'
            _write_declaration(declaration, op, workload)
            entry_path = Path(evaluator.__file__).with_name('evaluator_entry.py')
            spec = importlib.util.spec_from_file_location('_unified_evaluator_test', entry_path)
            entry = importlib.util.module_from_spec(spec)
            environment = {'EVOGRAD_OP': op.name, 'EVOGRAD_DECLARATION': f'{declaration}:op'}
            # Without a task pack's search rule the entry refuses to start.
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(evaluator, 'build_evaluate', return_value=lambda _: None) as build, \
                    self.assertRaisesRegex(RuntimeError, 'EVOGRAD_SEARCH_CORRECTNESS is not set'):
                spec.loader.exec_module(entry)
            build.assert_not_called()
            environment['EVOGRAD_SEARCH_CORRECTNESS'] = str(Path(directory) / 'task.json')
            entry = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(evaluator, 'build_evaluate', return_value=lambda _: None) as build:
                spec.loader.exec_module(entry)
            selected = build.call_args.args[0]
            self.assertEqual(selected.benchmark, (workload,))
            self.assertEqual(selected.coverage, (workload,))
            self.assertNotIn('benchmark_workloads', build.call_args.kwargs)
            self.assertEqual(selected.make_inputs, op.make_inputs)


def _rendered(fn, *inputs):
    """``fn``'s ATen graph as the unified seed renders it: each node's expression,
    and the graph's outputs computed from them on CPU (layout and host nodes only)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder, _pyname
    from evograd.pipelines.unified.capture import _serialize_graph
    from evograd.pipelines.unified.codegen import _outputs, _semantic_expr

    graph = _serialize_graph(make_fx(fn)(*inputs))
    builder, values, expressions = _ProgramBuilder(), {}, {}
    placeholders = [n['name'] for n in graph['nodes'] if n['op'] == 'placeholder']
    values.update({_pyname(name): value for name, value in zip(placeholders, inputs)})
    for node in graph['nodes']:
        if node['op'] == 'call_function':
            expressions[node['target']] = expression = _semantic_expr(builder, node)
            try:
                values[_pyname(node['name'])] = eval(expression, {'torch': torch}, values)
            except (NameError, NotImplementedError):  # a Triton kernel's (CUDA-only) result
                pass
    return expressions, [values.get(_pyname(name)) for name in _outputs(graph)]


def _node(fn, *inputs, target):
    from torch.fx.experimental.proxy_tensor import make_fx

    from evograd.pipelines.unified.capture import _serialize_graph

    graph = _serialize_graph(make_fx(fn)(*inputs))
    return next(n for n in graph['nodes'] if n.get('target') == target)


def _nonlocals(kernel):
    import inspect

    return inspect.getclosurevars(kernel).nonlocals


class TestSeedLowerings(unittest.TestCase):
    """Semantic nodes the new benchmark cases reach, rendered without a GPU."""

    def test_unbind_and_split_with_sizes_render_as_views(self):
        # qkv.unbind(0) in Qwen2-VL's vision attention and DLRM's interaction;
        # in_proj splits in Qwen3.5's gated delta net.
        x = torch.randn(3, 4, 5)
        expressions, outputs = _rendered(lambda x: [*x.unbind(1), *torch.split(x, [1, 4], 2)], x)
        self.assertEqual(expressions['aten.unbind.int'], 't_x_1.unbind(1)')
        self.assertEqual(expressions['aten.split_with_sizes.default'], 't_x_1.split([1, 4], 2)')
        for got, want in zip(outputs, [*x.unbind(1), *torch.split(x, [1, 4], 2)], strict=True):
            self.assertTrue(torch.equal(got, want))

    def test_embedding_bag_sum_dispatches_to_handwritten_kernels(self):
        # DLRM's embedding tables: Inductor falls back to ATen, the seed lowers it.
        from evograd.atenir.primitive_triton import scatter_gather
        from evograd.atenir.primitive_triton.dispatch import make_kernel
        from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder
        from evograd.pipelines.unified.capture import _SEED_LOWERED_FALLBACKS

        self.assertLessEqual({'torch.ops.aten._embedding_bag.default',
                              'torch.ops.aten._embedding_bag_backward.default'},
                             _SEED_LOWERED_FALLBACKS)
        weight, indices, offsets = torch.randn(9, 4), torch.tensor([1, 2, 1]), torch.tensor([0, 2])
        bag = _node(torch.ops.aten._embedding_bag, weight, indices, offsets,
                    target='aten._embedding_bag.default')
        self.assertIs(make_kernel(bag), scatter_gather.embedding_bag_sum)
        mean = _node(lambda w, i, o: torch.ops.aten._embedding_bag(w, i, o, False, 1),
                     weight, indices, offsets, target='aten._embedding_bag.default')
        self.assertTrue(_ProgramBuilder()._is_generic_fallback(make_kernel(mean)))

    def test_an_integer_sort_lowers_to_the_handwritten_counting_sort(self):
        # OLMoE sorts its 32768 int64 expert ids (sort.default); sort.stable by keyword.
        import ast

        from evograd.atenir.primitive_triton.dispatch import make_kernel
        from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder
        from evograd.pipelines.unified.codegen import _audit_primitive_library, _semantic_expr

        keys = torch.randint(0, 5, (6, 4))
        for fn, options in ((lambda k: torch.sort(k), {'dim': -1, 'descending': False}),
                            (lambda k: torch.sort(k, dim=0, descending=True, stable=True),
                             {'dim': 0, 'descending': True})):
            node = _node(fn, keys, target=('aten.sort.stable' if options['dim'] == 0
                                           else 'aten.sort.default'))
            kernel = make_kernel(node)
            self.assertEqual(_nonlocals(kernel), options)
            builder = _ProgramBuilder()
            expression = _semantic_expr(builder, node)
            library = builder.render_kernel_library()
            _audit_primitive_library(ast.parse(library))
            self.assertIn('def integer_sort(', library)
            self.assertNotIn('torch.sort', expression + library)
        floats = make_kernel(_node(lambda a: torch.sort(a, dim=0, descending=True), torch.randn(8),
                                   target='aten.sort.default'))
        self.assertEqual(floats.__name__, 'triton_sort')

    def test_moe_routing_counts_offsets_and_inverse_permutation(self):
        # OLMoE: histc of the sorted ids, an int32 cumsum of the counts (the
        # grouped GEMM's offsets) and fresh storage for the inverse permutation.
        from evograd.atenir.primitive_triton.dispatch import make_kernel
        from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder

        histc = make_kernel(_node(lambda x: torch.histc(x, 64, 0, 63), torch.rand(8),
                                  target='aten.histc.default'))
        self.assertEqual({k: _nonlocals(histc)[k] for k in ('bins', 'low', 'high')},
                         {'bins': 64, 'low': 0, 'high': 63})
        data_range = _node(lambda x: torch.histc(x, 8), torch.rand(8), target='aten.histc.default')
        self.assertTrue(_ProgramBuilder()._is_generic_fallback(make_kernel(data_range)))
        cumsum = make_kernel(_node(lambda c: torch.cumsum(c, 0, dtype=torch.int32),
                                   torch.zeros(4, dtype=torch.int32), target='aten.cumsum.default'))
        self.assertEqual(_nonlocals(cumsum)['_dtype'], torch.int32)
        _, (empty,) = _rendered(lambda x: [torch.empty([5], dtype=torch.int64)], torch.randn(2))
        self.assertEqual((empty.shape, empty.dtype), ((5,), torch.int64))

    def test_grouped_gemm_is_preserved(self):
        from evograd.pipelines.unified.capture import _SUPPORTED_EXTERNALS
        from evograd.pipelines.unified.codegen import _external_wrapper

        self.assertEqual(_SUPPORTED_EXTERNALS['extern_kernels._grouped_mm'], 'aten._grouped_mm.default')
        grouped = {'name': 'gmm', 'external_kernel': 'extern_kernels._grouped_mm',
                   'target': 'aten._grouped_mm.default',
                   'positional_args': [{'kind': 'node', 'name': n} for n in ('x', 'w', 'offs')]}
        self.assertIn('return _unified_extern_kernels._grouped_mm(arg0, arg1, arg2)',
                      _external_wrapper('forward', grouped)[0])

    def test_float32_attention_is_preserved_with_its_state_unpinned(self):
        # DiT's float32 attention: memory-efficient attention, as vendor_sdpa runs it.
        from evograd.pipelines.unified.capture import _SUPPORTED_EXTERNALS
        from evograd.pipelines.unified.codegen import _external_wrapper, _sdpa_state_outputs

        forward = 'torch.ops.aten._scaled_dot_product_efficient_attention.default'
        backward = 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default'
        self.assertEqual(_SUPPORTED_EXTERNALS[backward],
                         'aten._scaled_dot_product_efficient_attention_backward.default')
        source, _, _ = _external_wrapper('forward', _EFFICIENT_NODE)
        self.assertIn('return torch.ops.aten._scaled_dot_product_efficient_attention.default('
                      'arg0, arg1, arg2, None, True)', source)
        getitem = '<built-in function getitem>'
        graph = {'nodes': [
            {'op': 'call_function', 'name': 'attn', 'external_kernel': forward},
            *({'op': 'call_function', 'name': f'g{i}', 'target': getitem,
               'positional_args': [{'kind': 'node', 'name': 'attn'}, {'kind': 'scalar', 'value': i}]}
              for i in range(4)),
            {'op': 'output', 'name': 'output', 'args': [['g0', 'g1', 'g2', 'g3']]},
        ]}
        # logsumexp and the rng state, which this backend keeps on the host.
        self.assertEqual(_sdpa_state_outputs(graph, 1), {'g1', 'g2', 'g3'})

    def test_strided_storage_zero_padding_flip_and_copy(self):
        # Qwen3.5's gated delta rule pads by zero, flips a chunk axis, and copies
        # into fresh strided storage; the copy is of the source, not the storage.
        from evograd.atenir.primitive_triton.dispatch import make_kernel
        from evograd.pipelines.b_dispatch.program_codegen import _ProgramBuilder
        from evograd.pipelines.unified.codegen import UnsupportedSemanticNode, _semantic_expr

        x = torch.randn(2, 3)
        _, (strided, padded) = _rendered(
            lambda x: [x.new_empty_strided((3, 2), (1, 3)), torch.constant_pad_nd(x, [0, 0, 0, 0])], x)
        self.assertEqual((strided.shape, strided.stride(), strided.dtype), ((3, 2), (1, 3), x.dtype))
        self.assertIs(padded, x)
        flipped = _node(lambda x: torch.ops.prims.rev(x, [1]), x, target='prims.rev.default')
        self.assertEqual(_semantic_expr(_ProgramBuilder(), flipped), 'flip_copy(t_x_1, [1])')
        padding = _node(lambda x: torch.constant_pad_nd(x, [1, 0]), x,
                        target='aten.constant_pad_nd.default')
        with self.assertRaisesRegex(UnsupportedSemanticNode, 'nonzero padding'):
            _semantic_expr(_ProgramBuilder(), padding)
        dst, src = torch.empty(2, 3, dtype=torch.float64), torch.randn(3)
        copy = make_kernel(_node(torch.ops.aten.copy, dst, src, target='aten.copy.default'))
        self.assertTrue(torch.equal(copy(dst, src), src.to(torch.float64).expand(2, 3)))


_EFFICIENT_NODE = {
    'name': 'attn', 'external_kernel': 'torch.ops.aten._scaled_dot_product_efficient_attention.default',
    'target': 'aten._scaled_dot_product_efficient_attention.default',
    'positional_args': [{'kind': 'node', 'name': n} for n in 'qkv']
    + [{'kind': 'scalar', 'value': None}, {'kind': 'scalar', 'value': True}],
}


#: Runs in a child process with TRITON_INTERPRET=1: the handwritten kernels on
#: CPU, against ATen's CPU kernels and the CUDA backward's summation order.
_INTERPRETED_KERNELS = r"""
import json
import triton.runtime.interpreter as interpreter
_patch = interpreter._patch_lang_tensor
def _patched(tensor, scope):
    # NumPy 2 refuses int() of the interpreter's one-element scalars, which a
    # data-dependent loop bound needs.
    _patch(tensor, scope)
    scope.set_attr(tensor, '__index__', lambda self: int(self.handle.data.reshape(-1)[0]))
interpreter._patch_lang_tensor = _patched
import torch
import torch.nn.functional as F
from evograd.atenir.primitive_triton import scatter_gather as sg

torch.manual_seed(0)
result = {}
weight, indices = torch.randn(50, 70), torch.randint(0, 50, (23,))
offsets, grad = torch.tensor([0, 3, 3, 10, 15, 22]), torch.randn(6, 70)
out, offset2bag, bag_size, max_indices = sg.embedding_bag_sum(weight, indices, offsets)
result['forward'] = torch.equal(out, F.embedding_bag(indices, weight, offsets, mode='sum'))
result['bags'] = (offset2bag.tolist() == [0] * 3 + [2] * 7 + [3] * 5 + [4] * 7 + [5]
                  and bag_size.tolist() == [3, 0, 7, 5, 7, 1] and max_indices.numel() == 0)
reference = weight.clone().requires_grad_(True)
F.embedding_bag(indices, reference, offsets, mode='sum').backward(grad)
result['backward'] = torch.equal(sg.embedding_bag_sum_backward(grad, indices, offset2bag, 50),
                                 reference.grad)
# 24 positions of row 7: the CUDA backward sums runs of ten, then the runs.
indices = torch.full((25,), 7)
indices[3] = 2
offsets, grad = torch.arange(0, 25, 5), torch.randn(5, 70) * 1e3
_, offset2bag, _, _ = sg.embedding_bag_sum(weight, indices, offsets)
dweight = sg.embedding_bag_sum_backward(grad, indices, offset2bag, 50)
rows = [grad[int(offset2bag[k])] for k in range(25) if int(indices[k]) == 7]
runs, flat = torch.zeros(70), torch.zeros(70)
for start in range(0, len(rows), 10):
    run = torch.zeros(70)
    for row in rows[start:start + 10]:
        run = run + row
    runs = runs + run
for row in rows:
    flat = flat + row
result['cuda_order'] = (torch.equal(dweight[7], runs) and not torch.equal(runs, flat)
                        and torch.equal(dweight[2], grad[0]) and int(dweight.abs().sum(1).gt(0).sum()) == 2)
ids = torch.randint(0, 66, (1000,))
result['histc_int'] = torch.equal(sg.histc(ids.to(torch.int32), 64, 0, 63),
                                  torch.histc(ids.float(), 64, 0, 63).to(torch.int32))
values = torch.randn(4000) * 3
result['histc_float'] = torch.equal(sg.histc(values, 37, -2.5, 4.0), torch.histc(values, 37, -2.5, 4.0))
from evograd.pipelines.b_dispatch.layout_kernels import flip_copy
a = torch.randn(3, 4, 5)
result['flip'] = all(torch.equal(flip_copy(t, dims), torch.flip(t, dims))
                     for t in (a, a.transpose(0, 2)) for dims in ([0], [2], [0, 2], [1, 2, 0]))
from evograd.atenir.primitive_triton.reduction import integer_sort
def stable_sort(name, keys, dim=-1, descending=False):
    values, order = integer_sort(keys, dim, descending)
    expected = torch.sort(keys, dim=dim, descending=descending, stable=True)
    result['sort_' + name] = (torch.equal(values, expected.values) and values.dtype == keys.dtype
                              and torch.equal(order, expected.indices) and order.dtype == torch.int64)
# OLMoE's 32768 expert ids; then ties, presorted runs and ragged blocks (256 keys each).
stable_sort('olmoe', torch.randint(0, 64, (32768,)))
stable_sort('duplicates', torch.randint(-2, 3, (2500,)))
stable_sort('all_equal', torch.full((1500,), 7))
stable_sort('sorted', torch.arange(-300, 1700))
stable_sort('reversed', torch.arange(1700, -300, -1))
stable_sort('descending', torch.randint(0, 9, (2049,)), descending=True)
stable_sort('int32', torch.randint(0, 40, (777,), dtype=torch.int32))
stable_sort('int8', torch.randint(-128, 128, (300,), dtype=torch.int8), descending=True)
stable_sort('bool', torch.rand(500) > 0.5)
stable_sort('rows', torch.randint(0, 9, (5, 70)))
stable_sort('dim0', torch.randint(0, 9, (70, 3)), dim=0, descending=True)
stable_sort('single', torch.tensor([4]))
stable_sort('empty', torch.empty(0, dtype=torch.int64))
for name, keys in (('wide_range', torch.tensor([0, 4096])), ('float_keys', torch.randn(4))):
    try:
        integer_sort(keys)
        result['refuses_' + name] = False
    except NotImplementedError:
        result['refuses_' + name] = True
print(json.dumps(result))
"""


class TestHandwrittenKernelsInterpreted(unittest.TestCase):
    def test_new_kernels_match_aten_bit_for_bit(self):
        import subprocess

        environment = {**os.environ, 'TRITON_INTERPRET': '1'}
        run = subprocess.run([sys.executable, '-c', _INTERPRETED_KERNELS], env=environment,
                             capture_output=True, text=True, timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])
        result = json.loads(run.stdout.strip().splitlines()[-1])
        self.assertTrue(result and all(result.values()), result)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestSeedLoweringsOnCuda(unittest.TestCase):
    """The new handwritten kernels and preserved calls against CUDA ATen."""

    def test_embedding_bag_matches_cuda_aten_bit_for_bit(self):
        import torch.nn.functional as F

        from evograd.atenir.primitive_triton import scatter_gather

        torch.manual_seed(0)
        weight = torch.randn(1000, 64, device='cuda')
        indices = torch.cat([torch.full((37,), 5), torch.randint(0, 1000, (91,))]).cuda()
        offsets = torch.arange(0, 128, 4, device='cuda')
        grad = torch.randn(32, 64, device='cuda')
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            try:
                reference = weight.clone().requires_grad_(True)
                out = F.embedding_bag(indices, reference, offsets, mode='sum')
                out.backward(grad)
                mine, offset2bag, _, _ = scatter_gather.embedding_bag_sum(weight, indices, offsets)
                dweight = scatter_gather.embedding_bag_sum_backward(grad, indices, offset2bag, 1000)
            finally:
                torch.use_deterministic_algorithms(False)
            with self.subTest(deterministic=deterministic):
                self.assertTrue(torch.equal(mine, out.detach()))
                self.assertTrue(torch.equal(dweight, reference.grad))

    def test_histc_matches_cuda_aten_on_integer_ids(self):
        from evograd.atenir.primitive_triton import scatter_gather

        ids = torch.randint(0, 66, (32768,), device='cuda', dtype=torch.int32)
        self.assertTrue(torch.equal(scatter_gather.histc(ids, 64, 0, 63), torch.histc(ids, 64, 0, 63)))

    def test_preserved_float32_attention_is_what_eager_runs(self):
        from evograd.pipelines.unified.codegen import _external_wrapper

        source, _, metadata = _external_wrapper('forward', _EFFICIENT_NODE)
        namespace = {'torch': torch}
        exec(source, namespace)
        q, k, v = (torch.randn(2, 6, 256, 64, device='cuda') for _ in range(3))
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            try:
                out = namespace[metadata['wrapper']](q, k, v)[0]
                eager = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            finally:
                torch.use_deterministic_algorithms(False)
            with self.subTest(deterministic=deterministic):
                self.assertTrue(torch.equal(out, eager))

    def test_flip_copy_matches_torch_flip(self):
        from evograd.pipelines.b_dispatch.layout_kernels import flip_copy

        a = torch.randn(2, 32, 64, device='cuda')
        self.assertTrue(torch.equal(flip_copy(a, [2]), torch.flip(a, [2])))

    def test_integer_sort_matches_cuda_stable_sort_bit_for_bit(self):
        from evograd.atenir.primitive_triton.reduction import integer_sort

        torch.manual_seed(0)
        for keys, dim, descending in (
                (torch.randint(0, 64, (32768,), device='cuda'), -1, False),  # OLMoE's expert ids
                (torch.randint(-3, 4, (32768,), device='cuda'), -1, True),
                (torch.full((5000,), 3, device='cuda', dtype=torch.int32), -1, False),
                (torch.arange(3000, 0, -1, device='cuda'), -1, False),
                (torch.randint(0, 9, (70, 33), device='cuda'), 0, False)):
            values, order = integer_sort(keys, dim, descending)
            expected = torch.sort(keys, dim=dim, descending=descending, stable=True)
            with self.subTest(n=keys.numel(), descending=descending):
                self.assertTrue(torch.equal(values, expected.values))
                self.assertTrue(torch.equal(order, expected.indices))

def matrix_square_ref(a, offset, b):
    """Keep a nondifferentiable tensor between active inputs in the ABI."""
    return (a @ b).square() * offset


def _matrix_op():
    return declare_op(
        name='unified_matrix_square',
        forward='tests.test_pipeline_unified:matrix_square_ref',
        dims=('M', 'K', 'N'),
        args=(Active('a', '[M, K]'), Inactive('offset', '[M, N]'), Active('b', '[K, N]')),
        output=Active('y', '[M, N]'),
        grad_order=('db', 'da'),
        forward_semantics='Square a dense matrix product and multiply by a fixed tensor.',
        backward_semantics='Return gradients in the declared db, da order.',
        benchmark=(Workload(dims={'M': 7, 'K': 9, 'N': 5}, dtype='float32'),),
    )


_RUNTIME_SOURCE = '''
import torch

UNIFIED_EXTERNAL_REGIONS = (
    {'wrapper': '_external_forward', 'side': 'forward'},
    {'wrapper': '_external_da', 'side': 'backward'},
    {'wrapper': '_external_db', 'side': 'backward'},
)

def _external_forward(a, b):
    return torch.mm(a, b)

def _external_da(dproduct, b):
    return torch.mm(dproduct, b.t())

def _external_db(a, dproduct):
    return torch.mm(a.t(), dproduct)

# EVOLVE-BLOCK-START
def _original_forward(a, offset, b):
    product = _external_forward(a, b)
    return product.square() * offset, (a, offset, b, product)

def _original_backward(dy, saved):
    a, offset, b, product = saved
    dproduct = dy * offset * 2 * product
    return _external_db(a, dproduct), _external_da(dproduct, b)

def _bypass_forward(a, offset, b):
    product = torch.einsum('mk,kn->mn', a, b)
    return product.square() * offset, (a, offset, b, product)

def unified_matrix_square_forward_with_saved(a, offset, b):
    return _original_forward(a, offset, b)

def unified_matrix_square_backward_from_saved(dy, saved_tensors):
    return _original_backward(dy, saved_tensors)
# EVOLVE-BLOCK-END
'''


class TestExecutedRegions(unittest.TestCase):
    def _module(self, source):
        directory = tempfile.TemporaryDirectory(prefix='evograd-unified-runtime-')
        self.addCleanup(directory.cleanup)
        name = f'_unified_runtime_test_{self._testMethodName}'
        module = _load_generated(source, directory.name, name)
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def test_valid_public_pair_executes_each_preserved_region_and_restores_wrappers(self):
        from evograd.pipelines.unified.execution import validate_external_execution

        contract = build_contract(_RUNTIME_SOURCE)
        module = self._module(_RUNTIME_SOURCE)
        original = {region['wrapper']: getattr(module, region['wrapper'])
                    for region in contract['external_regions']}
        validate_external_execution(_matrix_op(), module, contract, device='cpu')
        for name, wrapper in original.items():
            self.assertIs(getattr(module, name), wrapper)

    def test_numerically_equivalent_bypass_with_unused_original_driver_is_rejected(self):
        from evograd.pipelines.unified.execution import validate_external_execution

        op = _matrix_op()
        contract = build_contract(_RUNTIME_SOURCE)
        bypass = _RUNTIME_SOURCE.replace(
            'return _original_forward(a, offset, b)',
            'return _bypass_forward(a, offset, b)',
        )
        # The old driver still contains the one original extern call, so fixed
        # code and static call counts alone cannot distinguish this candidate.
        validate_source(bypass, contract)
        module = self._module(bypass)
        inputs = make_case_inputs(op, op.benchmark[0], device='cpu')
        inputs['offset'] = torch.rand_like(inputs['offset']) + 0.5
        expected_output, expected_gradients = _oracle(op, inputs)
        with torch.no_grad():
            output, saved = module.unified_matrix_square_forward_with_saved(
                inputs['a'], inputs['offset'], inputs['b'],
            )
            gradients = module.unified_matrix_square_backward_from_saved(inputs['dy'], saved)
        torch.testing.assert_close(output, expected_output, atol=2e-5, rtol=2e-5)
        for got, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)
        original = {region['wrapper']: getattr(module, region['wrapper'])
                    for region in contract['external_regions']}
        with self.assertRaises(ValueError):
            validate_external_execution(op, module, contract, device='cpu')
        for name, wrapper in original.items():
            self.assertIs(getattr(module, name), wrapper)


def _load_generated(source, directory, name):
    path = Path(directory) / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _oracle(op, inputs):
    active = {arg.name for arg in op.active_args()}
    values = {
        name: value.detach().clone().requires_grad_(name in active)
        if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }
    output = resolve_callable(op.forward)(*[values[arg.name] for arg in op.args])
    outputs = as_output_tuple(op, output)
    gradients = torch.autograd.grad(
        outputs, [values[arg.name] for arg in op.active_args()],
        grad_outputs=tuple(inputs[name] for name in op.upstream_grad_names),
    )
    by_name = {arg.grad_name: grad for arg, grad in zip(op.active_args(), gradients)}
    detached = tuple(value.detach() for value in outputs)
    return (detached if op.is_multi_output else detached[0]), tuple(by_name[name] for name in op.grad_names())


class _NoEagerPrimitiveMath(TorchDispatchMode):
    """A pure-R seed must execute its numerical work in generated kernels."""

    forbidden = {'add', 'sub', 'mul', 'div', 'pow', 'sum', 'mean', 'amax', 'amin',
                 'mm', 'bmm', 'addmm', 'matmul', 'exp', 'log', 'rsqrt', 'sqrt',
                 '_softmax', '_log_softmax', 'softmax', 'log_softmax', 'var',
                 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'gather',
                 'sort', 'cumsum', 'where'}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func).split('.')[1] in self.forbidden:
            raise AssertionError(f'pure-R seed executed framework math: {func}')
        return func(*args, **(kwargs or {}))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for unified seed acceptance')
class TestUnifiedCudaAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.pipelines.unified.capture import capture_unified_pair
        from evograd.pipelines.unified.codegen import generate_unified_seed

        cls.directory = tempfile.TemporaryDirectory(prefix='evograd-unified-tests-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.artifacts = {}
        layernorm = get_task('layernorm')
        configs = [(layernorm, Workload(dims={'rows': 7, 'hidden': 31}, dtype=dtype))
                   for dtype in ('float32', 'bfloat16')]
        matrix = _matrix_op()
        configs.append((matrix, matrix.benchmark[0]))
        configs.append((get_task('fused_add_rms_norm'), SMALL['fused_add_rms_norm'][0]))
        configs.append((get_task('spatial_conv2d'), SMALL['spatial_conv2d'][0]))
        configs.append((get_task('cross_entropy'),
                        Workload(dims={'rows': 7, 'cols': 31}, dtype='float32')))
        for op, workload in configs:
            captured = capture_unified_pair(op, workload)
            source = generate_unified_seed(op, workload, captured)
            name = f'_unified_test_{op.name}_{workload.dtype}'
            module = _load_generated(source, cls.directory.name, name)
            cls.addClassCleanup(sys.modules.pop, name, None)
            cls.artifacts[(op.name, workload.dtype)] = (op, workload, captured, source, module)

    def _assert_pair_and_deployment(self, artifact, inputs=None):
        op, workload, captured, source, module = artifact
        validate_artifact(op, module, source=source)
        validate_source(source, build_contract(source))
        inputs = inputs if inputs is not None else make_case_inputs(op, workload, device='cuda')
        if op.name == 'unified_matrix_square':
            # Generic Inactive inputs default to zero. A positive multiplier
            # ensures every matrix product and gradient contributes numerically.
            inputs['offset'] = torch.rand_like(inputs['offset']) + 0.5
        expected_output, expected_gradients = _oracle(op, inputs)
        args = [inputs[arg.name] for arg in op.args]
        scalar_kwargs = {arg.name: inputs[arg.name] for arg in op.scalar_inactive_args()}
        runtime_guard = _NoEagerPrimitiveMath() if not captured.external_regions else nullcontext()
        with torch.no_grad(), runtime_guard:
            output, saved = getattr(module, op.forward_fn_name)(*args)
            gradients = getattr(module, op.backward_fn_name)(
                upstream_grad_values(op, inputs), saved, **scalar_kwargs,
            )
        # assert_close at its dtype defaults: the rule Gate 1's model policy adopts.
        for got, expected in zip(as_output_tuple(op, output), as_output_tuple(op, expected_output)):
            torch.testing.assert_close(got, expected)
        self.assertEqual(len(gradients), len(expected_gradients))
        self.assertTrue(all(torch.is_tensor(value) for value in saved))
        for name, got, expected in zip(op.grad_names(), gradients, expected_gradients):
            with self.subTest(gradient=name):
                torch.testing.assert_close(got, expected)

        # The real deployment entry must wire the same gradients to the original
        # argument order, including an inactive tensor between active ones.
        deployed_args = [arg.detach().clone().requires_grad_(arg.is_floating_point()) if torch.is_tensor(arg) else arg
                         for arg in args]
        deployed_output = deployment_entry(module)(*deployed_args)
        torch.autograd.backward(as_output_tuple(op, deployed_output),
                                tuple(inputs[name] for name in op.upstream_grad_names))
        for got, expected in zip(as_output_tuple(op, deployed_output), as_output_tuple(op, expected_output)):
            torch.testing.assert_close(got, expected)
        by_name = dict(zip(op.grad_names(), expected_gradients))
        for declaration, arg in zip(op.args, deployed_args):
            if isinstance(declaration, Active):
                torch.testing.assert_close(arg.grad, by_name[declaration.grad_name])
            elif torch.is_tensor(arg):
                self.assertIsNone(arg.grad)

    def test_layernorm_forward_and_every_gradient_at_fixed_configuration(self):
        for dtype in ('float32', 'bfloat16'):
            with self.subTest(dtype=dtype):
                artifact = self.artifacts[('layernorm', dtype)]
                self.assertEqual(artifact[2].external_regions, ())
                self._assert_pair_and_deployment(artifact)

    def test_structured_outputs_receive_independent_tangents(self):
        artifact = self.artifacts[('fused_add_rms_norm', 'float32')]
        op, workload, captured, _, _ = artifact
        self.assertEqual(len(op.outputs), 2)
        self.assertEqual({i for kind, i in captured.pair.backward_arg_spec if kind == 't'}, {0, 1})
        for disabled in (None, *op.upstream_grad_names):
            inputs = make_case_inputs(op, workload, device='cuda')
            if disabled is not None:
                inputs[disabled] = torch.zeros_like(inputs[disabled])
            with self.subTest(zero_tangent=disabled):
                self._assert_pair_and_deployment(artifact, inputs)

    def test_convolution_preserves_both_library_phases_and_deployment_gradients(self):
        from evograd.pipelines.unified.execution import validate_external_execution
        from evograd.pipelines.unified.synthesize import scoped_op

        artifact = self.artifacts[('spatial_conv2d', 'float32')]
        op, workload, captured, source, module = artifact
        self.assertTrue(any(region.side == 'forward' and 'convolution' in region.target
                            for region in captured.external_regions))
        self.assertTrue(any(region.side == 'backward' and 'convolution_backward' in region.target
                            for region in captured.external_regions))
        validate_external_execution(scoped_op(op, workload), module, build_contract(source), device='cuda')
        self._assert_pair_and_deployment(artifact)

    def test_softmax_semantics_and_generated_helper_dependencies(self):
        artifact = self.artifacts[('cross_entropy', 'float32')]
        self.assertEqual(artifact[2].external_regions, ())
        self.assertTrue(any(node.get('target', '').startswith('aten.amax')
                            for node in artifact[2].forward_graph['nodes']))
        # Importing and running the standalone file exercises every copied
        # helper transitively, including amax, exp, sum and backward kernels.
        self._assert_pair_and_deployment(artifact)

    def test_public_pair_accepts_strided_inputs_and_tangents_at_fixed_shape(self):
        artifact = self.artifacts[('layernorm', 'float32')]
        op, workload, _, _, _ = artifact
        inputs = make_case_inputs(op, workload, device='cuda')
        for name in ('x', 'dy'):
            inputs[name] = inputs[name].t().contiguous().t()
            self.assertFalse(inputs[name].is_contiguous())
        self._assert_pair_and_deployment(artifact, inputs)

    def test_hybrid_seed_preserves_compiler_regions_and_gradient_order(self):
        from evograd.pipelines.unified.execution import validate_external_execution

        artifact = self.artifacts[('unified_matrix_square', 'float32')]
        op, workload, captured, source, module = artifact
        self.assertTrue(captured.external_regions, 'dense GEMMs must have compiler evidence')
        self.assertEqual({r.side for r in captured.external_regions}, {'forward', 'backward'})
        for region in captured.external_regions:
            graph = getattr(captured, f'{region.side}_graph')
            node = next(n for n in graph['nodes'] if n.get('name') == region.node_name)
            self.assertEqual(node['external_kernel'], region.kernel)
            self.assertEqual(node['target'], region.target)
        self.assertEqual(len(module.UNIFIED_EXTERNAL_REGIONS), len(captured.external_regions))
        validate_external_execution(op, module, build_contract(source), device='cuda')
        self._assert_pair_and_deployment(artifact)

        # A direct pair can receive grad-bearing tensors without an enclosing
        # no_grad context. Preserved out= GEMMs must suppress eager autograd
        # locally; otherwise PyTorch rejects an otherwise valid pair call.
        inputs = make_case_inputs(op, workload, device='cuda')
        inputs['offset'] = torch.rand_like(inputs['offset']) + 0.5
        expected_output, expected_gradients = _oracle(op, inputs)
        args = [inputs[arg.name].detach().clone().requires_grad_(True) for arg in op.args]
        output, saved = getattr(module, op.forward_fn_name)(*args)
        gradients = getattr(module, op.backward_fn_name)(inputs[op.upstream_grad_name], saved)
        torch.testing.assert_close(output, expected_output)
        self.assertFalse(output.requires_grad)
        for got, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(got, expected)
            self.assertFalse(got.requires_grad)

    def test_layernorm_rejects_a_different_shape_dtype_or_static_scalar(self):
        op, workload, _, _, module = self.artifacts[('layernorm', 'float32')]
        inputs = make_case_inputs(op, workload, device='cuda')
        original = [inputs[arg.name] for arg in op.args]
        variants = [list(original) for _ in range(3)]
        variants[0][0] = original[0][:-1]
        variants[1][0] = original[0].to(torch.float16)
        variants[2][-1] = 1e-3
        for name, args in zip(('shape', 'dtype', 'eps'), variants):
            with self.subTest(changed=name), self.assertRaises((ValueError, NotImplementedError)):
                getattr(module, op.forward_fn_name)(*args)


class SeedVerificationHasTheDevice(unittest.TestCase):
    def test_the_capture_is_released_before_the_child_verifies(self):
        import gc
        import weakref
        from unittest import mock

        from evograd.pipelines.unified import synthesize as S

        class Captured:
            joint_graph = forward_graph = backward_graph = {"nodes": []}
            external_regions = ()
            pair = SimpleNamespace(forward_source="", backward_source="", device="cuda:0", baked_scalars={},
                                   saved=(), backward_arg_spec=(), grad_indices=())

        captured = Captured()
        alive = weakref.ref(captured)
        seen = []

        def capture(*args, **kwargs):
            nonlocal captured
            held, captured = captured, None
            return held

        def verify(**kwargs):
            gc.collect()
            seen.append(alive() is None)
            return {}

        op = SimpleNamespace(name="op", declaration=None, forward="m:f", validate=lambda: None)
        config = S.UnifiedSeedConfig(op=op, workload=SimpleNamespace(dims={}, dtype="float32"),
                                     output_dir=Path(tempfile.mkdtemp()))
        with mock.patch("evograd.pipelines.unified.capture.capture_unified_pair", capture), \
                mock.patch("evograd.pipelines.unified.codegen.generate_unified_seed", lambda *args: ""), \
                mock.patch("evograd.pipelines.shared.primitives.check_source"), \
                mock.patch.object(S, "select_workload"), mock.patch.object(S, "scoped_op"), \
                mock.patch.object(S, "build_contract", return_value={}), \
                mock.patch.object(S, "validate_source"), mock.patch.object(S, "verify_candidate", verify), \
                mock.patch.object(S, "report_passed", return_value=False), \
                mock.patch.object(S, "_write_declaration"), \
                mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.get_device_name", return_value="gpu"), \
                mock.patch("torch.cuda.empty_cache"):
            S.synthesize_unified_seed(config)
        self.assertEqual(seen, [True])

if __name__ == '__main__':
    unittest.main()
