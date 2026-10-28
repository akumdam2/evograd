"""Compiler-guided primitive synthesis, fixed providers, and the training ABI.

The CPU tests check preservation and graph serialization. CUDA tests exercise
real compiler capture and generated Triton against an independent autograd
reference; they skip on login nodes without a GPU.
"""

from __future__ import annotations

import importlib.util
import ast
from contextlib import nullcontext
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

        op = get_task('layernorm')
        self.assertEqual(select_workload(op, dtype='float16'),
                         next(case for case in op.correctness if case.dtype == 'float16'))
        for dims in ({'rows': 7}, {'rows': 0, 'hidden': 31}, {'rows': True, 'hidden': 31}):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                select_workload(op, dtype='float32', dims=dims)

    def test_a_declared_configuration_is_selected_even_with_a_zero_dim(self):
        # Padding is a dim: every patchify stem (ConvNeXt V2's, DiT's) is P=0.
        from evograd.pipelines.unified.synthesize import scoped_op, select_workload

        op = get_task('spatial_conv2d')
        declared = next(case for case in op.correctness if case.dims['P'] == 0)
        self.assertEqual(select_workload(op, dtype='float32', dims=dict(declared.dims)), declared)
        for suite in ('convnextv2_tiny_observed', 'dit_s_2_observed'):
            stem = next(case for case in op.benchmark_suites[suite] if case.dims['P'] == 0)
            scoped = scoped_op(op, stem)
            with self.subTest(suite=suite):
                self.assertEqual(select_workload(scoped, dtype=stem.dtype, dims=dict(stem.dims)),
                                 stem)
        # An undeclared configuration is still held to positive dims.
        padded = next(case for case in op.correctness if case.dims['P'] > 0)
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

    def test_verification_scope_keeps_oracle_and_tolerances_at_the_selected_configuration(self):
        from evograd.pipelines.unified.synthesize import scoped_op, select_workload

        op = get_task('layernorm')
        workload = select_workload(op, dtype='bfloat16', dims={'rows': 7, 'hidden': 31})
        scoped = scoped_op(op, workload)
        self.assertEqual(scoped.forward, op.forward)
        self.assertEqual(scoped.make_inputs, op.make_inputs)
        self.assertEqual(scoped.tolerances, op.tolerances)
        for name in ('correctness', 'coverage', 'benchmark'):
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
            self.assertEqual(loaded.correctness, (workload,))
            self.assertEqual(loaded.benchmark, (workload,))
            self.assertEqual(loaded.tolerances, op.tolerances)
            self.assertEqual(loaded.tolerance_hook, op.tolerance_hook)
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
        self.assertEqual(loaded.correctness, (observed,))
        expected = make_case_inputs(scoped_op(op, observed), observed, device='meta')
        generated = make_case_inputs(loaded, loaded.correctness[0], device='meta')
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
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(evaluator, 'build_evaluate', return_value=lambda _: None) as build:
                spec.loader.exec_module(entry)
            selected = build.call_args.args[0]
            self.assertEqual(selected.correctness, (workload,))
            self.assertEqual(build.call_args.kwargs['benchmark_workloads'], (workload,))
            self.assertEqual(selected.tolerances, op.tolerances)


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
        correctness=(Workload(dims={'M': 7, 'K': 9, 'N': 5}, dtype='float32'),),
        tolerances={'float32': (2e-5, 2e-5)},
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
        inputs = make_case_inputs(op, op.correctness[0], device='cpu')
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
        configs.append((matrix, matrix.correctness[0]))
        fused = get_task('fused_add_rms_norm')
        configs.append((fused, fused.correctness[0]))
        convolution = get_task('conv2d')
        configs.append((convolution, convolution.correctness[0]))
        configs.append((get_task('softmax'), Workload(dims={'rows': 7, 'cols': 31}, dtype='float32')))
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
        for name, got, expected in zip(op.output_names, as_output_tuple(op, output), as_output_tuple(op, expected_output)):
            atol, rtol = op.tolerance_for(workload, name)
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
        self.assertEqual(len(gradients), len(expected_gradients))
        self.assertTrue(all(torch.is_tensor(value) for value in saved))
        for name, got, expected in zip(op.grad_names(), gradients, expected_gradients):
            with self.subTest(gradient=name):
                atol, rtol = op.tolerance_for(workload, name)
                torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)

        # The real deployment entry must wire the same gradients to the original
        # argument order, including an inactive tensor between active ones.
        deployed_args = [arg.detach().clone().requires_grad_(arg.is_floating_point()) if torch.is_tensor(arg) else arg
                         for arg in args]
        deployed_output = deployment_entry(module)(*deployed_args)
        torch.autograd.backward(as_output_tuple(op, deployed_output),
                                tuple(inputs[name] for name in op.upstream_grad_names))
        for name, got, expected in zip(op.output_names, as_output_tuple(op, deployed_output), as_output_tuple(op, expected_output)):
            atol, rtol = op.tolerance_for(workload, name)
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
        by_name = dict(zip(op.grad_names(), expected_gradients))
        for declaration, arg in zip(op.args, deployed_args):
            if isinstance(declaration, Active):
                atol, rtol = op.tolerance_for(workload, declaration.grad_name)
                torch.testing.assert_close(arg.grad, by_name[declaration.grad_name], atol=atol, rtol=rtol)
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

        artifact = self.artifacts[('conv2d', 'float32')]
        op, workload, captured, source, module = artifact
        self.assertTrue(any(region.side == 'forward' and 'convolution' in region.target
                            for region in captured.external_regions))
        self.assertTrue(any(region.side == 'backward' and 'convolution_backward' in region.target
                            for region in captured.external_regions))
        validate_external_execution(scoped_op(op, workload), module, build_contract(source), device='cuda')
        self._assert_pair_and_deployment(artifact)

    def test_softmax_semantics_and_generated_helper_dependencies(self):
        artifact = self.artifacts[('softmax', 'float32')]
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
        atol, rtol = op.tolerances[workload.dtype]
        torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
        self.assertFalse(output.requires_grad)
        for got, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
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


if __name__ == '__main__':
    unittest.main()
