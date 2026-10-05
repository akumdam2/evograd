"""Numerical and fallback regressions for handwritten reductions and indexing."""
from __future__ import annotations

import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from evograd.atenir.primitive_triton import reduction, scatter_gather


class _NoFrameworkMath(TorchDispatchMode):
    forbidden = {'sum', 'mean', 'amax', 'amin', 'max', 'min', 'argmax', 'argmin',
                 '_softmax', '_log_softmax', 'softmax', 'log_softmax', 'var',
                 'cumsum', 'sort', 'scatter', 'scatter_', 'scatter_add',
                 'scatter_add_', 'gather', 'slice_scatter', 'select_scatter'}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func).split('.')[1] in self.forbidden:
            raise AssertionError(f'handwritten path called framework math: {func}')
        return func(*args, **(kwargs or {}))


class TestReductionShapePlanning(unittest.TestCase):
    def test_negative_axes_and_noncontiguous_inputs_preserve_element_groups(self):
        a = torch.arange(24).reshape(2, 3, 4).transpose(0, 1)
        matrix, shape = reduction._reduction_plan(a, [0, -1], True)
        self.assertEqual(shape, (1, 2, 1))
        torch.testing.assert_close(matrix.sum(1).reshape(shape), a.sum((0, -1), keepdim=True))

    def test_invalid_and_duplicate_axes_raise(self):
        for dims in ([3], [-4], [1, -2]):
            with self.subTest(dims=dims), self.assertRaises((IndexError, ValueError)):
                reduction._reduction_plan(torch.empty(2, 3, 4), dims, False)

    def test_scalar_reduction_shape_is_scalar(self):
        matrix, shape = reduction._reduction_plan(torch.tensor(3.0), [0], True)
        self.assertEqual(tuple(matrix.shape), (1, 1))
        self.assertEqual(shape, ())

    def test_select_scatter_rejects_indices_outside_both_ends(self):
        base, source = torch.empty(2, 3), torch.empty(2)
        for index in (-4, 3):
            with self.subTest(index=index), self.assertRaises(IndexError):
                scatter_gather.select_scatter(base, source, 1, index)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for primitive correctness')
class TestHandwrittenReductionCuda(unittest.TestCase):
    def test_half_mean_divides_before_casting_the_sum(self):
        x = torch.full((2, 100), 1000.0, device='cuda', dtype=torch.float16)
        with _NoFrameworkMath():
            actual = reduction.make_mean_kernel([1], True)(x)
        torch.testing.assert_close(actual, x.mean(1, keepdim=True), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_nd_multiple_axes_and_explicit_dtype(self):
        x = torch.randn(3, 5, 7, device='cuda', dtype=torch.float16).transpose(0, 1)
        for factory, oracle in ((reduction.make_sum_kernel, torch.sum),
                                (reduction.make_mean_kernel, torch.mean)):
            for keepdim in (False, True):
                expected = oracle(x, dim=(0, 2), keepdim=keepdim, dtype=torch.float32)
                with _NoFrameworkMath():
                    actual = factory([0, -1], keepdim, dtype=torch.float32)(x)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_boolean_sum_promotes_to_int64(self):
        x = torch.rand(3, 5, 7, device='cuda') > 0.4
        with _NoFrameworkMath():
            actual = reduction.make_sum_kernel([0, 2], False)(x)
        torch.testing.assert_close(actual, x.sum((0, 2)))
        self.assertEqual(actual.dtype, torch.int64)

    def test_long_sum_and_softmax_do_not_drop_the_last_element(self):
        x = torch.ones((2, 65537), device='cuda')
        with _NoFrameworkMath():
            actual = reduction.make_sum_kernel([1], False)(x)
        torch.testing.assert_close(actual, x.sum(1), atol=0, rtol=0)
        x.zero_()
        x[:, -1] = 20
        with _NoFrameworkMath():
            actual = reduction.make_softmax_kernel(-1)(x)
        torch.testing.assert_close(actual, torch.softmax(x, -1), atol=2e-6, rtol=2e-5)

    def test_softmax_and_log_softmax_on_a_nonfinal_axis(self):
        x = torch.randn(3, 5, 7, device='cuda')
        for factory, oracle in ((reduction.make_softmax_kernel, torch.softmax),
                                (reduction.make_log_softmax_kernel, torch.log_softmax)):
            with _NoFrameworkMath():
                actual = factory(1)(x)
            torch.testing.assert_close(actual, oracle(x, 1), atol=2e-6, rtol=2e-5)

    def test_extrema_handle_multiple_axes_nan_and_first_tie(self):
        x = torch.randn(3, 5, 7, device='cuda')
        x[0, 0, 0] = float('nan')
        for factory, oracle in ((reduction.make_amax_kernel, torch.amax),
                                (reduction.make_amin_kernel, torch.amin)):
            with _NoFrameworkMath():
                actual = factory([0, 2], True)(x)
            torch.testing.assert_close(actual, oracle(x, (0, 2), keepdim=True), equal_nan=True)
        ties = torch.tensor([[2., 3., 3., 1.], [1., float('nan'), float('nan'), 0.]], device='cuda')
        with _NoFrameworkMath():
            values, indices = reduction.make_max_dim_kernel(1, False)(ties)
        expected = torch.max(ties, 1)
        torch.testing.assert_close(values, expected.values, equal_nan=True)
        torch.testing.assert_close(indices, expected.indices)

    def test_variance_and_cumsum_cover_nd_and_long_rows(self):
        x = torch.randn(3, 5, 7, device='cuda') * 0.1 + 1000
        with _NoFrameworkMath():
            actual = reduction.make_var_kernel([0, 2], True, correction=1)(x)
        torch.testing.assert_close(actual, torch.var(x, (0, 2), correction=1, keepdim=True), atol=2e-6, rtol=2e-4)
        x = torch.ones(2, 65537, device='cuda', dtype=torch.int32)
        with _NoFrameworkMath():
            actual = reduction.make_cumsum_kernel(1)(x)
        torch.testing.assert_close(actual, torch.cumsum(x, 1))

    def test_sort_preserves_values_indices_ties_and_nan(self):
        x = torch.randn(3, 31, device='cuda')
        x[:, 1] = x[:, 0]
        x[0, 4] = float('nan')
        for descending in (False, True):
            with _NoFrameworkMath():
                values, indices = reduction.make_sort_kernel(1, descending, stable=True)(x)
            expected = torch.sort(x, dim=1, descending=descending, stable=True)
            torch.testing.assert_close(values, expected.values, equal_nan=True)
            torch.testing.assert_close(indices, expected.indices)

    def test_nd_scatter_add_gather_and_scalar_scatter_have_no_framework_fallback(self):
        base = torch.randn(2, 4, 5, device='cuda')
        index = torch.randint(0, 4, (2, 3, 5), device='cuda')
        src = torch.randn(2, 3, 5, device='cuda')
        with _NoFrameworkMath():
            actual_add = scatter_gather.scatter_add(base, 1, index, src)
            actual_gather = scatter_gather.gather(base, 1, index)
            actual_value = scatter_gather.scatter_value(base, 1, index, -2.5)
        torch.testing.assert_close(actual_add, torch.scatter_add(base, 1, index, src))
        torch.testing.assert_close(actual_gather, torch.gather(base, 1, index))
        torch.testing.assert_close(actual_value, torch.scatter(base, 1, index, -2.5))

    def test_slice_scatter_preserves_unselected_values(self):
        base = torch.randn(2, 9, 3, device='cuda')
        source = torch.randn(2, 3, 3, device='cuda')
        with _NoFrameworkMath():
            actual = scatter_gather.slice_scatter(base, source, 1, 1, 7, 2)
        torch.testing.assert_close(actual, torch.slice_scatter(base, source, 1, 1, 7, 2))

    def test_scatter_source_restores_a_sort_permutation(self):
        source = torch.randn(2, 3, 7, device='cuda')
        values, index = torch.sort(source, dim=2)
        base = torch.empty_like(source)
        with _NoFrameworkMath():
            actual = scatter_gather.scatter_src(base, 2, index, values)
        torch.testing.assert_close(actual, source)

    def test_select_scatter_handles_negative_indices_and_strided_tensors(self):
        for dtype in (torch.float32, torch.float16):
            base = torch.randn(2, 4, 5, device='cuda', dtype=dtype).transpose(0, 1)
            source = torch.randn(5, 4, device='cuda', dtype=dtype).t()
            for index in (0, -1):
                with self.subTest(dtype=dtype, index=index), _NoFrameworkMath():
                    actual = scatter_gather.select_scatter(base, source, -2, index)
                torch.testing.assert_close(actual, torch.select_scatter(base, source, -2, index))


if __name__ == '__main__':
    unittest.main()
