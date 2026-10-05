"""Numerical regression cases for the handwritten pointwise lowerers."""

import unittest

import torch

from evograd.atenir.primitive_triton import elementwise as ew


class TestElementwiseMetadata(unittest.TestCase):
    def test_scalar_tensor_promotion_is_resolved_before_broadcasting(self):
        scalar = torch.tensor(0.125, dtype=torch.float32)
        vector = torch.tensor([0.5, 1.0], dtype=torch.float16)
        a, b = ew._binary_inputs(scalar, vector)
        self.assertEqual(a.dtype, (scalar + vector).dtype)
        self.assertEqual(b.dtype, a.dtype)
        self.assertEqual(a.shape, vector.shape)

    def test_integral_true_division_uses_the_default_floating_dtype(self):
        x = torch.tensor([1, 2], dtype=torch.int32)
        y = torch.tensor([2, 3], dtype=torch.int64)
        a, b = ew._binary_inputs(x, y, true_divide=True)
        self.assertEqual(a.dtype, (x / y).dtype)
        self.assertEqual(b.dtype, a.dtype)

    def test_invalid_alpha_types_match_pytorch_rejection(self):
        for dtype, alpha in ((torch.int32, 0.5), (torch.float32, True),
                             (torch.float32, 1j)):
            with self.subTest(dtype=dtype, alpha=alpha):
                x = torch.ones(2, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    torch.add(x, x, alpha=alpha)
                with self.assertRaises(RuntimeError):
                    ew._check_alpha(alpha, dtype)


@unittest.skipUnless(torch.cuda.is_available() and ew.HAS_TRITON, "CUDA and Triton required")
class TestElementwiseCuda(unittest.TestCase):
    def assert_values(self, actual, expected):
        torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_bitwise_not_preserves_boolean_and_integer_semantics(self):
        mask = torch.tensor([[True, False, True], [False, True, False]], device="cuda").t()
        self.assert_values(ew.bitwise_not(mask), torch.bitwise_not(mask))
        for dtype in (torch.int32, torch.int64):
            values = torch.tensor([-7, -1, 0, 1, 7], device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assert_values(ew.bitwise_not(values), torch.bitwise_not(values))

    def test_fma_has_no_rounded_intermediate_product(self):
        for dtype, delta in ((torch.float32, 2**-12), (torch.float16, 2**-5),
                             (torch.bfloat16, 2**-3)):
            a = torch.tensor([[1 + delta, 2, -3]], device="cuda", dtype=dtype)
            b = torch.tensor([[1 - delta], [2]], device="cuda", dtype=dtype)
            c = torch.tensor(-1, device="cuda", dtype=dtype)
            expected = (a.double() * b.double() + c.double()).to(dtype)
            with self.subTest(dtype=dtype):
                torch.testing.assert_close(ew.fma(a, b, c), expected, atol=0, rtol=0)

    def test_causal_mask_inversion_and_masked_softmax_stay_finite(self):
        from evograd.atenir.primitive_triton import reduction

        positions = ew.arange(9)
        causal = ew.ge_tt(positions[:, None], positions[None, :])
        blocked = ew.bitwise_not(causal)
        softmax = reduction.make_softmax_kernel(-1)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.linspace(-2, 2, 81, device="cuda", dtype=dtype).reshape(9, 9)
            masked = ew.where(blocked, ew.fill_like(logits, float("-inf")), logits)
            actual = softmax(masked)
            expected = logits.masked_fill(~causal, float("-inf")).float().softmax(-1).to(dtype)
            with self.subTest(dtype=dtype):
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assert_values(actual, expected)
                self.assertTrue(bool((actual[~causal] == 0).all()))

    def test_scalar_power_handles_negative_bases_zero_and_negative_exponents(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.tensor([-2., -1., -0., 0., 1., 2., float('inf'), float('nan')],
                             device="cuda", dtype=dtype)
            for exponent in (0, 1, 2, 3, 4, 5, -1, -2, -3, 0.5):
                with self.subTest(dtype=dtype, exponent=exponent):
                    self.assert_values(ew.pow_scalar(x, exponent), torch.pow(x, exponent))

    def test_tensor_power_handles_negative_integer_exponents_and_broadcast(self):
        x = torch.tensor([-2., -0., 0., 2.], device="cuda").reshape(4, 1)
        exponent = torch.tensor([-3., -2., 0., 2., 4., 0.5], device="cuda").reshape(1, 6)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a, b = x.to(dtype), exponent.to(dtype)
                self.assert_values(ew.pow_tt(a, b), torch.pow(a, b))

    def test_scalar_base_power_handles_nonpositive_bases(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            exponent = torch.tensor([-3., -2., 0., 1., 2., 3., 0.5], device="cuda", dtype=dtype)
            for base in (-2., -0., 0., 2.):
                with self.subTest(dtype=dtype, base=base):
                    self.assert_values(ew.pow_scalar_base(base, exponent), torch.pow(base, exponent))

    def test_low_precision_cubes_have_no_rounded_intermediate_product(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.linspace(-3, 3, 257, device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assert_values(ew.pow_scalar(x, 3), torch.pow(x, 3))

    def test_binary_promotion_broadcasting_and_noncontiguous_inputs(self):
        functions = ((ew.add_tt, torch.add), (ew.sub_tt, torch.sub),
                     (ew.mul_tt, torch.mul), (ew.div_tt, torch.div))
        for left_dtype, right_dtype in (
            (torch.float16, torch.float32), (torch.float32, torch.float16),
            (torch.bfloat16, torch.float32), (torch.float16, torch.bfloat16),
            (torch.int32, torch.float32), (torch.int32, torch.int64),
        ):
            a = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda", dtype=left_dtype).t()
            b = torch.tensor([[2, 3]], device="cuda", dtype=right_dtype)
            for lower, reference in functions:
                with self.subTest(op=lower.__name__, left=left_dtype, right=right_dtype):
                    self.assert_values(lower(a, b), reference(a, b))
        scalar = torch.tensor(0.0001, device="cuda", dtype=torch.float32)
        vector = torch.tensor([0.125, 0.5], device="cuda", dtype=torch.float16)
        for lower, reference in functions:
            with self.subTest(op=lower.__name__, scalar_tensor=True):
                self.assert_values(lower(vector, scalar), reference(vector, scalar))

    def test_add_sub_alpha_uses_promoted_opmath_without_intermediate_rounding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            a = torch.tensor([0.01, -2., 3000.], device="cuda", dtype=dtype)
            b = torch.tensor([0.123, 0.499, -2990.], device="cuda", dtype=dtype)
            for alpha in (0, 2, 0.75):
                for lower, reference in ((ew.add_tt, torch.add), (ew.sub_tt, torch.sub)):
                    with self.subTest(dtype=dtype, alpha=alpha, op=lower.__name__):
                        self.assert_values(lower(a, b, alpha=alpha), reference(a, b, alpha=alpha))

    def test_scalar_arithmetic_preserves_promotion_and_alpha(self):
        for dtype in (torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32):
            x = torch.tensor([-3, 1, 7], device="cuda", dtype=dtype)
            for scalar in (2, 0.75):
                for lower, reference in ((ew.add_scalar, torch.add), (ew.sub_scalar, torch.sub),
                                         (ew.mul_scalar, torch.mul), (ew.div_scalar, torch.div)):
                    with self.subTest(dtype=dtype, scalar=scalar, op=lower.__name__):
                        self.assert_values(lower(x, scalar), reference(x, scalar))
                for alpha in (0, 3):
                    with self.subTest(dtype=dtype, scalar=scalar, alpha=alpha):
                        self.assert_values(ew.add_scalar(x, scalar, alpha), torch.add(x, scalar, alpha=alpha))
                        self.assert_values(ew.sub_scalar(x, scalar, alpha), torch.sub(x, scalar, alpha=alpha))
                        self.assert_values(ew.rsub_scalar(x, scalar, alpha), torch.rsub(x, scalar, alpha=alpha))

    def test_constant_allocation_and_integer_ranges_use_triton(self):
        for dtype, value in ((torch.float32, 1.25), (torch.float16, -2.),
                             (torch.bfloat16, 3.), (torch.int64, 2**40 + 1),
                             (torch.bool, True)):
            with self.subTest(dtype=dtype):
                expected = torch.full((3, 5), value, device="cuda", dtype=dtype)
                self.assert_values(ew.fill_tensor((3, 5), value, dtype=dtype), expected)
                self.assert_values(ew.fill_like(expected, value), expected)
                self.assert_values(ew.fill_tensor((0, 3), value, dtype=dtype), expected[:0, :3])
        for start, end, step in ((0, 9, 1), (10, -2, -3), (3, 3, 1),
                                 (2**40, 2**40 + 7, 2), (0., 1., .125)):
            with self.subTest(start=start, end=end, step=step):
                self.assert_values(ew.arange(start, end, step), torch.arange(start, end, step, device="cuda"))


if __name__ == "__main__":
    unittest.main()
