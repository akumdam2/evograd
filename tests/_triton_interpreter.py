"""Make Triton's CPU interpreter (TRITON_INTERPRET=1) run the handwritten kernels
as the GPU does, for bit-for-bit checks on a login node. Import before any kernel
launches; the patches are idempotent.

* A one-element interpreter tensor converts to a Python int, which a
  data-dependent loop bound (``range(tensor)``) needs under NumPy 2.
* Conversions to and from bfloat16 round as compiled Triton does (to nearest,
  ties to even, as LLVM's fptrunc and sitofp): the interpreter truncates a
  float32 and reinterprets the bits of every other source dtype.
* bfloat16 arithmetic, comparisons, dots and reductions run on the values,
  in float32, each result rounded to bfloat16 as the GPU rounds one bfloat16
  operation: the interpreter computes on the raw uint16 bit patterns. (A
  reduction's own order is the interpreter's, as for any dtype.)
* A Python bool argument becomes an int1 scalar (the interpreter rejects it).

The interpreter cannot call CUDA's libdevice; :data:`libdevice` stands in for
``triton.language.extra.cuda.libdevice`` in a program run here, each function
on PyTorch's CPU kernel (rounding may differ from libdevice by an ulp).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import triton.language as tl
import triton.runtime.interpreter as interpreter

_TORCH_DTYPES = {
    tl.float64: torch.float64, tl.float32: torch.float32, tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16, tl.int64: torch.int64, tl.int32: torch.int32,
    tl.int16: torch.int16, tl.int8: torch.int8, tl.uint8: torch.uint8, tl.int1: torch.bool,
}


def _to_torch(data: np.ndarray, dtype) -> torch.Tensor:
    array = np.ascontiguousarray(data)
    if dtype == tl.bfloat16:  # stored as uint16
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def patch_interpreter() -> None:
    if getattr(interpreter, '_evograd_patched', False):
        return
    interpreter._evograd_patched = True

    patch_tensor = interpreter._patch_lang_tensor

    def patched_tensor(tensor, scope):
        patch_tensor(tensor, scope)
        scope.set_attr(tensor, '__index__', lambda self: int(self.handle.data.reshape(-1)[0]))

    interpreter._patch_lang_tensor = patched_tensor

    builder = interpreter.InterpreterBuilder
    cast = builder.cast_impl

    def cast_impl(self, src, dst_type):
        source, target = src.dtype.scalar, dst_type.scalar
        if tl.bfloat16 in (source, target) and source in _TORCH_DTYPES and target in _TORCH_DTYPES:
            converted = _to_torch(src.data, source).to(_TORCH_DTYPES[target])
            return interpreter.TensorHandle(_to_numpy(converted).reshape(np.shape(src.data)), target)
        return cast(self, src, dst_type)

    # create_fp_trunc, create_fp_ext, create_si_to_fp, ... all call cast_impl.
    builder.cast_impl = cast_impl
    _patch_bfloat16_math(builder)

    implicit = interpreter._implicit_cvt

    def implicit_cvt(arg):
        if isinstance(arg, bool):
            return tl.tensor(interpreter.TensorHandle(np.array([arg], dtype=np.bool_), tl.int1), tl.int1)
        return implicit(arg)

    interpreter._implicit_cvt = implicit_cvt


def _is_bf16(handle) -> bool:
    return getattr(handle.dtype, 'scalar', handle.dtype) == tl.bfloat16


def _values(handle) -> np.ndarray:
    """A handle's data as numbers: bfloat16 bit patterns widened to float32."""
    if _is_bf16(handle):
        return (np.asarray(handle.data).astype(np.uint32) << 16).view(np.float32)
    return handle.data


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    rounded = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(torch.bfloat16)
    return _to_numpy(rounded).reshape(np.shape(values))


def _patch_bfloat16_math(builder) -> None:
    binary, unary, ternary = builder.binary_op, builder.unary_op, builder.ternary_op
    fma, dot = builder.create_fma, builder.create_dot

    def binary_op(self, lhs, rhs, op):
        if not (_is_bf16(lhs) or _is_bf16(rhs)):
            return binary(self, lhs, rhs, op)
        output = op(_values(lhs), _values(rhs))
        if output.dtype == np.bool_:  # a comparison
            return interpreter.TensorHandle(output, tl.int1)
        return interpreter.TensorHandle(_bf16_bits(output), tl.bfloat16)

    def unary_op(self, arg, op):
        if not _is_bf16(arg):
            return unary(self, arg, op)
        return interpreter.TensorHandle(_bf16_bits(op(_values(arg))), tl.bfloat16)

    def ternary_op(self, lhs, rhs, other, op):
        if op is np.where or not _is_bf16(other):  # a select moves bits
            return ternary(self, lhs, rhs, other, op)
        return interpreter.TensorHandle(_bf16_bits(op(_values(lhs), _values(rhs), _values(other))), tl.bfloat16)

    def create_fma(self, x, y, z):
        if not any(map(_is_bf16, (x, y, z))):
            return fma(self, x, y, z)
        return interpreter.TensorHandle(_bf16_bits(_values(x) * _values(y) + _values(z)), tl.bfloat16)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if not (_is_bf16(a) or _is_bf16(b)):
            return dot(self, a, b, d, input_precision, max_num_imprecise_acc)
        product = np.matmul(_values(a), _values(b), dtype=np.float32)
        if _is_bf16(d):
            return interpreter.TensorHandle(_bf16_bits(product + _values(d)), tl.bfloat16)
        return interpreter.TensorHandle((product + d.data).astype(d.data.dtype), d.dtype.scalar)

    builder.binary_op, builder.unary_op, builder.ternary_op = binary_op, unary_op, ternary_op
    # A bfloat16 constant (a Python float against a bfloat16 tensor).
    builder.get_bf16 = lambda self, value: interpreter.TensorHandle(
        _bf16_bits(np.array([value], dtype=np.float32)), tl.bfloat16)
    builder.create_fma, builder.create_dot = create_fma, create_dot

    reduce_ops, scan_ops = interpreter.ReduceOps, interpreter.ScanOps
    total, extreme, running = reduce_ops.sum, reduce_ops.min_max, scan_ops.cumsum

    def as_bf16(self, data):
        return self.to_tensor(_bf16_bits(data), tl.bfloat16)

    def sum_(self, input):
        if not _is_bf16(input.handle):
            return total(self, input)
        return as_bf16(self, np.sum(_values(input.handle), axis=self.axis, keepdims=self.keep_dims))

    def min_max(self, input, val_reduce_op, idx_reduce_op=None):
        values = input[0] if isinstance(input, tuple) else input  # (value, index) pairs
        if not _is_bf16(values.handle):
            return extreme(self, input, val_reduce_op, idx_reduce_op)
        data = _values(values.handle)
        value = index = None
        if val_reduce_op:
            value = as_bf16(self, val_reduce_op(data, axis=self.axis, keepdims=self.keep_dims))
        if idx_reduce_op:
            index = self.to_tensor(idx_reduce_op(data, axis=self.axis, keepdims=self.keep_dims), tl.int32)
        return (value, index) if value is not None and index is not None else (
            value if value is not None else index)

    def cumsum(self, input):
        if not _is_bf16(input.handle):
            return running(self, input)
        return [as_bf16(self, np.cumsum(_values(input.handle), axis=self.axis))]

    reduce_ops.sum, reduce_ops.min_max, scan_ops.cumsum = sum_, min_max, cumsum


def _elementwise(function):
    def call(*args, **kwargs):
        args = [arg.value if isinstance(arg, tl.constexpr) else arg for arg in args]
        first = next(arg for arg in args if isinstance(arg, tl.tensor))
        dtype = first.dtype.scalar
        operands = [_to_torch(arg.handle.data, arg.dtype.scalar) if isinstance(arg, tl.tensor) else arg
                    for arg in args]
        result = function(*operands).to(_TORCH_DTYPES[dtype])
        return tl.tensor(interpreter.TensorHandle(_to_numpy(result).reshape(np.shape(first.handle.data)),
                                                  dtype), first.type)
    return call


#: libdevice for the interpreter: the functions the handwritten kernels call.
libdevice = SimpleNamespace(**{name: _elementwise(function) for name, function in {
    'acos': torch.acos, 'acosh': torch.acosh, 'asin': torch.asin, 'asinh': torch.asinh,
    'atan': torch.atan, 'atan2': torch.atan2, 'atanh': torch.atanh, 'ceil': torch.ceil,
    'erf': torch.erf, 'exp': torch.exp, 'floor': torch.floor, 'log': torch.log,
    'log1p': torch.log1p, 'expm1': torch.expm1, 'pow': torch.pow, 'rint': torch.round,
    'sqrt': torch.sqrt, 'rsqrt': torch.rsqrt, 'tan': torch.tan, 'tanh': torch.tanh,
    'trunc': torch.trunc, 'isnan': torch.isnan, 'isinf': torch.isinf,
}.items()})


patch_interpreter()
