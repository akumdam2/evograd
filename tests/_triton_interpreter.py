"""Make Triton's CPU interpreter (TRITON_INTERPRET=1) run the handwritten kernels
as the GPU does, for bit-for-bit checks on a login node. Import before any kernel
launches; the patches are idempotent.

* A one-element interpreter tensor converts to a Python int, which a
  data-dependent loop bound (``range(tensor)``) needs under NumPy 2.
* Conversions to and from bfloat16 round as compiled Triton does (to nearest,
  ties to even, as LLVM's fptrunc and sitofp): the interpreter truncates a
  float32 and reinterprets the bits of every other source dtype.

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
