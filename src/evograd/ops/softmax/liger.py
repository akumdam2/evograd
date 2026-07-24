"""Fairness-reviewed Liger softmax adapter, including its wide-row fix."""

from __future__ import annotations

import inspect

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass

_TRITON = None
_TL = None
_FIXED_FORWARD = None
_FIXED_BACKWARD = None


def _fallback_num_warps(block_size):
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def _fixed_forward_kernel():
    global _TRITON, _TL, _FIXED_FORWARD
    if _FIXED_FORWARD is not None:
        return _FIXED_FORWARD
    import triton
    import triton.language as tl

    _TRITON, _TL = triton, tl

    @triton.jit
    def kernel(y_ptr, y_stride, x_ptr, x_stride, cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        maximum = -float("inf")
        denominator = 0.0
        for start in range(0, cols, BLOCK_SIZE):
            indices = start + offsets
            mask = indices < cols
            x = tl.load(
                x_ptr + row * x_stride + indices, mask=mask, other=-float("inf")
            ).to(tl.float32)
            block_max = tl.max(x, axis=0)
            next_max = tl.maximum(maximum, block_max)
            denominator = denominator * tl.exp(maximum - next_max) + tl.sum(
                tl.exp(x - next_max), axis=0
            )
            maximum = next_max
        for start in range(0, cols, BLOCK_SIZE):
            indices = start + offsets
            mask = indices < cols
            x = tl.load(
                x_ptr + row * x_stride + indices, mask=mask, other=-float("inf")
            ).to(tl.float32)
            tl.store(
                y_ptr + row * y_stride + indices,
                tl.exp(x - maximum) / denominator,
                mask=mask,
            )

    _FIXED_FORWARD = kernel
    return kernel


def _fixed_backward_kernel():
    global _TRITON, _TL, _FIXED_BACKWARD
    if _FIXED_BACKWARD is not None:
        return _FIXED_BACKWARD
    import triton
    import triton.language as tl

    _TRITON, _TL = triton, tl

    @triton.jit
    def kernel(
        dx_ptr,
        dx_stride,
        dy_ptr,
        dy_stride,
        y_ptr,
        y_stride,
        cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        dot = 0.0
        for start in range(0, cols, BLOCK_SIZE):
            indices = start + offsets
            mask = indices < cols
            y = tl.load(
                y_ptr + row * y_stride + indices, mask=mask, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                dy_ptr + row * dy_stride + indices, mask=mask, other=0.0
            ).to(tl.float32)
            dot += tl.sum(dy * y, axis=0)
        for start in range(0, cols, BLOCK_SIZE):
            indices = start + offsets
            mask = indices < cols
            y = tl.load(
                y_ptr + row * y_stride + indices, mask=mask, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                dy_ptr + row * dy_stride + indices, mask=mask, other=0.0
            ).to(tl.float32)
            tl.store(
                dx_ptr + row * dx_stride + indices, y * (dy - dot), mask=mask
            )

    _FIXED_BACKWARD = kernel
    return kernel


class _ForwardLauncher:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs = dict(kwargs)
            block = kwargs.pop(
                "BLOCK_SIZE", kwargs.pop("block_size", kwargs.pop("BLOCK", None))
            )
            args = list(args)
            if block is None and len(args) >= 6:
                block, args = args[5], args[:5]
            if block is None or len(args) < 5:
                raise TypeError("unexpected Liger multiblock forward arguments")
            return _fixed_forward_kernel()[grid](
                *args[:5], BLOCK_SIZE=int(block), **kwargs
            )

        return launch


class _BackwardLauncher:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs = dict(kwargs)
            block = kwargs.pop(
                "BLOCK_SIZE", kwargs.pop("block_size", kwargs.pop("BLOCK", None))
            )
            args = list(args)
            if block is None and len(args) in (6, 8):
                block, args = args[-1], args[:-1]
            if block is None:
                raise TypeError("missing BLOCK_SIZE for Liger multiblock backward")
            if len(args) >= 7:
                values = args[:7]
            elif len(args) >= 5:
                dy, dy_stride, y, y_stride, cols = args[:5]
                values = (dy, dy_stride, dy, dy_stride, y, y_stride, cols)
            else:
                raise TypeError("unexpected Liger multiblock backward arguments")
            return _fixed_backward_kernel()[grid](
                *values, BLOCK_SIZE=int(block), **kwargs
            )

        return launch


def _patch_multiblock(module, forward, backward):
    forward_launcher, backward_launcher = _ForwardLauncher(), _BackwardLauncher()
    forward_globals = getattr(forward, "__globals__", {})
    backward_globals = getattr(backward, "__globals__", {})
    if "_softmax_multi_block_forward_kernel" in forward_globals:
        forward_globals["_softmax_multi_block_forward_kernel"] = forward_launcher
    if hasattr(module, "_softmax_multi_block_forward_kernel"):
        module._softmax_multi_block_forward_kernel = forward_launcher
    if "_softmax_multi_block_backward_kernel" in backward_globals:
        backward_globals["_softmax_multi_block_backward_kernel"] = backward_launcher
    if hasattr(module, "_softmax_multi_block_backward_kernel"):
        module._softmax_multi_block_backward_kernel = backward_launcher


def make_liger_softmax_autograd_pair_fns():
    import liger_kernel.ops.softmax as module
    import liger_kernel.ops.utils as utils
    from liger_kernel.ops.softmax import _softmax_backward, _softmax_forward
    from liger_kernel.ops.utils import calculate_settings as default_settings

    _patch_multiblock(module, _softmax_forward, _softmax_backward)
    forward_globals = getattr(_softmax_forward, "__globals__", {})
    backward_globals = getattr(_softmax_backward, "__globals__", {})
    original_settings = forward_globals.get(
        "calculate_settings", getattr(module, "calculate_settings", default_settings)
    )
    max_fused = int(getattr(utils, "MAX_FUSED_SIZE", 65536))

    def derive_settings(cols):
        try:
            block, warps = original_settings(int(cols))
        except Exception:
            if cols <= max_fused:
                raise
            try:
                block, warps = original_settings(max_fused)
            except Exception:
                block, warps = max_fused, _fallback_num_warps(max_fused)
        return int(block), int(warps)

    def patched_settings(cols):
        return derive_settings(cols)

    def with_settings(fn):
        sentinel = object()
        old_forward = forward_globals.get("calculate_settings", sentinel)
        old_backward = backward_globals.get("calculate_settings", sentinel)
        old_module = getattr(module, "calculate_settings", sentinel)
        forward_globals["calculate_settings"] = patched_settings
        backward_globals["calculate_settings"] = patched_settings
        module.calculate_settings = patched_settings
        try:
            return fn()
        finally:
            if old_forward is sentinel:
                forward_globals.pop("calculate_settings", None)
            else:
                forward_globals["calculate_settings"] = old_forward
            if old_backward is sentinel:
                backward_globals.pop("calculate_settings", None)
            else:
                backward_globals["calculate_settings"] = old_backward
            if old_module is sentinel:
                delattr(module, "calculate_settings")
            else:
                module.calculate_settings = old_module

    try:
        parameters = [
            p
            for p in inspect.signature(_softmax_backward).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(_softmax_backward).parameters.values()
        )
    except Exception:
        parameters, var_kwargs = [], False

    def raw_backward(dout, output):
        cols = int(output.shape[-1])
        block, warps = derive_settings(cols)
        multiblock = cols > block

        def call():
            if not parameters or len(parameters) <= 2:
                return _softmax_backward(dout, output)
            kwargs, unknown = {}, False
            for parameter in parameters[2:]:
                name, lower = parameter.name, parameter.name.lower()
                if lower in ("block_size", "blocksize") or name == "BLOCK_SIZE":
                    kwargs[name] = block
                elif lower == "num_warps":
                    kwargs[name] = warps
                elif "multi" in lower and "block" in lower:
                    kwargs[name] = multiblock
                elif lower in ("n_cols", "ncols", "num_cols", "cols"):
                    kwargs[name] = cols
                elif lower in ("in_place", "inplace"):
                    kwargs[name] = False
                elif parameter.default is inspect.Parameter.empty and not var_kwargs:
                    unknown = True
            return (
                _softmax_backward(dout, output, **kwargs)
                if not unknown
                else _softmax_backward(dout, output)
            )

        result = with_settings(call)
        return result[0] if isinstance(result, tuple) else result

    def forward_with_saved(x):
        contiguous = x.contiguous()
        shape = contiguous.shape
        x_2d = contiguous.reshape(1, 1) if contiguous.ndim == 0 else contiguous.reshape(
            -1, int(shape[-1])
        )
        result = with_settings(lambda: _softmax_forward(x_2d))
        y_2d = result[0] if isinstance(result, tuple) else result
        output = y_2d.reshape(shape)
        return output, (output,)

    def backward_from_saved(dout, saved):
        (output,) = saved
        output, dout = output.contiguous().clone(), dout.contiguous().clone()
        shape = output.shape
        if output.ndim == 0:
            output_2d, dout_2d = output.reshape(1, 1), dout.reshape(1, 1)
        else:
            cols = int(shape[-1])
            output_2d, dout_2d = output.reshape(-1, cols), dout.reshape(-1, cols)
        return raw_backward(dout_2d, output_2d).reshape(shape)

    return forward_with_saved, backward_from_saved
