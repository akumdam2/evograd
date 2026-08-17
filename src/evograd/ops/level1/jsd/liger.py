"""Reviewed Liger Jensen-Shannon-divergence autograd-pair adapter."""

import importlib
import inspect

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def _load():
    last_error = None
    for name in ("liger_kernel.ops.jsd", "liger_kernel.ops.jsd_loss"):
        try:
            module = importlib.import_module(name)
            return module.jsd_forward, module.jsd_backward
        except Exception as exc:
            last_error = exc
    raise ImportError("Could not import Liger JSD raw ops") from last_error


def _backward(fn, precomputed, dout):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "in_place" in params:
        return fn(precomputed, dout, in_place=False)
    if "inplace" in params:
        return fn(precomputed, dout, inplace=False)
    try:
        return fn(precomputed, dout, False)
    except TypeError:
        return fn(precomputed, dout)


def make_liger_jsd_autograd_pair_fns():
    forward, backward = _load()

    def forward_with_saved(log_q, target):
        dtype_marker = log_q.new_empty(())
        compute_q = log_q.float() if log_q.dtype in (torch.float16, torch.bfloat16) else log_q
        compute_target = (
            target.float() if target.dtype in (torch.float16, torch.bfloat16) else target
        )
        result = forward(
            compute_q.contiguous(),
            compute_target.contiguous(),
            None,
            0.5,
            -100,
            False,
        )
        return result[0].float(), (result[1].contiguous(), dtype_marker)

    def backward_from_saved(dout, saved):
        precomputed, dtype_marker = saved
        if dout.dtype != precomputed.dtype:
            dout = dout.to(precomputed.dtype)
        output = _backward(backward, precomputed, dout)
        if isinstance(output, tuple):
            output = output[0]
        return output.to(dtype_marker.dtype)

    return forward_with_saved, backward_from_saved
