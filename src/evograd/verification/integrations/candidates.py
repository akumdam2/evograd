"""Control candidates for a model's cases: the native computation, or a known fault.

``native_pair(model, task)`` is a forward/backward pair whose forward is the
model reference and whose backward re-runs that reference under autograd: the
same kernels the unmodified model runs, so every gate must accept it.
``scale`` multiplies one gradient (``fault_on``, default the first) -- a 1%
error that Gate 1's dtype tolerance may miss and the calibrated gates must
catch. They are verification controls, not benchmark candidates.

    python -m evograd.verification.integrations.candidates --model llama_3_2_1b \\
        --task linear_no_bias --scale 1.01 --out linear_x1.01.py
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

TEMPLATE = '''"""Control candidate: {model} {task}, gradient {fault} scaled by {scale!r} (1.0: native)."""
from evograd.verification.integrations.candidates import native_pair

_PAIR = native_pair({model!r}, {task!r}, scale={scale!r}, fault_on={fault!r})
forward_with_saved = _PAIR.forward_with_saved
backward_from_saved = _PAIR.backward_from_saved
'''


def native_pair(model_name: str, task: str, *, scale: float = 1.0, fault_on: str | None = None):
    import torch

    from evograd.opdecl.oracle import resolve_runtime_forward

    from . import load

    op = load(model_name).reference_op(task)
    forward = resolve_runtime_forward(op)
    names = [a.name for a in op.args]
    active = [a.name for a in op.active_args()]
    grads = list(op.grad_names())
    fault = grads.index(fault_on) if fault_on is not None else 0

    def forward_with_saved(*args):
        with torch.no_grad():
            out = forward(*args)
        return out, tuple(args)

    def backward_from_saved(dout, saved, **_scalars):
        values = dict(zip(names, saved))
        leaves = {n: values[n].detach().requires_grad_(True) for n in active}
        with torch.enable_grad():
            out = forward(*[leaves.get(n, values[n]) for n in names])
        outs = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        douts = tuple(dout) if isinstance(dout, (tuple, list)) else (dout,)
        found = torch.autograd.grad(outs, [leaves[n] for n in active], douts, allow_unused=True)
        result = [torch.zeros_like(leaves[n]) if g is None else g for n, g in zip(active, found)]
        if scale != 1.0:
            result[fault] = result[fault] * scale
        return tuple(result) if len(result) > 1 else result[0]

    return types.SimpleNamespace(__name__=f"{model_name}:{task}:x{scale}",
                                 forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved)


def write(model_name: str, task: str, out: str | Path, *, scale: float = 1.0,
          fault_on: str | None = None) -> Path:
    from . import load

    op = load(model_name).reference_op(task)
    fault = fault_on or list(op.grad_names())[0]
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.format(model=model_name, task=task, scale=float(scale), fault=fault))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fault-on", default=None, help="gradient to scale (default: the first)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(write(args.model, args.task, args.out, scale=args.scale, fault_on=args.fault_on))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
