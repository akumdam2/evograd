"""What one real MegaFold training step records, and how.

The recorder installs the deployment's own site adapters (the tier-3
workload's ``_module_patches``) on the eager tree, each calling a kernel that
computes the site's model reference (:mod:`.reference`). A recorded call is
therefore exactly what a deployed kernel receives -- the task arguments after
the adapter's flattening, views and autocast casts -- and the shared capture
loop's check that the instrumented step equals the plain one bit for bit is
also the check that the adapters are MegaFold's computation.

Only training calls count: a call under ``no_grad`` or inside a backward pass
(an activation-checkpoint recompute) is neither counted nor recorded. A kept
call is copied to host memory as it is recorded, layouts and the storage it
shares with the call's other arguments kept, so the step's GPU memory is the
model's own (a step's kept calls are ~37 GiB at 384 residues).

Which calls the corpus keeps (plan B): per case and data order, the first,
middle and last of the case's calls in step order, or only the first when one
call's arguments and cotangent exceed :data:`RECORD_ONE_ABOVE` bytes. A full
step at 384 residues is ~275 GiB of payloads; this keeps ~28 GiB. Every training
call is still counted against the snapshot's calls per step.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Sequence

import torch

from evograd.benchmark.topdown.common.harvest_roles import role_of
from evograd.benchmark.topdown.megafold import MEGAFOLD_SITE_OPS

from ..base import Case, case_for, invocation

RECORD_CALLS = 3
RECORD_ONE_ABOVE = 2 << 30

#: How each site's adapter turns MegaFold's tensors into its task's arguments.
INPUT_ADAPTERS = {
    "layer_norm": "leading axes flattened; float32 under autocast",
    "layernorm_linear": "leading axes flattened; under autocast float32 rows and the weight in the GEMM dtype",
    "transition": "SwiGLU halves as [rows, cols] views of the projection",
    "pair_bias_attention": "Attend's [b, h, n, d] viewed as [B, S, N, H, D]; first row of the bias",
    "triangle_attention": "Attend's [b, h, n, d] viewed as [B, S, N, H, D]; first row of the bias",
}


def _meta(t) -> dict[str, Any]:
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
            "requires_grad": t.requires_grad}


def _host(values: dict[str, Any]) -> dict[str, Any]:
    """One call's tensors on the host: shapes, strides and dtypes kept, and arguments that
    share a storage still sharing one (offsets kept relative to the shared span). A tensor
    that is a small view of a much larger storage gets its own minimal storage instead,
    as the corpus would store it compacted."""
    out = dict(values)
    groups: dict[int, list[str]] = {}
    for name, v in values.items():
        if torch.is_tensor(v):
            groups.setdefault(v.untyped_storage().data_ptr(), []).append(name)
    for names in groups.values():
        views = [values[n] for n in names]
        spans = [(t.storage_offset() * t.element_size(),
                  (t.storage_offset() + sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1)
                  * t.element_size()) for t in views]
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        storage = views[0].untyped_storage()
        if storage.nbytes() > 4 * (hi - lo) + (64 << 20):
            for name, t in zip(names, views):
                out[name] = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cpu").copy_(t)
            continue
        region = torch.empty(0, dtype=torch.uint8, device=views[0].device).set_(storage, lo, (hi - lo,), (1,))
        host = torch.empty(hi - lo, dtype=torch.uint8).copy_(region).untyped_storage()
        for name, t, (start, _) in zip(names, views, spans):
            out[name] = torch.empty(0, dtype=t.dtype).set_(host, (start - lo) // t.element_size(),
                                                           t.shape, t.stride())
    return out


def _summary(t) -> dict[str, Any]:
    """What a payload keeps of a live output: its digest and layout (``base.record_step``)."""
    from evograd.verification.common.tensors import tensor_digest

    return {"digest": tensor_digest(t), "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype)}


def _dims(task: str, v: dict) -> tuple[dict[str, int], torch.dtype]:
    """A call's dims and case dtype, read off its arguments."""
    if task == "layernorm":
        return dict(rows=v["x"].shape[0], hidden=v["x"].shape[1]), v["x"].dtype
    if task == "swiglu":
        return dict(rows=v["a"].shape[0], cols=v["a"].shape[1]), v["a"].dtype
    if task == "megafold_layernorm_linear":
        (m, k), n = v["x"].shape, v["linear_weight"].shape[1]
        return dict(M=m, K=k, N=n), v["linear_weight"].dtype
    b, s, n, h, d = v["q"].shape
    return dict(B=b, S=s, H=h, N=n, D=d), v["q"].dtype


def call_bytes(case: Case) -> int:
    """Bytes one call of ``case`` puts in the corpus: its tensor arguments and output cotangents."""
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import bind_shape
    from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

    op = get_task(case.task)
    total = 0
    for arg in (*op.args, *op.outputs):
        if getattr(arg, "shape", None) in (None, "unspecified"):
            continue
        dtype = resolve_dtype(arg_dtype_name(arg, case.workload))
        total += math.prod(bind_shape(arg.shape, case.workload.dims)) * dtype.itemsize
    return total


def recorded_indices(case: Case) -> tuple[int, ...]:
    """Which of a case's calls per step the corpus keeps (0-based, in step order)."""
    n = case.frequency
    if call_bytes(case) > RECORD_ONE_ABOVE:
        return (0,)
    return tuple(sorted({0, (n - 1) // 2, n - 1}))[:RECORD_CALLS]


def recorded_calls(case: Case) -> int:
    return len(recorded_indices(case))


class MegaFoldCapture:
    """The site adapters with recording reference kernels, on an eager MegaFold tree.

    ``cases`` given (capture): every training call is matched to its case and
    counted, and the declared subset is kept with its tensors. ``cases`` None
    (harvest): every training call's metadata is kept and no tensor is.

    ``remove`` takes the hooks off but leaves the adapters in place: the capture
    loop discards the model after recording, and the adapters compute the
    model's own result.
    """

    def __init__(self, model, cases: Sequence[Case] | None = None):
        self.model = model
        self.cases = cases
        self.calls: list[dict[str, Any]] = []
        self.kept: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self._picks = {c.case_id: set(recorded_indices(c)) for c in cases or ()}
        self._stack: list[str] = []
        self._handles: list = []

    def install(self) -> "MegaFoldCapture":
        from evograd.evaluation.tier3.patch import KernelSet, patch_modules
        from evograd.evaluation.tier3.workloads.megafold import workload as w

        kernels = KernelSet(registry=w.MEGAFOLD_SITES, patched=tuple(MEGAFOLD_SITE_OPS),
                            overrides={site: self._kernel(site) for site in MEGAFOLD_SITE_OPS})
        patch_modules(self.model, w._module_patches(self.model), kernels)
        adapters = (w._MegaFoldPatchedLayerNorm, w._PatchedLayernormLinear,
                    w._PatchedSwiGLUTransition, w._PatchedAttend)
        def enter(_module, _args, path):
            self._stack.append(path)

        def leave(_module, _args, _output):
            self._stack.pop()  # a forward hook that returns a value replaces the output

        for path, module in self.model.named_modules():
            if isinstance(module, adapters):
                self._handles.append(module.register_forward_pre_hook(
                    lambda m, a, path=path: enter(m, a, path)))
                self._handles.append(module.register_forward_hook(leave))
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _kernel(self, site: str):
        from evograd.benchmark import get_task
        from evograd.opdecl.oracle import resolve_runtime_forward

        from .reference import reference_op

        task = MEGAFOLD_SITE_OPS[site]
        op = get_task(task)
        names = [a.name for a in op.args]
        forward = resolve_runtime_forward(reference_op(task))

        def kernel(*args):
            out = forward(*args)
            if torch.is_grad_enabled() and torch._C._current_graph_task_id() == -1:
                self._observe(site, task, op, dict(zip(names, args)), out)
            return out

        return kernel

    def _observe(self, site, task, op, values, out) -> None:
        path = self._stack[-1]
        dims, dtype = _dims(task, values)
        tensors = {n: v for n, v in values.items() if torch.is_tensor(v)}
        self.calls.append({
            "level": op.level, "task": task, "site": site, "path": path, "dims": dims,
            "dtype": str(dtype).removeprefix("torch."),
            "inputs": [_meta(v) for v in tensors.values()],
            "strides": {n: list(v.stride()) for n, v in tensors.items()},
            "input_adapter": INPUT_ADAPTERS[site],
            "provenance": {"component": task, "free": {}, "source": "execution", "note": site}})
        if self.cases is None:
            return
        case = case_for(self.cases, task, values, path=path)
        index = self.counts.get(case.case_id, 0)
        self.counts[case.case_id] = index + 1
        if index not in self._picks[case.case_id]:
            return
        record = {"case": case.case_id, "task": task, "level": op.level, "site": site, "path": path,
                  "index": index, "args": _host({n: (v.detach() if torch.is_tensor(v) else v)
                                                 for n, v in values.items()}),
                  "out": {op.output_names[0]: _summary(out)}, "grad_name": op.upstream_grad_names[0]}
        if out.requires_grad:
            out.register_hook(lambda g, r=record: r.__setitem__("cotangent", _host({"g": g.detach()})["g"]))
        self.kept.append(record)


def invocations(capture: MegaFoldCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """The kept calls of one step, in step order, as invocations of their cases."""
    expected = {c.case_id: c.frequency for c in cases}
    if capture.counts != expected:
        wrong = {k: (capture.counts.get(k, 0), n) for k, n in expected.items() if capture.counts.get(k) != n}
        raise RuntimeError(f"training calls per case differ from the snapshot (seen, expected): {wrong}")
    for r in capture.kept:
        cotangent, notes = r.get("cotangent"), {}
        if cotangent is None:  # no gradient reached this output in the model
            out = next(iter(r["out"].values()))
            cotangent = torch.zeros(out["shape"], dtype=getattr(torch, out["dtype"].removeprefix("torch.")))
            notes["absent_cotangents"] = [r["grad_name"]]
        call = {"path": f"{r['path']}#{r['index']}", "layer": None, "role": role_of(r["path"]),
                "kind": f"site:{r['site']}", "call_of_case": r["index"]}
        yield invocation(cases, r["task"], r["level"], call, r["args"], {r["grad_name"]: cotangent},
                         r["out"], notes)
