"""What one real MegaFold training step records, and how.

The recorder changes no arithmetic. Boundaries nest (a transition holds a
LayerNorm+Linear, which holds a LayerNorm and a GEMM), so nothing can be replaced
while recording: a kernel at the outer boundary would hide every call inside it.
Instead, on the plain eager tree:

* a forward hook at every site's module (``sites.BOUNDARIES``) records the
  arguments its deployment adapter would hand a kernel, from the same code, so a
  recorded call is exactly what a deployed kernel receives. A keep mask is read
  from a hook on the module's own ``nn.Dropout``: nothing here draws;
* a ``TorchFunctionMode`` sees every torch call at the Python level, above
  autocast. It files each primitive under its Level-1 task, with the casts
  autocast is about to make, and counts every other call in ``census``.

So the shared capture loop's check that the instrumented step equals the plain
one proves the recording harmless and nothing more. ``fidelity`` proves each
reference, call by call, and ``site_check`` proves the deployment adapters
(``docs/AF3.md`` section 7).

Only training calls count: a call under ``no_grad`` or inside a backward pass (an
activation-checkpoint recompute) is neither counted nor recorded. A kept call is
copied to host memory as it is recorded, layouts and the storage it shares with
the call's other arguments kept, so the step's GPU memory is the model's own.

Which calls the corpus keeps (plan B): per case and data order, the first,
middle and last of the case's calls in step order, or only the first when one
call's arguments and cotangent exceed :data:`RECORD_ONE_ABOVE` bytes. Every
training call is still counted against the snapshot's calls per step.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterator, Sequence

import torch

from torch.overrides import TorchFunctionMode

from evograd.benchmark.topdown.common.harvest_roles import role_of
from evograd.benchmark.topdown.megafold.harvest.harvest import UNBOUND

from ..base import Case, case_for, invocation

RECORD_CALLS = 3
RECORD_ONE_ABOVE = 2 << 30

TRIANGLE_MULTIPLICATION = ("the module's input, float32 under autocast; weights in the GEMM dtype; "
                           "the keep mask MegaFold drew")

TRIANGLE_ATTENTION = ("PreLayerNorm's input, float32 under autocast; weights in the GEMM dtype; "
                      "the keep mask MegaFold drew")

#: How each task's arguments are made from MegaFold's tensors.
INPUT_ADAPTERS = {
    "layernorm": "leading axes flattened; float32 under autocast",
    "layernorm_no_affine": "leading axes flattened; float32 under autocast",
    "layernorm_no_bias": "leading axes flattened; float32 under autocast",
    "linear": "leading axes flattened; x, weight and bias in the GEMM dtype",
    "linear_no_bias": "leading axes flattened; x and weight in the GEMM dtype",
    "glu": "leading axes flattened",
    "sigmoid": "leading axes flattened",
    "relu": "leading axes flattened",
    "softmax": "leading axes flattened; float32 under autocast",
    "megafold_layernorm_linear": "leading axes flattened; under autocast float32 rows and the "
                                 "weight in the GEMM dtype",
    "swiglu": "SwiGLU halves as [rows, cols] views of the projection",
    "megafold_evoattention": "Attend's [b, h, n, d] viewed as [B, S, N, H, D]; first row of the bias",
    "megafold_triangle_multiplication_outgoing": TRIANGLE_MULTIPLICATION,
    "megafold_triangle_multiplication_incoming": TRIANGLE_MULTIPLICATION,
    "megafold_triangle_attention_starting": TRIANGLE_ATTENTION,
    "megafold_triangle_attention_ending": TRIANGLE_ATTENTION,
    "megafold_transition": "leading axes flattened, float32 under autocast; weights in the GEMM dtype",
    "megafold_outer_product_mean": "the module's MSA input, float32 under autocast; weights and the "
                                   "MSA mask in the GEMM dtype; the float32 count MSAModule summed",
    "megafold_msa_pair_weighted_averaging": "the module's MSA and pair inputs, float32 under autocast; "
                                            "weights in the GEMM dtype; the keep mask MegaFold drew",
}


def module_role(path: str) -> str:
    """A call's role: the last two names of its module path, without indices.

    Equal shapes with different roles stay separate cases, and the same module in
    two stacks is one role: ``attn.to_q`` wherever it runs.
    """
    return ".".join([part for part in path.split(".") if not part.isdigit()][-2:])


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


def _shape_tokens(shape: str) -> list[str]:
    return [token.strip() for token in shape.strip()[1:-1].split(",") if token.strip()]


#: Dims no argument carries, from those that are: GLU's output is half its input.
DERIVED_DIMS = {"glu": lambda dims: {"cols": dims["width"] // 2}}


def _dims(op, values: dict) -> tuple[dict[str, int], torch.dtype]:
    """A call's dims and case dtype, read off its arguments and the task's declared shapes.

    The case dtype is that of the first tensor argument whose dtype the task leaves
    to the case: for a mixed-precision contract, the dtype its GEMMs run in.
    """
    dims: dict[str, int] = {}
    dtype = None
    for arg in op.args:
        value = values.get(arg.name)
        if not torch.is_tensor(value):
            continue
        for token, size in zip(_shape_tokens(arg.shape), value.shape):
            if token in op.dims:
                dims[token] = size
        if dtype is None and getattr(arg, "dtype", None) is None:
            dtype = value.dtype
    if op.name in DERIVED_DIMS:
        dims.update(DERIVED_DIMS[op.name](dims))
    return dims, dtype


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


def _signature(name, args, kwargs) -> tuple[str, str]:
    """An unbound call's configuration: argument shapes and dtypes, a static string, autocast."""
    tensors = [a for a in (*args, *kwargs.values()) if torch.is_tensor(a)]
    static = next((a for a in (*args, *kwargs.values()) if isinstance(a, str)), None)
    autocast = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
    return name, json.dumps({"shapes": [list(t.shape) for t in tensors],
                             "dtypes": [str(t.dtype).removeprefix("torch.") for t in tensors],
                             "static": static, "autocast": str(autocast)})


def training_call() -> bool:
    """Neither under ``no_grad`` nor inside a backward pass (a checkpoint recompute)."""
    return torch.is_grad_enabled() and torch._C._current_graph_task_id() == -1


def _rows(t):
    return t.reshape(-1, t.shape[-1])


def _layer_norm(args, kwargs):
    """``F.layer_norm`` as autocast runs it: float32 rows, the task by the parameters it has."""
    from evograd.evaluation.tier3.workloads.megafold.sites import norm_input

    x, shape = args[0], args[1] if len(args) > 1 else kwargs["normalized_shape"]
    weight = args[2] if len(args) > 2 else kwargs.get("weight")
    bias = args[3] if len(args) > 3 else kwargs.get("bias")
    eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
    if len(tuple(shape)) != 1:
        return None
    rows = norm_input(_rows(x))
    if weight is None and bias is None:
        return "layernorm_no_affine", {"x": rows, "eps": eps}
    if bias is None:
        return "layernorm_no_bias", {"x": rows, "weight": weight, "eps": eps}
    return "layernorm", {"x": rows, "weight": weight, "bias": bias, "eps": eps}


def _linear(args, kwargs):
    """``F.linear`` as autocast runs it: rows and weights in the GEMM's dtype.

    A biased call on a rank-4 or non-contiguous input takes ``matmul`` then ``add``,
    where the task's rows take the fused ``addmm``, so its rounding is not the
    task's reference. Those calls stay in the inventory and out of the cases.
    """
    from evograd.evaluation.tier3.workloads.megafold.sites import gemm_dtype

    x, weight = args[0], args[1] if len(args) > 1 else kwargs["weight"]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if weight.ndim != 2 or (bias is not None and not (x.ndim <= 3 and x.is_contiguous())):
        return None
    dtype = gemm_dtype(x)
    values = {"x": _rows(x).to(dtype), "weight": weight.to(dtype)}
    if bias is None:
        return "linear_no_bias", values
    return "linear", {**values, "bias": bias.to(dtype)}


def _elementwise(task):
    def bind(args, _kwargs):
        x = args[0]
        return None if x.ndim == 0 else (task, {"x": _rows(x)})

    return bind


def _glu(args, kwargs):
    x, dim = args[0], args[1] if len(args) > 1 else kwargs.get("dim", -1)
    return ("glu", {"x": _rows(x)}) if dim in (-1, x.ndim - 1) else None


def _softmax(args, kwargs):
    """Autocast runs softmax in float32, over any dtype it is given."""
    from evograd.evaluation.tier3.workloads.megafold.sites import norm_input

    x, dim = args[0], args[1] if len(args) > 1 else kwargs.get("dim")
    if dim not in (-1, x.ndim - 1) or kwargs.get("dtype") is not None:
        return None
    return "softmax", {"x": norm_input(_rows(x))}


#: Primitive name -> ``(args, kwargs) -> (task, values)``, or None where a call is
#: not what its task describes, which leaves it in the inventory. Anything else is
#: counted in ``census`` only.
PRIMITIVES = {"layer_norm": _layer_norm, "linear": _linear, "glu": _glu, "softmax": _softmax,
              "sigmoid": _elementwise("sigmoid"), "relu": _elementwise("relu")}


class _Primitives(TorchFunctionMode):
    """Every torch call of the step, passed straight through.

    A mode is off while its own handler runs, so ``F.layer_norm`` is seen once
    and not again as the ``layer_norm`` it calls.
    """

    def __init__(self, capture: "MegaFoldCapture"):
        super().__init__()
        self.capture = capture

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.capture._primitive(func, args, kwargs, out)
        return out


class MegaFoldCapture:
    """Pass-through recorders on an eager MegaFold tree.

    ``cases`` given (capture): every training call is matched to its case and
    counted, and the declared subset is kept with its tensors. ``cases`` None
    (harvest): every training call's metadata is kept and no tensor is.
    """

    def __init__(self, model, cases: Sequence[Case] | None = None):
        self.model = model
        self.cases = cases
        self.calls: list[dict[str, Any]] = []
        self.kept: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        #: Primitive name -> training calls in the step, bound or not.
        self.census: dict[str, int] = {}
        #: Each unbound operator's configurations -> calls, for the inventory.
        self._unbound: dict[tuple, int] = {}
        self._picks = {c.case_id: set(recorded_indices(c)) for c in cases or ()}
        self._stack: list[str] = []
        self._handles: list = []
        self._mode: _Primitives | None = None
        #: The modules of a primitive's own site (its calls file under that site).
        self._site_of: dict[str, str] = {}
        self._quiet = 0  # >0 while the recorder's own tensor work runs

    def install(self) -> "MegaFoldCapture":
        from evograd.evaluation.tier3.workloads.megafold import sites, workload as w

        if self.cases is not None:
            self._require_current_snapshot()
        paths = {id(m): path for path, m in self.model.named_modules()}

        def enter(_module, _args, path):
            self._stack.append(path)

        def leave(_module, _args, _output):
            self._stack.pop()  # a forward hook that returns a value replaces the output

        for module in self.model.modules():
            self._handles.append(module.register_forward_pre_hook(
                lambda m, a, path=paths[id(module)]: enter(m, a, path)))
            self._handles.append(module.register_forward_hook(leave))
        for spec in w._module_patches(self.model):
            boundary = sites.BOUNDARIES[spec.site]
            for module in list(self.model.modules()):
                if spec.match(module):
                    target = boundary.module(module)
                    if boundary.recorded:
                        self._watch(boundary, target, paths[id(target)])
                    else:
                        self._site_of[paths[id(target)]] = spec.site
        self._mode = _Primitives(self)
        self._mode.__enter__()
        return self

    def remove(self) -> None:
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _require_current_snapshot() -> None:
        from evograd.benchmark.topdown import load_snapshot
        from evograd.benchmark.topdown.megafold.harvest.harvest import SCHEMA

        found = load_snapshot("megafold")["schema_version"]
        if found != SCHEMA:
            raise RuntimeError(f"the tracked snapshot is {found}, recorded before every primitive "
                               f"was harvested; regenerate it ({SCHEMA}) before capturing")

    def _watch(self, boundary, module, path: str) -> None:
        from evograd.benchmark import get_task

        op = get_task(boundary.task)
        tapped: dict[str, Any] = {}
        if boundary.tap is not None:
            self._handles.append(boundary.tap(module).register_forward_hook(
                lambda _m, _a, out: tapped.__setitem__("value", out)))

        def post(mod, args, kwargs, out):
            if not training_call():
                return
            self._quiet += 1
            try:
                with torch.no_grad():
                    values = boundary.arguments(mod, args, kwargs, tapped.get("value"))
                    self._observe(op, boundary.site, path, values, out,
                                  lambda t: boundary.output(t, args, kwargs))
            finally:
                self._quiet -= 1

        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _primitive(self, func, args, kwargs, out) -> None:
        from evograd.benchmark import get_task
        from evograd.evaluation.tier3.workloads.megafold.workload import NOT_SITES

        name = getattr(func, "__name__", None) or str(func)
        if self._quiet or name in ("__get__", "__set__") or not training_call():
            return
        self.census[name] = self.census.get(name, 0) + 1
        if name in UNBOUND:
            key = _signature(name, args, kwargs)
            self._unbound[key] = self._unbound.get(key, 0) + 1
        bind = PRIMITIVES.get(name)
        path = self._stack[-1] if self._stack else ""
        if bind is None or path.split(".", 1)[0] in NOT_SITES:
            return
        self._quiet += 1
        try:
            with torch.no_grad():
                bound = bind(args, kwargs)
                if bound is None:
                    key = _signature(name, args, kwargs)
                    self._unbound[key] = self._unbound.get(key, 0) + 1
                    return
                task, values = bound
                self._observe(get_task(task), self._site_of.get(path, "primitive"), path, values, out,
                              _rows)
        finally:
            self._quiet -= 1

    def unbound_configurations(self) -> list[dict[str, Any]]:
        """Each unbound operator's observed configurations, with calls per step."""
        return [{**json.loads(key[1]), "name": key[0], "calls": calls}
                for key, calls in sorted(self._unbound.items())]

    def _observe(self, op, site, path, values, out, layout) -> None:
        dims, dtype = _dims(op, values)
        tensors = {n: v for n, v in values.items() if torch.is_tensor(v)}
        free = {"role": module_role(path)} if op.level == 1 else {}
        self.calls.append({
            "level": op.level, "task": op.name, "site": site, "path": path, "dims": dims,
            "dtype": str(dtype).removeprefix("torch."),
            "inputs": [_meta(v) for v in tensors.values()],
            "strides": {n: list(v.stride()) for n, v in tensors.items()},
            "input_adapter": INPUT_ADAPTERS[op.name],
            "provenance": {"component": op.name, "free": free, "source": "execution", "note": site}})
        if self.cases is None:
            return
        case = case_for(self.cases, op.name, values, path=path)
        index = self.counts.get(case.case_id, 0)
        self.counts[case.case_id] = index + 1
        if index not in self._picks[case.case_id]:
            return
        record = {"case": case.case_id, "task": op.name, "level": op.level, "site": site, "path": path,
                  "index": index, "args": _host({n: (v.detach() if torch.is_tensor(v) else v)
                                                 for n, v in values.items()}),
                  "out": {op.output_names[0]: _summary(layout(out))}, "grad_name": op.upstream_grad_names[0]}
        if out.requires_grad:
            out.register_hook(lambda g, r=record: r.__setitem__(
                "cotangent", _host({"g": layout(g.detach())})["g"]))
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
