"""What a real training step of ConvNeXt V2 or Qwen2-VL records, and how.

Shared by the ``convnextv2_tiny`` and ``qwen2_vl_2b`` integrations. One native
step of the streamed real-data workload (pinned pretrained weights) runs per
data order with recorders attached; every Level-1 call and Level-2 boundary
becomes one payload of the observed case it belongs to.

* Level 1: forward hooks on ``nn.Linear``, ``nn.Conv2d``, ``nn.LayerNorm``
  (ConvNeXt's channels-first norms are recorded in the NHWC view they
  normalize), ``ConvNextV2GRN``, the RMSNorm, and the GELU / QuickGELU
  activations; leading dimensions are flattened into each task's 2-D view.
* Level 2: the boundaries Tier 3 installs at. Module boundaries
  (``vision.operators.bindings``, arguments read off the call's positional and
  keyword arguments): ConvNeXt's ``dwconv .. layernorm`` and ``pwconv1 ..
  pwconv2``, Qwen2-VL's ``VisionAttention``, ``VisionMlp``, ``Qwen2VLAttention``
  and ``Qwen2MLP``. Residual streams (``vision.operators.residual_streams``):
  each add of an attention or MLP branch to Qwen2-VL's vision or text stream
  with the norm that follows it, 2 per block. Their ``dsummed`` is the gradient
  of the next sum in the stream (the sum's other consumer); the last sum's
  (into ``merger.ln_q`` / ``model.norm``) has none and is supplied as zeros.

Output cotangents come from tensor hooks on each call's output. Recorders
change no arithmetic (checked per order by the shared loop).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

from .base import Case, invocation

ACTIVATIONS = {"GELUActivation": "gelu", "GELU": "gelu", "QuickGELUActivation": "quick_gelu"}
RMSNORMS = ("Qwen2RMSNorm", "Qwen2VLRMSNorm")


def _grad(record: dict, key: str, tensor) -> None:
    if tensor is not None and getattr(tensor, "requires_grad", False):
        tensor.register_hook(lambda g, r=record, k=key: r.__setitem__(k, g.detach().clone()))


class VisionCapture:
    """Hooks on an unmodified HF vision or vision-language model."""

    model_name = ""

    def __init__(self, model):
        self.model = model
        self.calls: list[dict] = []
        self.sites: list[dict] = []
        self.streams: list[dict] = []
        self._handles: list = []

    def install(self) -> "VisionCapture":
        import torch

        from evograd.benchmark.topdown.common.vision.operators import bindings, residual_streams

        for path, module in self.model.named_modules():
            kind = type(module).__name__
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)) \
                    or kind in ("ConvNextV2GRN", *RMSNORMS, *ACTIVATIONS):
                self._module(path, module, kind)
        for binding in bindings(self.model):
            self._site(binding)
        for stream in residual_streams(self.model):
            self._stream(stream)
        return self

    def _module(self, path, module, kind):
        state: dict = {}

        def pre(mod, args):
            state["record"] = {"path": path, "kind": kind, "module": mod, "x": args[0].detach()}

        def post(mod, args, output):
            record = state.pop("record")
            record["y"] = output.detach()
            _grad(record, "dy", output)
            self.calls.append(record)

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def _site(self, binding):
        state: dict = {}

        def pre(mod, args, kwargs):
            state["args"] = tuple(a.detach() if hasattr(a, "detach") else a
                                  for a in binding.pack(args, kwargs))

        def post(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            record = {"path": binding.path, "site": binding.site, "args": state.pop("args"),
                      "out": out.detach()}
            _grad(record, "dout", out)
            self.sites.append(record)

        self._handles.append(binding.first.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(binding.last.register_forward_hook(post))

    def _stream(self, stream):
        """Each norm's input (with its total gradient) and output, and each branch's output."""
        record = {"stream": stream, "norms": {}, "branches": {}}
        self.streams.append(record)
        for k, (path, norm) in enumerate(stream.norms):

            def pre(mod, args, k=k, path=path):
                entry = record["norms"][k] = {"path": path, "module": mod, "x": args[0].detach()}
                _grad(entry, "dx", args[0])

            def post(mod, args, output, k=k):
                entry = record["norms"][k]
                entry["y"] = output.detach()
                _grad(entry, "dy", output)

            self._handles.append(norm.register_forward_pre_hook(pre))
            self._handles.append(norm.register_forward_hook(post))
        for k, (_, branch) in enumerate(stream.branches, start=1):

            def branch_out(mod, args, output, k=k):
                out = output[0] if isinstance(output, tuple) else output
                record["branches"][k] = out.detach()

            self._handles.append(branch.register_forward_hook(branch_out))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def capture_for(model_name: str):
    """The recorder class of one vision model."""
    return type(f"{model_name}_capture", (VisionCapture,), {"model_name": model_name})


def _flat(t):
    return t.reshape(-1, t.shape[-1])


def _level1(r) -> tuple[str, dict, dict, dict, dict]:
    """``(task, args, cotangents, live outputs, notes)`` of one recorded module call."""
    import torch

    m, x, y, dy, kind = r["module"], r["x"], r["y"], r["dy"], r["kind"]
    if isinstance(m, torch.nn.Linear):
        args = {"x": _flat(x), "weight": m.weight}
        if m.bias is not None:
            args["bias"] = m.bias
        return ("linear" if m.bias is not None else "linear_no_bias", args, {"dy": _flat(dy)},
                {"y": _flat(y)}, {})
    if isinstance(m, torch.nn.Conv2d):
        return ("spatial_conv2d", {"x": x, "weight": m.weight, "bias": m.bias, "stride": m.stride[0],
                                   "padding": m.padding[0], "groups": m.groups},
                {"dout": dy}, {"out": y}, {})
    if isinstance(m, torch.nn.LayerNorm):
        notes = {}
        if getattr(m, "data_format", None) == "channels_first":
            x, y, dy = (t.permute(0, 2, 3, 1) for t in (x, y, dy))
            notes = {"layout": "channels-first norm recorded in its NHWC view, flattened"}
        return ("layernorm", {"x": _flat(x), "weight": m.weight, "bias": m.bias, "eps": m.eps},
                {"dy": _flat(dy)}, {"y": _flat(y)}, notes)
    if kind == "ConvNextV2GRN":
        return "grn", {"x": x, "weight": m.weight, "bias": m.bias}, {"dout": dy}, {"out": y}, {}
    if kind in RMSNORMS:
        return ("rmsnorm", {"x": _flat(x), "weight": m.weight, "eps": m.variance_epsilon},
                {"dy": _flat(dy)}, {"y": _flat(y)}, {})
    return ACTIVATIONS[kind], {"x": _flat(x)}, {"dout": _flat(dy)}, {"out": _flat(y)}, {}


def invocations(capture: VisionCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 boundary of one step, in its case's task ABI."""
    import torch

    from evograd.benchmark import get_task
    from evograd.benchmark.topdown.common.harvest_roles import role_of

    seen: dict[str, int] = {}

    def call(path, kind):
        seen[path] = seen.get(path, 0) + 1
        unique = path if seen[path] == 1 else f"{path}#{seen[path]}"
        # ``component`` resolves configurations the harvest separated by how the
        # model's tensor became the task's view (ConvNeXt's stem norm takes a
        # copy, its block norms a view): same task inputs, named by module role.
        return {"path": unique, "layer": None, "role": role_of(path), "component": role_of(path),
                "kind": kind}

    for r in capture.calls:
        task, args, cotangents, outputs, notes = _level1(r)
        yield invocation(cases, task, 1, call(r["path"], "module:" + r["kind"]), args, cotangents,
                         outputs, notes)
    for r in capture.sites:
        task = f"{capture.model_name}_{r['site']}"
        names = [a.name for a in get_task(task).args]
        yield invocation(cases, task, 2, call(r["path"] + "#" + r["site"], "boundary:" + r["site"]),
                         dict(zip(names, r["args"])), {"dout": r["dout"]}, {"out": r["out"]})
    for r in capture.streams:
        stream, norms = r["stream"], r["norms"]
        task = f"{capture.model_name}_{stream.site}"
        names = [a.name for a in get_task(task).args]
        for k in range(1, len(stream.norms)):
            norm = norms[k]
            args = dict(zip(names, (r["branches"][k], norms[k - 1]["x"],
                                    *stream.norm_args(norm["module"]))))
            if k + 1 < len(stream.norms):
                dsummed = norms[k + 1]["dx"]
                notes = {"dsummed_origin": "total gradient of the next sum in the stream"}
            else:
                dsummed = torch.zeros_like(norm["x"])
                notes = {"dsummed_origin": "absent in the model (the last sum feeds only the final "
                                           "norm); zeros supplied for the declared two-output ABI",
                         "absent_cotangents": ["dsummed"]}
            yield invocation(cases, task, 2, call(norm["path"] + "#" + stream.site,
                                                  "boundary:" + stream.site),
                             args, {"dnormed": norm["dy"], "dsummed": dsummed},
                             {"normed": norm["y"], "summed": norm["x"]}, notes)
