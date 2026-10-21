"""What a real training step of DiT-S/2 or DLRM records, and how.

Shared by the ``dit_s_2`` and ``torchrec_dlrm`` integrations (both models are
evograd's own ``generated.model``). One native step of the streamed real-data
workload runs per data order with recorders attached; every Level-1 call and
Level-2 site call becomes one payload of the observed case it belongs to.

* ``F.linear`` (functional in DiT's blocks and DLRM's dense MLP, and inside
  ``nn.Linear``): a pass-through recorder; leading dimensions are flattened
  into the task's ``[M, K]`` view.
* ``nn.Conv2d`` and ``nn.EmbeddingBag``: module forward hooks; the bag's
  indices and offsets are recorded as passed.
* Level-2 sites (``mod_fn``/``mlp_fn`` of every DiT block, DLRM's
  ``interaction_fn``): the model attribute is wrapped by a pass-through that
  records the task-ABI arguments.

Output cotangents come from tensor hooks on each call's output (the total
gradient of that output). Recorders change no arithmetic (checked per order
by the shared loop).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

from .base import Case, invocation

#: Level-2 site attribute -> its task suffix (``<model>_<suffix>``).
SITES = {"dit_s_2": {"mod_fn": "dit_modulation", "mlp_fn": "dit_mlp"},
         "torchrec_dlrm": {"interaction_fn": "dlrm_interaction"}}


def _grad(record: dict, key: str, tensor) -> None:
    if tensor is not None and getattr(tensor, "requires_grad", False):
        tensor.register_hook(lambda g, r=record, k=key: r.__setitem__(k, g.detach().clone()))


class GeneratedCapture:
    """Hooks and pass-through recorders on an unmodified generated model."""

    model_name = ""

    def __init__(self, model):
        self.model = model
        self.linear: list[dict] = []
        self.conv: list[dict] = []
        self.bags: list[dict] = []
        self.sites: list[dict] = []
        self._handles: list = []
        self._restore: list[Callable[[], None]] = []

    def install(self) -> "GeneratedCapture":
        import torch
        import torch.nn.functional as F

        capture, model = self, self.model
        names = {id(p): n for n, p in model.named_parameters()}
        original_linear = F.linear

        def linear(x, weight, bias=None):
            y = original_linear(x, weight, bias)
            record = {"path": names.get(id(weight), "functional"), "x": x.detach(),
                      "weight": weight, "bias": bias, "y": y.detach()}
            _grad(record, "dy", y)
            capture.linear.append(record)
            return y

        F.linear = linear
        self._restore.append(lambda: setattr(F, "linear", original_linear))
        for path, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._module(path, module, self.conv, self._conv_args)
            elif isinstance(module, torch.nn.EmbeddingBag):
                self._module(path, module, self.bags, self._bag_args)
        for path, owner, attribute, suffix in self._site_owners():
            self._wrap_site(path, owner, attribute, f"{self.model_name}_{suffix}")
        return self

    def _site_owners(self):
        if self.model_name == "dit_s_2":
            for i, block in enumerate(self.model.blocks):
                for attribute, suffix in SITES["dit_s_2"].items():
                    yield f"blocks.{i}.{attribute}", block, attribute, suffix
        else:
            for attribute, suffix in SITES["torchrec_dlrm"].items():
                yield attribute, self.model, attribute, suffix

    def _wrap_site(self, path, owner, attribute, task):
        from evograd.benchmark import get_task

        names = [a.name for a in get_task(task).args]
        original = getattr(owner, attribute)
        capture = self

        def site(*args):
            out = original(*args)
            record = {"path": path, "task": task,
                      "args": {n: (a.detach() if hasattr(a, "detach") else a) for n, a in zip(names, args)},
                      "out": out.detach()}
            _grad(record, "dout", out)
            capture.sites.append(record)
            return out

        setattr(owner, attribute, site)
        self._restore.append(lambda: setattr(owner, attribute, original))

    @staticmethod
    def _conv_args(module, args):
        return {"x": args[0].detach(), "weight": module.weight, "bias": module.bias,
                "stride": module.stride[0], "padding": module.padding[0], "groups": module.groups}

    @staticmethod
    def _bag_args(module, args):
        return {"weight": module.weight, "indices": args[0].detach(), "offsets": args[1].detach()}

    def _module(self, path, module, sink, read_args):
        state: dict = {}

        def pre(mod, args):
            state["record"] = {"path": path, "args": read_args(mod, args)}

        def post(mod, args, output):
            record = state.pop("record")
            record["out"] = output.detach()
            _grad(record, "dout", output)
            sink.append(record)

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        for restore in reversed(self._restore):
            restore()
        self._handles.clear()
        self._restore.clear()


def capture_for(model_name: str):
    """The recorder class of one generated model."""
    return type(f"{model_name}_capture", (GeneratedCapture,), {"model_name": model_name})


def invocations(capture: GeneratedCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 site call of one step, in its case's task ABI."""
    from evograd.benchmark.topdown.common.harvest_roles import role_of

    def call(path, kind):
        return {"path": path, "layer": None, "role": role_of(path), "kind": kind}

    seen: dict[str, int] = {}

    def unique(path):
        seen[path] = seen.get(path, 0) + 1
        return path if seen[path] == 1 else f"{path}#{seen[path]}"

    for r in capture.linear:
        x, y, dy = r["x"], r["y"], r["dy"]
        k, n = x.shape[-1], y.shape[-1]
        task = "linear" if r["bias"] is not None else "linear_no_bias"
        args = {"x": x.reshape(-1, k), "weight": r["weight"]}
        if r["bias"] is not None:
            args["bias"] = r["bias"]
        yield invocation(cases, task, 1, call(unique(r["path"]), "function:F.linear"), args,
                         {"dy": dy.reshape(-1, n)}, {"y": y.reshape(-1, n)},
                         {"layout": "leading dimensions flattened"} if x.dim() > 2 else {})
    for r in capture.conv:
        yield invocation(cases, "spatial_conv2d", 1, call(unique(r["path"]), "module:nn.Conv2d"),
                         r["args"], {"dout": r["dout"]}, {"out": r["out"]})
    for r in capture.bags:
        yield invocation(cases, "embedding_bag", 1, call(unique(r["path"]), "module:nn.EmbeddingBag"),
                         r["args"], {"dout": r["dout"]}, {"out": r["out"]})
    for r in capture.sites:
        yield invocation(cases, r["task"], 2, call(unique(r["path"]), "site:" + r["path"]),
                         r["args"], {"dout": r["dout"]}, {"out": r["out"]})
