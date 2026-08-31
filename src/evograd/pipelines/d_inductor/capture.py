"""Capture Inductor's generated forward/backward kernels for a declared op.

AOTAutograd traces the declared forward reference into a joint graph, the
min-cut partitioner splits it, and Inductor lowers each half. This module runs
that pipeline and keeps the two generated Python modules plus the save-set the
partitioner chose -- the forward graph's outputs past the operator's own.

Everything here needs a real device: Inductor emits Triton on CUDA and C++ on
CPU. The capture logic is device-agnostic; only the kernel text differs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evograd.opdecl.activity import Active, OpDecl, Workload


@dataclass(frozen=True)
class SavedTensor:
    """One entry of the partitioner's save-set."""

    name: str
    producer: str
    shape: tuple[Any, ...]
    dtype: str
    is_input: bool
    # Size at the traced sizes. Under dynamic shapes the dims are symbolic, but
    # each carries the hint it was traced with, so this stays a real number --
    # the same quantity the benchmark harness reports as `saved_bytes`.
    nbytes_at_capture: int

    @property
    def is_tensor(self) -> bool:
        """Under dynamic shapes the save-set can also carry symbolic sizes."""
        return self.dtype.startswith("torch.")

    def describe(self) -> str:
        origin = "forward input" if self.is_input else self.producer
        if not self.is_tensor:
            return f"{self.name}: {self.dtype} <- {origin}"
        return f"{self.name}: {list(self.shape)} {self.dtype} <- {origin}"


@dataclass(frozen=True)
class CapturedPair:
    """Inductor output for one operator, plus the contract between the halves."""

    forward_source: str
    backward_source: str
    saved: tuple[SavedTensor, ...]
    grad_indices: tuple[int, ...]
    # How to build the backward's argument list: ("s", i) takes saved[i],
    # ("t", i) takes tangent i. Needed because AOTAutograd orders the backward's
    # placeholders with symbolic sizes first, while the forward returns them in
    # graph order -- so the two sides are a permutation apart, not identical.
    backward_arg_spec: tuple[tuple[str, int], ...]
    # Launch config the autotuner chose for each kernel, per side, keyed by the
    # kernel's original name. Only used when pinning configs (autotune off).
    kernel_configs: dict[str, dict[str, dict[str, Any]]]
    device: str
    dynamic: bool
    baked_scalars: dict[str, Any] = field(default_factory=dict)

    @property
    def saved_bytes_at_capture(self) -> int:
        return sum(s.nbytes_at_capture for s in self.saved)


def _tensor_arg_names(op: OpDecl) -> tuple[str, ...]:
    return tuple(a.name for a in op.args if getattr(a, "shape", None) is not None)


def _active_tensor_names(op: OpDecl) -> tuple[str, ...]:
    return tuple(
        a.name
        for a in op.args
        if isinstance(a, Active) and getattr(a, "shape", None) is not None
    )


def _grad_indices(op: OpDecl) -> tuple[int, ...]:
    """Position of each declared gradient among the backward's returns.

    AOTAutograd emits one slot per *tensor input* in input order, holding None
    where the input does not require grad -- so Inactive tensors such as
    ``res_mask`` occupy a slot too. ``op.grad_names()`` may reorder the
    contract (``grad_order``) or expose fewer, so select by index.
    """
    tensor_names = _tensor_arg_names(op)
    by_grad = {a.grad_name: a.name for a in op.active_args()}
    return tuple(tensor_names.index(by_grad[g]) for g in op.grad_names())


@contextlib.contextmanager
def _record_generated_modules(
    sink: dict[str, list[str]], modules: dict[str, list[Any]], slot: list[str]
):
    """Record each Inductor-generated module's source and the module itself.

    The module object is kept so the autotuner's chosen config can be read back
    after the capture has actually run the kernels.
    """
    import torch._inductor.graph as inductor_graph

    original = inductor_graph.GraphLowering.compile_to_module

    def patched(self):
        module = original(self)
        path = getattr(module, "__file__", None)
        if isinstance(path, str):
            try:
                sink.setdefault(slot[0], []).append(
                    Path(path).read_text(encoding="utf-8")
                )
                modules.setdefault(slot[0], []).append(module)
            except OSError:
                pass
        return module

    inductor_graph.GraphLowering.compile_to_module = patched
    try:
        yield
    finally:
        inductor_graph.GraphLowering.compile_to_module = original


def _selected_configs(modules: list[Any]) -> dict[str, dict[str, Any]]:
    """The launch config the autotuner settled on, per kernel.

    Read after the capture has run the kernels, so the sweep has happened and
    each CachingAutotuner has collapsed to a single launcher. Used to pin a
    config when autotuning is disabled -- pinning the winner rather than an
    arbitrary default keeps the untuned seed as fast as the tuned one.
    """
    configs: dict[str, dict[str, Any]] = {}
    for module in modules:
        for name in dir(module):
            if name.startswith("__"):
                continue
            launchers = getattr(getattr(module, name, None), "launchers", None)
            if not launchers:
                continue
            config = getattr(launchers[0], "config", None)
            if config is None:
                continue
            entry = dict(getattr(config, "kwargs", None) or {})
            for option in ("num_warps", "num_stages"):
                value = getattr(config, option, None)
                if value is not None:
                    entry[option] = value
            if entry:
                configs[name] = entry
    return configs


def _plain_dim(dim: Any) -> int | str:
    """Concrete dims stay ints; symbolic dims become their symbol name.

    A SymInt carries a hint, so ``int(dim)`` would silently report the traced
    size and hide the fact that the kernel is shape-generic.
    """
    return dim if isinstance(dim, int) else str(dim)


def _dim_hint(dim: Any) -> int:
    """The size a symbolic dim was traced with, read without guarding on it.

    ``int(sym_int)`` installs a guard, which -- since this runs before Inductor
    lowers the graph -- would specialize the kernel to the capture shape and
    throw away shape genericity. ``node.hint`` is a plain int and is inert.
    """
    if isinstance(dim, int):
        return dim
    hint = getattr(getattr(dim, "node", None), "hint", None)
    return int(hint) if isinstance(hint, int) else 0


def _describe_saved(nodes: list[Any]) -> tuple[SavedTensor, ...]:
    """Describe each saved value.

    Under dynamic shapes the partitioner may hand symbolic sizes across the
    boundary alongside tensors, so entries are not all tensors.
    """
    saved = []
    for node in nodes:
        val = node.meta.get("val")
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            shape = tuple(_plain_dim(d) for d in val.shape)
            dtype = str(val.dtype)
            numel = 1
            for dim in val.shape:
                numel *= _dim_hint(dim)
            nbytes = numel * val.element_size()
        else:
            shape = ()
            dtype = "symint" if val is not None else "unknown"
            nbytes = 0
        is_input = node.op == "placeholder"
        producer = "placeholder" if is_input else str(getattr(node, "target", node.op))
        saved.append(
            SavedTensor(
                name=str(node),
                producer=producer,
                shape=shape,
                dtype=dtype,
                is_input=is_input,
                nbytes_at_capture=nbytes,
            )
        )
    return tuple(saved)


def _check_triton_backend(device: str) -> None:
    """Fail early, and by name, when Triton cannot bring up its CUDA driver.

    On CUDA every Inductor kernel goes through ``triton_hash_with_backend``,
    which forces ``driver.active`` -- and that shells out to ``cc`` to build a
    small shim against libcuda and the CPython headers. Triton runs that compile
    with ``stdout=DEVNULL`` and re-raises a bare ``CalledProcessError``, so the
    failure arrives forty frames deep inside ``codegen_kernel`` looking like a
    lowering bug, while the compiler's actual message is only in the job log.
    Probing it here costs one cached call and keeps the diagnosis attached to
    the cause.
    """
    if not device.startswith("cuda"):
        return
    from torch.utils._triton import triton_hash_with_backend

    try:
        triton_hash_with_backend()
    except Exception as exc:  # noqa: BLE001 - re-raised with the real diagnosis
        raise RuntimeError(
            "Triton cannot initialize its CUDA driver, so no kernel can be "
            "lowered on this node -- this is an environment failure, not a "
            "capture failure. Run `python scripts/triton_env_check.py` here for "
            "the compiler's own error message and the inputs it depends on "
            f"(compiler, libcuda, Python headers). Underlying: {exc!r}"
        ) from exc


def capture_inductor_pair(
    op: OpDecl,
    workload: Workload,
    *,
    device: str = "cuda",
    dynamic: bool = True,
) -> CapturedPair:
    """Trace, partition, and lower ``op``'s forward; return both generated halves."""
    import torch
    import torch._inductor.config as inductor_config
    from functorch.compile import aot_function, min_cut_rematerialization_partition
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table

    from evograd.opdecl.importing import resolve_callable
    from evograd.opdecl.inputs import make_case_inputs

    _check_triton_backend(device)

    forward_fn = resolve_callable(op.forward)
    values = make_case_inputs(op, workload, device=device)

    tensor_names = _tensor_arg_names(op)
    active_names = set(_active_tensor_names(op))
    baked = {
        a.name: values.get(a.name, a.default)
        for a in op.args
        if getattr(a, "shape", None) is None
    }

    tensors = []
    for name in tensor_names:
        tensor = values[name].detach().clone()
        if name in active_names:
            tensor.requires_grad_(True)
        tensors.append(tensor)

    def traced(*args):
        bound = dict(zip(tensor_names, args))
        bound.update(baked)
        return forward_fn(*[bound[a.name] for a in op.args])

    sources: dict[str, list[str]] = {}
    modules: dict[str, list[Any]] = {}
    slot = ["forward"]
    # The fwd/bwd contract must be read *before* lowering. compile_fx_inner runs
    # post-grad passes that rewrite the graph in place, and fusion can replace an
    # output node with a getitem off a fused op -- so a save-set read afterwards
    # no longer lines up with the backward's placeholders, which keep their
    # original names.
    contract: dict[str, Any] = {}
    num_fwd_outputs = 1

    def fw_compiler(gm, example_inputs):
        out_node = next(n for n in gm.graph.nodes if n.op == "output")
        outputs = list(out_node.args[0])
        if len(outputs) < num_fwd_outputs:
            raise RuntimeError(f"{op.name}: forward graph has no outputs")
        saved_nodes = outputs[num_fwd_outputs:]
        contract["saved"] = _describe_saved(saved_nodes)
        contract["saved_names"] = [str(n) for n in saved_nodes]
        slot[0] = "forward"
        return compile_fx_inner(gm, example_inputs)

    def bw_compiler(gm, example_inputs):
        contract["bw_placeholders"] = [
            str(n) for n in gm.graph.nodes if n.op == "placeholder"
        ]
        slot[0] = "backward"
        return compile_fx_inner(gm, example_inputs)

    # A warm FX-graph cache skips codegen entirely, so there would be no module
    # to record. Force a real compile for the duration of the capture.
    with inductor_config.patch(force_disable_caches=True):
        with _record_generated_modules(sources, modules, slot):
            compiled = aot_function(
                traced,
                fw_compiler,
                bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
                # compile_fx_inner expects a graph already lowered through
                # Inductor's decomposition table; without it, ops that have
                # both a decomp and a fallback trip an assertion.
                decompositions=select_decomp_table(),
                dynamic=dynamic,
            )
            output = compiled(*tensors)
            if isinstance(output, (tuple, list)):
                output = output[0]
            dout = values[op.upstream_grad_name]
            differentiable = [t for t in tensors if t.requires_grad]
            slot[0] = "backward"
            torch.autograd.grad(output, differentiable, grad_outputs=dout)

    for side in ("forward", "backward"):
        emitted = sources.get(side, [])
        if not emitted:
            raise RuntimeError(
                f"{op.name}: Inductor produced no {side} module. The graph may "
                f"have been served from cache, or {side} lowering fell back "
                "entirely to extern kernels."
            )
        if len(emitted) > 1:
            raise RuntimeError(
                f"{op.name}: Inductor emitted {len(emitted)} {side} modules; "
                "this pipeline supports single-module graphs only."
            )

    return CapturedPair(
        forward_source=sources["forward"][0],
        backward_source=sources["backward"][0],
        saved=contract["saved"],
        grad_indices=_grad_indices(op),
        backward_arg_spec=_backward_arg_spec(
            op, contract["saved_names"], contract["bw_placeholders"]
        ),
        kernel_configs={
            side: _selected_configs(modules.get(side, []))
            for side in ("forward", "backward")
        },
        device=device,
        dynamic=dynamic,
        baked_scalars=baked,
    )


def _backward_arg_spec(
    op: OpDecl, saved_names: list[str], bw_placeholders: list[str]
) -> tuple[tuple[str, int], ...]:
    """Map the backward's placeholders onto (save-set entry | tangent) slots."""
    spec: list[tuple[str, int]] = []
    tangents_seen = 0
    for name in bw_placeholders:
        if name.startswith("tangents"):
            spec.append(("t", tangents_seen))
            tangents_seen += 1
            continue
        try:
            spec.append(("s", saved_names.index(name)))
        except ValueError:
            raise RuntimeError(
                f"{op.name}: backward expects {name!r}, which the forward does "
                f"not return; forward save-set is {saved_names}"
            ) from None
    if tangents_seen != 1:
        raise RuntimeError(
            f"{op.name}: backward takes {tangents_seen} tangents; this pipeline "
            "supports single-output forwards only"
        )
    return tuple(spec)
