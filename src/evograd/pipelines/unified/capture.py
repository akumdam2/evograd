"""Capture semantic ATen graphs and Inductor's external-kernel decisions.

The generated Inductor kernels are evidence for the partition only: unified's
seed rebuilds every remaining semantic node with primitive lowering. Shapes
and scalar arguments are fixed by the supplied deployment workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evograd.opdecl.activity import OpDecl, Workload
from evograd.pipelines.d_inductor.capture import CapturedPair, capture_inductor_pair


@dataclass(frozen=True)
class ExternalRegion:
    """A supported external call, tied to an exact semantic graph node."""

    side: str
    node_name: str
    target: str
    kernel: str
    buffer_name: str


@dataclass(frozen=True)
class UnifiedCapture:
    pair: CapturedPair
    joint_graph: dict[str, Any]
    forward_graph: dict[str, Any]
    backward_graph: dict[str, Any]
    external_regions: tuple[ExternalRegion, ...]


# This is a rendering capability list, not the L/R partition rule. A node is
# external only when the actual compiler IR below selects that exact kernel.
_SUPPORTED_EXTERNALS = {
    "extern_kernels.mm": "aten.mm.default",
    "extern_kernels.bmm": "aten.bmm.default",
    "extern_kernels.addmm": "aten.addmm.default",
    "extern_kernels.convolution": "aten.convolution.default",
    "torch.ops.aten.convolution_backward.default": "aten.convolution_backward.default",
}


def _encode_arg(value: Any) -> dict[str, Any]:
    from evograd.atenir.extract import _serialize_arg

    return _serialize_arg(value)


def _capture_device(device: str) -> str:
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return str(resolved)


def _serialize_graph(gm: Any) -> dict[str, Any]:
    from evograd.atenir.extract import _serialise

    # _serialise is the schema consumed by B's primitive code generator. Keep
    # exact positional/keyword arguments as well: its flattened args_ordered
    # intentionally elides reduction parameters and keyword names.
    graph = _serialise(gm)
    by_name = {n["name"]: n for n in graph["nodes"] if "name" in n}
    for node in gm.graph.nodes:
        if node.op in {"placeholder", "call_function"}:
            value = node.meta.get("val")
            if hasattr(value, "stride"):
                by_name[node.name]["stride"] = list(value.stride())
            if isinstance(value, (tuple, list)):
                by_name[node.name]["output_shapes"] = [
                    list(item.shape) if hasattr(item, "shape") else None for item in value
                ]
                by_name[node.name]["output_dtypes"] = [
                    str(item.dtype) if hasattr(item, "dtype") else None for item in value
                ]
        elif node.op != "output":
            raise NotImplementedError(
                f"Unified pipeline requires functional ATen graphs, got {node.op}: {node}"
            )
    return graph


def _external_regions(side: str, lowering: Any, graph: dict[str, Any]) -> list[ExternalRegion]:
    from torch._inductor import ir

    by_name = {n["name"]: n for n in graph["nodes"] if "name" in n}
    regions: list[ExternalRegion] = []
    seen_buffers: set[int] = set()
    seen_nodes: set[str] = set()
    # Externs normally appear in both lists. Include operations so a compiler
    # version that registers a side-effecting extern only there cannot hide it.
    for buffer in [*lowering.buffers, *getattr(lowering, "operations", ())]:
        if isinstance(buffer, ir.MultiOutput):
            # A getitem-like descriptor for one result of an already recorded
            # multi-output call (e.g. convolution_backward), not another launch.
            continue
        deferred_type = getattr(ir, "MultiTemplateBuffer", ())
        if isinstance(buffer, deferred_type):
            raise NotImplementedError(
                "Unified pipeline cannot classify a deferred compiler autotuning "
                "choice before its external-vs-generated kernel decision is final"
            )
        if not isinstance(buffer, ir.ExternKernel) or id(buffer) in seen_buffers:
            continue
        seen_buffers.add(id(buffer))
        kernel = getattr(buffer, "python_kernel_name", None)
        target = _SUPPORTED_EXTERNALS.get(kernel)
        if target is None:
            raise NotImplementedError(
                f"Unified pipeline cannot preserve {side} external kernel {kernel!r}; "
                "add a verified semantic mapping and renderer first"
            )
        # ExternKernel.fx_node is the exact call being lowered. origins includes
        # transitive inputs as well, so selecting an arbitrary origin is unsafe.
        origin = getattr(buffer, "fx_node", None)
        if origin is None:
            origin = getattr(buffer, "origin_node", None)
        name = getattr(origin, "name", None)
        semantic = by_name.get(name)
        if (
            semantic is None
            or semantic.get("target") != target
            or str(getattr(origin, "target", "")) != target
            or semantic.get("positional_args") != [_encode_arg(a) for a in origin.args]
            or semantic.get("kwargs_ordered")
            != {k: _encode_arg(v) for k, v in origin.kwargs.items()}
        ):
            raise NotImplementedError(
                f"Unified pipeline cannot map {side} {kernel} ({name!r}) back to an "
                "unchanged semantic ATen call; compiler fusion/rewriting needs "
                "an explicit region mapping"
            )
        if name in seen_nodes:
            raise NotImplementedError(
                f"Unified pipeline found multiple external calls for {side} node {name!r}"
            )
        seen_nodes.add(name)
        semantic["external_kernel"] = kernel
        regions.append(
            ExternalRegion(
                side=side,
                node_name=name,
                target=target,
                kernel=kernel,
                buffer_name=buffer.get_name(),
            )
        )
    return regions


def capture_unified_pair(
    op: OpDecl, workload: Workload, *, device: str = "cuda"
) -> UnifiedCapture:
    """Capture one fixed deployment configuration, without invoking an LLM.

    Unsupported external lowerings fail closed; treating them as primitive R
    nodes would silently change the compiler-aware partition.
    """
    import torch
    from torch._inductor.decomposition import select_decomp_table

    from evograd.atenir.extract import _decomposition_table

    graphs: dict[str, dict[str, Any]] = {}
    regions: list[ExternalRegion] = []

    def observe_graph(side: str, gm: Any) -> None:
        if side in graphs:
            raise NotImplementedError(f"Unified pipeline captured multiple {side} graphs")
        graphs[side] = _serialize_graph(gm)

    def observe_lowering(side: str, lowering: Any) -> None:
        regions.extend(_external_regions(side, lowering, graphs[side]))

    # Retain the compiler's decomposition choices except for var_mean: B's
    # pure-ATen implementation avoids tuple-valued prims.var in the semantic
    # graph. The same decomposed graph feeds both the compiler and seed.
    decompositions = dict(select_decomp_table())
    primitive_decompositions = _decomposition_table()
    decompositions[torch.ops.aten.var_mean.correction] = primitive_decompositions[
        torch.ops.aten.var_mean.correction
    ]
    for target in (
        torch.ops.aten.native_layer_norm.default,
        torch.ops.aten.native_layer_norm_backward.default,
    ):
        if target in primitive_decompositions:
            decompositions.setdefault(target, primitive_decompositions[target])

    pair = capture_inductor_pair(
        op,
        workload,
        device=_capture_device(device),
        dynamic=False,
        graph_observer=observe_graph,
        lowering_observer=observe_lowering,
        decompositions=decompositions,
        allow_multi_output=True,
    )
    if set(graphs) != {"joint", "forward", "backward"}:
        raise RuntimeError(f"Incomplete AOTAutograd capture: {sorted(graphs)}")
    return UnifiedCapture(
        pair=pair,
        joint_graph=graphs["joint"],
        forward_graph=graphs["forward"],
        backward_graph=graphs["backward"],
        external_regions=tuple(regions),
    )
