"""Deterministic semantic seed assembly for compiler-aware synthesis.

The compiler contributes the forward/backward cut and external-call provenance.
Only the remaining ATen semantics are lowered through B's handwritten kernels;
no compiler-generated Triton source is copied into the seed.
"""

from __future__ import annotations

import ast
from typing import Any

from evograd.opdecl.activity import OpDecl, Workload, format_default
from evograd.pipelines.b_dispatch.program_codegen import (
    _ProgramBuilder,
    _ordered_arg_exprs,
    _pyname,
    _scalar_literal,
)
from evograd.pipelines.shared.artifact import render_deployment_layer


class UnsupportedSemanticNode(ValueError):
    """The evolvable region has no explicit handwritten implementation."""


def _outputs(graph: dict) -> list[Any]:
    node = graph["nodes"][-1]
    if node.get("op") != "output":
        raise ValueError("semantic graph must end with an output node")
    result = node["args"][0]
    return list(result) if isinstance(result, (tuple, list)) else [result]


def _output_expr(value: Any) -> str:
    if isinstance(value, str):
        return _pyname(value)
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_output_expr(v) for v in value) + ",)"
    return repr(value)


def _tuple_expr(values: list[str]) -> str:
    return "(" + ", ".join(values) + ("," if len(values) == 1 else "") + ")"


def _external_wrapper(side: str, node: dict) -> tuple[str, str, dict]:
    kernel = node["external_kernel"]
    target = node["target"]
    from evograd.pipelines.unified.capture import _SUPPORTED_EXTERNALS

    if _SUPPORTED_EXTERNALS.get(kernel) != target:
        raise UnsupportedSemanticNode(f"unsupported preserved call {target}: {kernel}")
    name = f"_unified_external_{side}_{_pyname(node['name'])}"
    names: dict[str, str] = {}

    def argument(entry):
        kind = entry["kind"]
        if kind in {"node", "sym_node"}:
            return names.setdefault(entry["name"], f"arg{len(names)}")
        if kind == "shape_list":
            return "[" + ", ".join(argument(item) for item in entry["items"]) + "]"
        if kind in {"torch_constant", "dtype", "layout", "memory_format"}:
            return entry["value"]
        if kind == "device":
            return f"torch.device({entry['value']!r})"
        return _scalar_literal(entry["value"])

    # Exact positional and named arguments also handle tuple-returning
    # convolution backward, without guessing a tensor-only call signature.
    args = [argument(entry) for entry in node["positional_args"]]
    args += [f"{key}={argument(value)}" for key, value in node.get("kwargs_ordered", {}).items()]
    callee = kernel.replace("extern_kernels.", "_unified_extern_kernels.", 1)
    prefix = f"def {name}({', '.join(names.values())}):\n    with torch.no_grad():\n"
    if kernel in {"extern_kernels.mm", "extern_kernels.bmm", "extern_kernels.addmm"}:
        shape, dtype = tuple(node["output_shape"]), node["output_dtype"]
        source = (prefix + f"        out = torch.empty({shape!r}, device=arg0.device, dtype={dtype})\n"
                  + f"        {callee}({', '.join(args)}, out=out)\n        return out\n")
    else:
        source = prefix + f"        return {callee}({', '.join(args)})\n"
    metadata = {"side": side, "node_name": node["name"], "target": target,
                "kernel": kernel, "wrapper": name}
    runtime = [_pyname(ref) for ref in names]
    return source, f"{name}({', '.join(runtime)})", metadata


def _semantic_expr(builder: _ProgramBuilder, node: dict) -> str:
    from evograd.atenir.primitive_triton.dispatch import make_kernel

    target = node["target"]
    ordered = node.get("args_ordered", [])
    args = _ordered_arg_exprs(ordered)
    operation = target.split(".")[1] if target.startswith("aten.") else ""
    if target == "prims.fma.default":
        from evograd.atenir.primitive_triton.elementwise import fma

        return f"{builder._resolve(fma)}({', '.join(args)})"
    if target == "prims.iota.default":
        from evograd.atenir.primitive_triton.elementwise import arange

        options = node["kwargs_ordered"]
        start = _scalar_literal(options["start"]["value"])
        step = _scalar_literal(options["step"]["value"])
        return (f"{builder._resolve(arange)}({start}, ({start}) + ({args[0]}) * ({step}), "
                f"{step}, dtype={node['output_dtype']}, device={options['device']['value']!r})")
    if target.startswith("prims.convert_element_type."):
        dtype = node.get("output_dtype")
        if not dtype or not dtype.startswith("torch."):
            raise UnsupportedSemanticNode(f"conversion has no output dtype: {target}")
        return f"{args[0]}.to(dtype={dtype})"
    if operation in {"alias", "detach"}:
        return args[0]
    if operation in {"view", "_unsafe_view", "reshape"}:
        return f"{args[0]}.reshape({args[1]})"
    if operation == "expand":
        return f"expand_copy({args[0]}, {args[1]})"
    if operation == "clone":
        return f"clone_copy({args[0]})"
    if operation == "permute":
        return f"permute_copy({args[0]}, {args[1]})"
    if operation == "t":
        if len(node["input_nodes"][0]["shape"]) < 2:
            return args[0]
        return f"permute_copy({args[0]}, (1, 0))"
    if operation == "transpose":
        rank = len(node["input_nodes"][0]["shape"])
        dims = list(range(rank))
        dim0, dim1 = int(ordered[1]["value"]), int(ordered[2]["value"])
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return f"permute_copy({args[0]}, {tuple(dims)!r})"
    if operation in {"squeeze", "unsqueeze"}:
        return f"{args[0]}.{operation}({', '.join(args[1:])})"
    if operation == "split":
        return f"{args[0]}.split({', '.join(args[1:])})"
    if operation == "select_scatter":
        from evograd.atenir.primitive_triton.scatter_gather import select_scatter

        return f"{builder._resolve(select_scatter)}({', '.join(args)})"
    if operation == "sym_size":
        return f"int({args[0]}.shape[{args[1]}])"
    if "getitem" in target:
        return f"{args[0]}[{args[1]}]"
    kernel = make_kernel(node)
    if builder._is_generic_fallback(kernel) or builder._is_unimplemented(kernel):
        raise UnsupportedSemanticNode(f"no handwritten semantic lowering for {target}")
    try:
        return builder.call_expr_for_node(node, kernel, args)
    except (ValueError, NotImplementedError) as exc:
        raise UnsupportedSemanticNode(f"{target}: {exc}") from exc


class _RejectFallbacks(ast.NodeTransformer):
    """Make conditional primitive-library math fallbacks explicit failures.

    GPU primitive helpers sometimes retain CPU fallback returns. Deployment is
    CUDA-only, and those paths must never silently execute PyTorch mathematics.
    """

    _torch_math = {"mm", "bmm", "addmm", "matmul", "sum", "mean", "prod", "var",
                   "softmax", "argmax", "argmin", "cumsum", "topk", "gather",
                   "scatter_add", "zeros_like", "ones_like", "full_like", "full"}
    _tensor_math = {"sum", "mean", "prod", "var", "amax", "amin", "any", "cumsum"}

    def visit_Return(self, node: ast.Return) -> ast.AST:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            fn = child.func
            if isinstance(fn.value, ast.Name) and fn.value.id in {"tl", "triton", "libdevice"}:
                continue
            is_torch = isinstance(fn.value, ast.Name) and fn.value.id == "torch"
            if (is_torch and fn.attr in self._torch_math) or (not is_torch and fn.attr in self._tensor_math):
                return ast.copy_location(ast.Raise(
                    exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                                 args=[ast.Constant("unsupported primitive fallback in semantic seed")],
                                 keywords=[]), cause=None), node)
        return self.generic_visit(node)


def _audit_primitive_library(tree: ast.AST) -> None:
    """Reject hidden host-side mathematics in any generated helper statement.

    Allocation, layout, dtype selection/conversion, and shape metadata remain
    legal host work. Arithmetic and reductions must execute in Triton. This
    audits assignments too, rather than relying on fallback-return rewriting.
    """
    torch_host_calls = {
        "empty", "empty_like", "empty_strided", "broadcast_tensors",
        "broadcast_shapes", "result_type", "promote_types", "is_tensor",
        "finfo", "iinfo", "get_default_dtype", "device", "split", "unbind",
    }
    tensor_math = _RejectFallbacks._tensor_math | {
        "mm", "bmm", "matmul", "addmm", "softmax", "log_softmax", "pow",
        "add", "sub", "mul", "div", "sqrt", "rsqrt", "exp", "log",
        "sigmoid", "tanh", "relu", "scatter", "scatter_add", "gather",
    }

    def dotted(expr):
        if isinstance(expr, ast.Name):
            return [expr.id]
        if isinstance(expr, ast.Attribute):
            prefix = dotted(expr.value)
            return prefix + [expr.attr] if prefix else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            raise UnsupportedSemanticNode("host matrix multiplication in primitive library")
        if not isinstance(node, ast.Call):
            continue
        path = dotted(node.func)
        if path and path[0] == "torch":
            if len(path) != 2 or path[1] not in torch_host_calls:
                raise UnsupportedSemanticNode(f"hidden PyTorch call in primitive library: {'.'.join(path)}")
        elif isinstance(node.func, ast.Attribute) and node.func.attr.rstrip("_") in tensor_math:
            if not path or path[0] not in {"tl", "triton", "libdevice"}:
                raise UnsupportedSemanticNode(f"hidden tensor math in primitive library: {node.func.attr}")


def _pair_source(op: OpDecl, workload: Workload, captured: Any) -> str:
    pair = captured.pair
    tensor_args = [arg for arg in op.args if getattr(arg, "shape", None) is not None]
    names = [arg.name for arg in tensor_args]
    scalars = list(op.scalar_inactive_args())
    scalar_sig = "".join(
        f", {arg.name}" + (f"={format_default(arg.default)}" if arg.default is not None else "")
        for arg in scalars
    )
    placeholders = [node for node in captured.forward_graph["nodes"] if node["op"] == "placeholder"]
    if len(placeholders) != len(names):
        raise ValueError("forward placeholder count does not match declaration tensor arguments")
    guards = []
    for arg, node in zip(tensor_args, placeholders):
        expected = tuple(node["shape"])
        dtype = node["dtype"]
        guards.extend([
            f"    if not isinstance({arg.name}, torch.Tensor) or not {arg.name}.is_cuda:",
            f"        raise ValueError({('unified seed requires CUDA tensor ' + arg.name)!r})",
            f"    if tuple({arg.name}.shape) != {expected!r} or {arg.name}.dtype != {dtype} or {arg.name}.device != torch.device({pair.device!r}):",
            f"        raise ValueError({('unified seed configuration mismatch for ' + arg.name)!r})",
        ])
    if len(names) > 1:
        guards.extend([
            f"    if any(t.device != {names[0]}.device for t in {_tuple_expr(names[1:])}):",
            "        raise ValueError('all unified seed inputs must use the same CUDA device')",
        ])
    scalar_guards = []
    for arg in scalars:
        value = pair.baked_scalars.get(arg.name, arg.default)
        scalar_guards.extend([
            f"    if {arg.name} != {value!r}:",
            f"        raise ValueError({('unified seed scalar configuration mismatch for ' + arg.name)!r})",
        ])
    n_outputs = len(op.outputs)
    if len(_outputs(captured.forward_graph)) != n_outputs + len(pair.saved):
        raise ValueError("forward outputs do not match user outputs plus captured save-set")
    if any(not value.is_tensor for value in pair.saved):
        raise ValueError("initial static unified seeds only support tensor saved state")
    upstream_names = list(op.upstream_grad_names)
    upstream_param = op.upstream_grad_parameter
    bwd_args = [f"saved_tensors[{index}]" if kind == "s" else f"{upstream_names[index]}.contiguous()"
                for kind, index in pair.backward_arg_spec]
    bwd_placeholders = [node for node in captured.backward_graph["nodes"] if node["op"] == "placeholder"]
    if len(bwd_placeholders) != len(bwd_args):
        raise ValueError("backward argument spec does not match semantic graph")
    unpack = []
    if len(upstream_names) > 1:
        unpack = [f"    {', '.join(upstream_names)} = {upstream_param}"]
    picked = [f"_grads[{index}]" for index in pair.grad_indices]
    backward_guards = []
    for (kind, index), node in zip(pair.backward_arg_spec, bwd_placeholders):
        value = f"saved_tensors[{index}]" if kind == "s" else upstream_names[index]
        backward_guards.extend([
            f"    if not isinstance({value}, torch.Tensor) or not {value}.is_cuda:",
            "        raise ValueError('unified backward requires CUDA tensors')",
            f"    if tuple({value}.shape) != {tuple(node['shape'])!r} or {value}.dtype != {node['dtype']} or {value}.device != torch.device({pair.device!r}):",
            "        raise ValueError('unified backward configuration mismatch')",
        ])
    lines = [
        f"def {op.forward_fn_name}({', '.join(names)}{scalar_sig}):",
        *guards, *scalar_guards,
        f"    with torch.cuda.device({pair.device!r}):",
        f"        _outs = _unified_forward_graph({', '.join(names)})",
        f"    return {'_outs[0]' if n_outputs == 1 else f'_outs[:{n_outputs}]'}, tuple(_outs[{n_outputs}:])",
        "",
        f"def {op.backward_fn_name}({upstream_param}, saved_tensors{scalar_sig}):",
        *scalar_guards, *unpack,
        f"    if len(saved_tensors) != {len(pair.saved)}:",
        "        raise ValueError('unified seed saved-state arity mismatch')",
        *backward_guards,
        f"    with torch.cuda.device({pair.device!r}):",
        f"        _grads = _unified_backward_graph({', '.join(bwd_args)})",
        f"    return {_tuple_expr(picked)}",
    ]
    return "\n".join(lines) + "\n"


def generate_unified_seed(op: OpDecl, workload: Workload, captured: Any) -> str:
    """Emit one statically specialized, compiler-aware handwritten seed."""
    if captured.pair.dynamic:
        raise ValueError("unified pipeline currently requires static capture")
    if not str(captured.pair.device).startswith("cuda"):
        raise ValueError("unified semantic seeds require a CUDA capture")
    from evograd.pipelines.b_dispatch.layout_kernels import embeddable_source

    builder = _ProgramBuilder()
    drivers, wrappers, regions = [], [], []
    for side, graph in (("forward", captured.forward_graph), ("backward", captured.backward_graph)):
        placeholders = [node["name"] for node in graph["nodes"] if node["op"] == "placeholder"]
        lines = [f"def _unified_{side}_graph({', '.join(map(_pyname, placeholders))}):"]
        for node in graph["nodes"]:
            if node["op"] != "call_function":
                if node["op"] not in {"placeholder", "output"}:
                    raise UnsupportedSemanticNode(f"unsupported semantic node kind {node['op']}")
                continue
            if node.get("external_kernel"):
                wrapper, expr, metadata = _external_wrapper(side, node)
                wrappers.append(wrapper)
                regions.append(metadata)
            else:
                expr = _semantic_expr(builder, node)
            lines.append(f"    {_pyname(node['name'])} = {expr}  # {node['target']}")
        lines.append(f"    return {_tuple_expr([_output_expr(value) for value in _outputs(graph)])}")
        drivers.append("\n".join(lines))
    expected_regions = {(region.side, region.node_name, region.kernel) for region in captured.external_regions}
    actual_regions = {(region["side"], region["node_name"], region["kernel"]) for region in regions}
    if expected_regions != actual_regions:
        raise ValueError("external-region manifest does not match graph annotations")
    library = builder.render_kernel_library() + "\n" + embeddable_source()
    parsed = _RejectFallbacks().visit(ast.parse(library))
    _audit_primitive_library(parsed)
    library = ast.unparse(ast.fix_missing_locations(parsed))
    source = "\n\n".join([
        '"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""',
        "from __future__ import annotations\nimport torch\nimport triton\nimport triton.language as tl\n"
        "import triton.language.extra.cuda.libdevice as libdevice\n"
        "from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels",
        f"UNIFIED_EXTERNAL_REGIONS = {regions!r}",
        f"UNIFIED_DEPLOYMENT_CONFIG = {{'dims': {workload.dims!r}, 'dtype': {workload.dtype!r}, 'device': {captured.pair.device!r}, 'family': {op.family!r}, 'dynamic': False}}",
        *wrappers,
        "# EVOLVE-BLOCK-START",
        library, *drivers, _pair_source(op, workload, captured),
        "# EVOLVE-BLOCK-END",
        render_deployment_layer(op),
    ]) + "\n"
    compile(source, "<unified-semantic-seed>", "exec")
    return source
