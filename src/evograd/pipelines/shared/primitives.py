"""Trusted Level-1 primitives an evolved Level-2 operator may call.

The default search space for a generated pair is Triton and nothing else, which
is the right default: an operator that quietly calls ``F.linear`` has not been
evolved, it has been wrapped. But "no PyTorch at all" is a different rule from
"nothing untrusted", and for a Level-2 operator built around dense projections
the two disagree. `qwen3_qkv_norm_rope`'s measured loss to Inductor is entirely
in hand-written GEMMs: 859 of 1050 device microseconds in three matmuls that
cuBLAS does in 165. Forbidding cuBLAS there does not make the search harder, it
makes the objective unreachable.

So a primitive is *opted into by name*, per run, and nothing is relaxed for any
operator that did not ask:

    allowed_primitives = ("vendor_gemm",)

What that buys is one fixed function, rendered outside the evolve block and
compared against this file's copy, which the evolvable pair may call. What it
does not buy is a fallback: the whole-operator spellings, autograd, and
``torch.compile`` stay forbidden, and any contraction the artifact performs
outside the trusted primitive is a violation whether or not the capability was
granted.

Enforcement is on the AST, not on tokens. A comment cannot grant a capability, a
string cannot hide a call, and ``a @ b`` is the same violation as
``torch.matmul(a, b)``.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from typing import Any

#: Library entry points rooted at ``torch``, by exact name: contractions
#: (vector, distance and sparse ones included), ``kl_div`` (a loss without the
#: word) and ``torch.compile``. Denied inside the evolvable region always;
#: permitted inside a trusted primitive's own fixed body when it declares them.
_TORCH_DENIED = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "einsum", "tensordot",
    "dot", "inner", "outer", "chain_matmul", "matrix_power", "kron",
    "mv", "addmv", "addmv_", "addr", "ger", "vdot", "cdist", "multi_dot", "vecdot",
    "smm", "spmm", "hspmm", "dsmm", "hsmm", "saddmm", "sspaddmm", "sampled_addmm",
    "kl_div", "compile",
})

#: Library families denied wherever they appear in a ``torch`` name, in any
#: case: ``torch.native_layer_norm``, ``torch.embedding_bag``, ``torch.ctc_loss``,
#: ``torch.bilinear``, ``torch.fbgemm_linear_fp16_weight`` and every attention
#: spelling are a library routine under another name. An opaque softmax counts:
#: an attention decomposition whose softmax stage is ``torch.softmax`` has been
#: wrapped, not evolved.
_DENIED_TORCH_FAMILIES = ("attention", "softmax", "embedding", "layer_norm", "group_norm",
                          "batch_norm", "instance_norm", "rms_norm", "loss", "cross_entropy",
                          "linear", "gemm")

#: Library families denied by the shape of a ``torch`` name (searched in its
#: lower case), keyed by how the task statement names them: every cuDNN/MIOpen/
#: oneDNN convolution entry point, and the cuDNN recurrent layers.
_DENIED_TORCH_PATTERNS = {
    "conv1d/2d/3d, conv_tbc, conv_transpose*, *convolution*": re.compile(r"conv(\d|_t|olution)"),
    "lstm, gru, rnn_tanh, rnn_relu (and their cells)":
        re.compile(r"^(quantized_)?(lstm|gru|rnn_tanh|rnn_relu)(_cell)?$|^(miopen|mkldnn)_rnn"),
}

#: Namespaces no generated pair math may reach into at all. ``ctypes`` and
#: ``cffi`` would load a compiled library that calls cuBLAS or cuDNN directly;
#: ``importlib`` and ``builtins`` would import or look up a name unseen.
_DENIED_ROOTS = frozenset({"F", "functional", "transformers", "evograd", "liger_kernel",
                           "ctypes", "cffi", "importlib", "builtins"})

#: Denied whatever the receiver is (``torch`` included): the method and
#: operator-module spellings of a contraction, ``x.mv(v)``, ``x.__matmul__(w)``,
#: ``operator.matmul(x, w)``. ``dot`` is deliberately absent: ``tl.dot`` is the
#: Triton language's own contraction and is the evolved kernel's arithmetic,
#: not an escape from it.
_DENIED_METHODS = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "linear", "backward",
    "softmax", "log_softmax",
    "mv", "addmv", "addmv_", "addbmm", "addbmm_", "addmm_", "baddbmm_", "addr", "addr_",
    "ger", "outer", "inner", "kron", "vdot", "matrix_power", "smm", "sspaddmm",
    "__matmul__", "__rmatmul__", "__imatmul__", "imatmul", "rmatmul",
})

#: Namespaces under ``torch`` that are denied whole: module wrappers
#: (``torch.nn.Conv2d``, ``torch.nn.grad.conv2d_input``), autograd and its
#: functional and checkpointing forms, tracing and compilation, sparse
#: contractions, C++/CUDA extensions, and ``torch.ops`` (whole-operator escapes
#: the leaf rule cannot see: ``torch.ops.aten._scaled_dot_product_flash_attention``
#: is fused attention with a different name). Private modules (``torch._C``,
#: ``torch._inductor``) are denied at any depth by :func:`_denied_torch_path`.
_DENIED_TORCH_PREFIXES = (("nn",), ("autograd",), ("func",), ("utils", "checkpoint"),
                          ("jit",), ("fx",), ("compiler",), ("export",), ("ops",),
                          ("sparse",), ("utils", "cpp_extension"))


def _denied_torch_leaf(leaf: str) -> bool:
    """A ``torch`` name the rules deny: by exact name, as a method, or by family."""
    low = leaf.lower()
    return (leaf in _TORCH_DENIED or leaf in _DENIED_METHODS
            or any(family in low for family in _DENIED_TORCH_FAMILIES)
            or any(pattern.search(low) for pattern in _DENIED_TORCH_PATTERNS.values()))


class PrimitiveViolation(RuntimeError):
    """Generated pair math used something it was not granted."""


@dataclass(frozen=True)
class TrustedPrimitive:
    """One fixed Level-1 provider: its name, its symbols, its exact source."""

    name: str
    symbols: tuple[str, ...]
    source: str
    #: torch leaves this primitive's own body is allowed to call.
    uses: tuple[str, ...]
    summary: str


VENDOR_GEMM_SOURCE = '''

# ── trusted primitive: vendor_gemm ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it.

def vendor_gemm(a, b):
    """Row-major 2-D GEMM through the vendor library (cuBLAS via ``torch.mm``).

    Two dimensions only, so the caller owns every reshape and the primitive
    cannot be handed a batched contraction it would silently loop. A
    ``.t()`` argument stays a metadata-only view -- cuBLAS consumes the
    transposed layout directly -- so ``vendor_gemm(x, w.t())`` costs one
    kernel and no copy.

    ``no_grad`` is not an optimisation: the pair differentiates itself, and a
    graph built here would be a second, hidden derivative.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"vendor_gemm is 2-D; got {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"vendor_gemm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    with torch.no_grad():
        return torch.mm(a, b)
'''


VENDOR_SDPA_SOURCE = '''

# ── trusted primitive: vendor_sdpa ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it. The fused-attention kernel eager PyTorch itself runs, chosen the
# way its dispatcher chooses it: cuDNN in a training configuration, flash
# under torch.use_deterministic_algorithms (cuDNN's backward is not
# reproducible, so the deterministic dispatcher excludes it). Verification
# runs deterministic and compares flash with flash; timing runs as training
# does and compares cuDNN with cuDNN.

def _vendor_sdpa_deterministic():
    return torch.are_deterministic_algorithms_enabled()


def vendor_sdpa_forward(q, k, v, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, forward.

    ``q`` is ``[B, HQ, T, D]``; ``k`` and ``v`` are ``[B, HK, T, D]`` with
    ``HK`` dividing ``HQ`` (grouped-query attention is the kernel's own). No
    mask, no dropout. Returns ``(out, logsumexp)``; ``logsumexp`` is what the
    backward needs and is the caller's to save (its layout is the backend's;
    pass it back unchanged). ``no_grad`` because the pair differentiates itself.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("vendor_sdpa_forward takes [B, H, T, D] tensors")
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, bool(is_causal), False, scale=scale)
        else:
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
                q, k, v, None, True, 0.0, bool(is_causal), False, scale=scale)
    return out, logsumexp


def vendor_sdpa_backward(dout, q, k, v, out, logsumexp, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, backward.

    Takes the forward's inputs and both of its results, from the same mode
    (deterministic or not) as the forward; returns ``(dq, dk, dv)`` with the
    shapes of ``q``, ``k`` and ``v``.
    """
    if dout.shape != out.shape:
        raise ValueError(f"vendor_sdpa_backward: dout {tuple(dout.shape)} vs out {tuple(out.shape)}")
    # No dropout, no mask, dense [B, H, T, D] inputs: philox state, attn_bias
    # and the nested-layout cum_seq tensors stay undefined (None). A defined
    # cum_seq tensor would select the ragged layout.
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, q.shape[2], k.shape[2], 0.0,
                bool(is_causal), None, None, scale=scale)
        else:
            dq, dk, dv = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, None, None, None,
                q.shape[2], k.shape[2], 0.0, bool(is_causal), scale=scale)
    return dq, dk, dv
'''


REGISTRY: dict[str, TrustedPrimitive] = {
    "vendor_gemm": TrustedPrimitive(
        name="vendor_gemm",
        symbols=("vendor_gemm",),
        source=VENDOR_GEMM_SOURCE,
        uses=("mm",),
        summary="2-D row-major GEMM through cuBLAS (torch.mm), no autograd",
    ),
    "vendor_sdpa": TrustedPrimitive(
        name="vendor_sdpa",
        symbols=("_vendor_sdpa_deterministic", "vendor_sdpa_forward", "vendor_sdpa_backward"),
        source=VENDOR_SDPA_SOURCE,
        uses=("_scaled_dot_product_cudnn_attention",
              "_scaled_dot_product_cudnn_attention_backward",
              "_scaled_dot_product_flash_attention",
              "_scaled_dot_product_flash_attention_backward"),
        summary=("fused causal (grouped-query) attention through the vendor kernel eager runs "
                 "(cuDNN; flash under deterministic algorithms), as an explicit "
                 "forward/backward pair, no autograd"),
    ),
}


def normalize(allowed) -> tuple[str, ...]:
    """Validate and order a requested capability list."""
    names = tuple(sorted({str(name) for name in (allowed or ())}))
    unknown = [name for name in names if name not in REGISTRY]
    if unknown:
        raise PrimitiveViolation(
            f"unknown trusted primitives {unknown}; known: {sorted(REGISTRY)}"
        )
    return names


def render_primitive_layer(allowed) -> str:
    """The fixed source for every granted primitive, in registry order."""
    return "".join(REGISTRY[name].source for name in normalize(allowed))


def granted_symbols(allowed) -> tuple[str, ...]:
    return tuple(
        symbol for name in normalize(allowed) for symbol in REGISTRY[name].symbols
    )


def catalogue(allowed) -> list[dict[str, str]]:
    """What the report says was on offer."""
    return [
        {"name": name, "symbols": list(REGISTRY[name].symbols),
         "summary": REGISTRY[name].summary}
        for name in normalize(allowed)
    ]


#: What a prompt says a grant reaches (TASK.md and the evolve system message).
GRANT_SCOPE = ("It is the only way to reach a vendor contraction or fused attention: "
               "every other contraction must be written in Triton (`tl.dot`).")


def describe_grant(allowed, where: str) -> list[str]:
    """One prompt line per granted primitive, in the order given: its symbols,
    what it does, and ``where`` it is defined."""
    return [f"`{'`, `'.join(REGISTRY[name].symbols)}`: {REGISTRY[name].summary}. "
            f"{where}; call it by name, do not define or import it." for name in allowed]


def describe_rules(where: str) -> str:
    """The prompt bullet stating what :func:`check_source` rejects in ``where``,
    read from its own tables (TASK.md section 3 and Pipeline A's prompt)."""
    torch_prefixes = sorted(".".join(("torch", *p)) for p in _DENIED_TORCH_PREFIXES)
    rules = (
        f"`torch.<name>` for: {', '.join(sorted(_TORCH_DENIED))};",
        "any `torch.*` name containing, in any case: "
        f"{', '.join(sorted(_DENIED_TORCH_FAMILIES))};",
        f"any `torch.*` convolution or recurrent entry point: {'; '.join(_DENIED_TORCH_PATTERNS)};",
        "anything under a private torch module or name (`torch._*`, at any depth), and "
        f"anything under {', '.join(torch_prefixes)};",
        f"imports of or calls into: {', '.join(sorted(_DENIED_ROOTS))};",
        f"calls to the builtins {', '.join(f'`{b}`' for b in sorted(_DENIED_BUILTINS))};",
        "methods `.<name>()` on any object, `torch` included, and these names imported "
        f"from any module, for: {', '.join(sorted(_DENIED_METHODS))};",
        "the `@` and `@=` operators, and `from torch import *` (or from any torch module);",
        "`torch` or any torch module used as a value (passed, returned, unpacked or put in a "
        "container): a torch module may only be the receiver of an attribute, as in "
        "`torch.empty_like(x)`, or bound to a plain name (`la = torch.linalg`); dtypes such as "
        "`torch.bfloat16` are values and stay allowed.",
    )
    return (f"- {where} is checked statically.\n"
            "  Rejected, whether called, read or imported, and under any name bound to it\n"
            "  (`from torch.X import Y` counts as `torch.X.Y`; after `import torch.X as Y`\n"
            "  or `Y = torch.X`, `Y.Z` counts as `torch.X.Z`):\n"
            + "\n".join(f"  - {rule}" for rule in rules))


# ── source analysis ──────────────────────────────────────────────────────────


def _dotted(node: ast.AST) -> tuple[str, ...] | None:
    """``torch.nn.functional.linear`` -> ('torch','nn','functional','linear')."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def evolve_span(source: str) -> tuple[int, int]:
    """Line numbers (1-based, inclusive) the evolve markers span."""
    lines = source.splitlines()
    start = next((i for i, l in enumerate(lines) if "EVOLVE-BLOCK-START" in l), None)
    end = next((i for i, l in enumerate(lines) if "EVOLVE-BLOCK-END" in l), None)
    if start is None or end is None:
        raise PrimitiveViolation("the artifact has no EVOLVE-BLOCK markers")
    return start + 1, end + 1


#: Dynamic access that would let a name reach a denied attribute unseen.
_DENIED_BUILTINS = frozenset({"getattr", "__import__", "eval", "exec", "vars", "globals"})


def _denied_torch_path(rest: list[str] | tuple[str, ...]) -> bool:
    """A dotted path under ``torch`` that the rules deny: by prefix, by a private
    component anywhere (``torch._inductor.select_algorithm.extern_kernels``), or
    by its leaf."""
    return bool(rest) and (any(tuple(rest[: len(p)]) == p for p in _DENIED_TORCH_PREFIXES)
                           or any(part.startswith("_") for part in rest)
                           or _denied_torch_leaf(rest[-1]))


def _denied_path(path: tuple[str, ...]) -> bool:
    """A dotted name the rules deny: under a denied root, under ``torch``, or a
    method spelling on any other receiver (``x.matmul``; ``tl.dot`` is allowed)."""
    root, *rest = path
    if root in _DENIED_ROOTS:
        return True
    if root == "torch":
        return _denied_torch_path(rest)
    return bool(rest) and path[-1] in _DENIED_METHODS


def _is_torch_module(path: tuple[str, ...]) -> bool:
    """Whether ``path`` (rooted at ``torch``) names a module, e.g. ``torch.linalg``."""
    if not path or path[0] != "torch":
        return False
    try:
        import torch

        obj = torch
        for part in path[1:]:
            obj = getattr(obj, part)
    except Exception:  # not resolvable here: not a module we can vouch for
        return False
    return inspect.ismodule(obj)


def _torch_aliases(node: ast.AST) -> dict[str, set[tuple[str, ...]]]:
    """Every name ``node`` binds to a ``torch`` path, with each path it is bound to.

    ``import torch.linalg as la``, ``from torch import conv2d``, ``T = torch`` and
    ``U = T.linalg`` give a torch namespace or routine a name that does not start
    with ``torch``; the rules judge that name's uses by the path it stands for.
    """
    aliases: dict[str, set[tuple[str, ...]]] = {}
    assigned: list[tuple[str, tuple[str, ...]]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for a in child.names:
                if a.asname and a.name.split(".")[0] == "torch":
                    aliases.setdefault(a.asname, set()).add(tuple(a.name.split(".")))
        elif isinstance(child, ast.ImportFrom) and (child.module or "").split(".")[0] == "torch":
            for a in child.names:  # ``from torch import *`` is rejected on its own
                if a.name != "*":
                    aliases.setdefault(a.asname or a.name, set()).add(
                        (*child.module.split("."), a.name))
        elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and child.value:
            value = _dotted(child.value)
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            if value is not None:
                assigned.extend((t.id, value) for t in targets if isinstance(t, ast.Name))
    # ``U = T`` may come before ``T = torch`` in the walk: propagate until nothing
    # grows, at most once per assignment (no chain of aliases is longer).
    for _ in assigned:
        grown = False
        for name, (head, *tail) in assigned:
            roots = {("torch",)} if head == "torch" else aliases.get(head, set())
            new = {(*root, *tail) for root in roots} - aliases.get(name, set())
            if new:
                aliases.setdefault(name, set()).update(new)
                grown = True
        if not grown:
            break
    return aliases


def _violations_in(node: ast.AST, *, granted: frozenset[str]) -> list[str]:
    """Every denied construct under ``node``, named with its line.

    The rules are checked wherever a denied name is *reached*, not only where it
    is called: ``import torch.nn.functional as fn``, ``_f = F.silu`` and
    ``getattr(torch, "matmul")`` each bind a library routine to a name the call
    rule could not recognise later, so the import, the attribute read and the
    dynamic lookup are violations themselves. A name bound to a torch path
    (:func:`_torch_aliases`) is judged by that path wherever it is used.
    """
    found: list[str] = []
    aliases = _torch_aliases(node)
    # Receivers of an attribute (``torch.nn`` inside ``torch.nn.functional.silu``,
    # ``la`` in ``la.vecdot``) are judged with their chain, not again on their own.
    receivers = {id(child.value) for child in ast.walk(node) if isinstance(child, ast.Attribute)}
    # ``la = torch.linalg``: a module bound to a plain name is followed by
    # ``_torch_aliases``, so only the other ways a module can travel are refused.
    tracked = {id(child.value) for child in ast.walk(node)
               if isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and child.value
               and all(isinstance(t, ast.Name)
                       for t in (child.targets if isinstance(child, ast.Assign) else [child.target]))}
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            found.append(f"line {child.lineno}: the '@' matrix-multiply operator")
            continue
        if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.MatMult):
            found.append(f"line {child.lineno}: the '@=' matrix-multiply operator")
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            module = getattr(child, "module", None) or ""
            from_torch = module.split(".")[0] == "torch"
            # ``from torch import conv2d`` is judged by its full path here, and
            # every later use of ``conv2d`` by the same path (``_torch_aliases``).
            paths = [module] if module else []
            paths += [f"{module}.{a.name}" if from_torch else a.name for a in child.names]
            for path in paths:
                root, *rest = path.split(".")
                if root in _DENIED_ROOTS or (root == "torch" and _denied_torch_path(rest)):
                    # ``import torch.nn.functional as fn``: the alias would call
                    # the denied namespace under a name the rules never see.
                    found.append(f"line {child.lineno}: import of {path!r}")
            if from_torch and any(a.name == "*" for a in child.names):
                # Binds every torch name, denied ones included, to a bare name.
                found.append(f"line {child.lineno}: from {module} import *")
            if module and not from_torch:
                # ``from operator import matmul``: a method spelling as a bare name.
                found.extend(f"line {child.lineno}: from {module} import {a.name}"
                             for a in child.names if a.name in _DENIED_METHODS)
            continue
        if isinstance(child, ast.Attribute):
            if id(child) in receivers:
                continue
            path = _dotted(child)
            if path is None:
                # An attribute of a non-name receiver, e.g. ``f(x).matmul``.
                if child.attr in _DENIED_METHODS:
                    found.append(f"line {child.lineno}: .{child.attr}")
                continue
            # As written, and as every torch path its receiver stands for
            # (``la.vecdot`` after ``import torch.linalg as la``).
            spelled = ".".join(path)
            meanings = (path, *((*root, *path[1:]) for root in aliases.get(path[0], ())))
            full = next((".".join(p) for p in meanings if _denied_path(p)), None)
            if full is not None:
                found.append(f"line {child.lineno}: {spelled}"
                             + (f" ({full})" if full != spelled else ""))
            elif (isinstance(child.ctx, ast.Load) and id(child) not in tracked
                  and any(_is_torch_module(p) for p in meanings)):
                # ``[torch.linalg]``: a torch module passed on as a value could be
                # read later under a name the rules never connect to it.
                found.append(f"line {child.lineno}: {spelled} used as a value")
            continue
        if (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                and id(child) not in receivers):
            # A bare alias: ``conv2d(x, w)`` after ``from torch import conv2d``.
            full = next((p for p in aliases.get(child.id, ()) if _denied_path(p)), None)
            if full is not None:
                found.append(f"line {child.lineno}: {child.id} ({'.'.join(full)})")
            elif id(child) not in tracked and (
                    child.id == "torch" or any(_is_torch_module(p) for p in aliases.get(child.id, ()))):
                # ``t=torch`` as a default, ``(torch,)``, ``f(torch)``, ``T, _ = torch, 1``:
                # a torch module may only be the receiver of an attribute, never a value.
                found.append(f"line {child.lineno}: {child.id} used as a value")
            continue
        if not isinstance(child, ast.Call):
            continue
        path = _dotted(child.func)
        if path is None or len(path) != 1:
            continue  # dotted callees were judged as attributes above
        leaf = path[0]
        if leaf in _DENIED_BUILTINS:
            found.append(f"line {child.lineno}: {leaf}()")
        elif leaf in _DENIED_ROOTS:
            found.append(f"line {child.lineno}: {leaf}()")
    return found


def _primitive_definitions(tree: ast.Module, allowed) -> dict[str, ast.FunctionDef]:
    wanted = set(granted_symbols(allowed))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    }


def check_source(source: str, *, allowed=()) -> dict[str, Any]:
    """Structural check of one artifact's generated pair math.

    Raises :class:`PrimitiveViolation` on anything the run did not grant, and
    otherwise returns what it used, so a report can state it rather than assume.
    """
    allowed = normalize(allowed)
    tree = ast.parse(source)
    start, end = evolve_span(source)
    granted = frozenset(granted_symbols(allowed))

    # 1. The trusted implementations are ours, fixed, and outside the block.
    definitions = _primitive_definitions(tree, allowed)
    for name in allowed:
        spec = REGISTRY[name]
        for symbol in spec.symbols:
            node = definitions.get(symbol)
            if node is None:
                raise PrimitiveViolation(
                    f"{name} was granted but the artifact does not define "
                    f"{symbol!r} at module level"
                )
            if start <= node.lineno <= end:
                raise PrimitiveViolation(
                    f"{symbol!r} is defined inside the EVOLVE-BLOCK at line "
                    f"{node.lineno}; a trusted primitive must be fixed"
                )
        if spec.source not in source:
            raise PrimitiveViolation(
                f"the artifact's {name} implementation is not the trusted one; "
                "it must be rendered verbatim from evograd.pipelines.shared."
                "primitives"
            )

    # 2. A primitive symbol may not be defined at all when it was not granted:
    #    a look-alike with the same name would read as trusted in every report.
    for spec in REGISTRY.values():
        if spec.name in allowed:
            continue
        for symbol in spec.symbols:
            shadow = next((n for n in tree.body
                           if isinstance(n, ast.FunctionDef) and n.name == symbol), None)
            if shadow is not None:
                raise PrimitiveViolation(
                    f"{symbol!r} is defined at line {shadow.lineno} but the "
                    f"{spec.name!r} capability was not granted"
                )

    # 3. The evolvable region itself.
    region = ast.Module(
        body=[n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
              and start <= n.lineno <= end],
        type_ignores=[],
    )
    # Statements at module level inside the block belong to the region too.
    region.body.extend(
        n for n in tree.body
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and start <= getattr(n, "lineno", 0) <= end
    )
    violations = _violations_in(region, granted=granted)
    if violations:
        raise PrimitiveViolation(
            "generated pair math used forbidden operations: "
            + "; ".join(sorted(set(violations))[:8])
        )

    # 4. Ungranted calls to a trusted symbol name, anywhere.
    used = {symbol: 0 for spec in REGISTRY.values() for symbol in spec.symbols}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _dotted(node.func)
        if path is None or len(path) != 1:
            continue
        symbol = path[0]
        if symbol not in used:
            continue
        if symbol not in granted:
            raise PrimitiveViolation(
                f"line {node.lineno}: {symbol}() is a trusted primitive that "
                "this run did not grant"
            )
        used[symbol] += 1

    return {
        "allowed_primitives": list(allowed),
        "primitive_call_sites": {k: v for k, v in used.items() if v},
        "uses_trusted_primitives": any(used.values()),
        "evolve_block_lines": [start, end],
    }
