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
from dataclasses import dataclass
from typing import Any

#: Contractions rooted at ``torch``. Denied inside the evolvable region always;
#: permitted inside a trusted primitive's own fixed body when it declares them.
_TORCH_DENIED = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "einsum", "tensordot",
    "dot", "inner", "outer", "chain_matmul", "matrix_power", "kron",
    "scaled_dot_product_attention", "compile", "rms_norm", "layer_norm",
    "group_norm", "batch_norm",
})

#: Namespaces no generated pair math may reach into at all.
_DENIED_ROOTS = frozenset({"F", "functional", "transformers", "evograd", "liger_kernel"})

#: Denied whatever the receiver is. ``dot`` is deliberately absent: ``tl.dot``
#: is the Triton language's own contraction and is the evolved kernel's
#: arithmetic, not an escape from it.
_DENIED_METHODS = frozenset({
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "linear", "backward",
})

#: Attribute paths under ``torch`` that are denied by prefix rather than leaf.
_DENIED_TORCH_PREFIXES = (("nn", "functional"), ("autograd",), ("_dynamo",),
                          ("jit",), ("fx",))


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


REGISTRY: dict[str, TrustedPrimitive] = {
    "vendor_gemm": TrustedPrimitive(
        name="vendor_gemm",
        symbols=("vendor_gemm",),
        source=VENDOR_GEMM_SOURCE,
        uses=("mm",),
        summary="2-D row-major GEMM through cuBLAS (torch.mm), no autograd",
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


def _violations_in(node: ast.AST, *, granted: frozenset[str]) -> list[str]:
    """Every denied construct under ``node``, named with its line."""
    found: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            found.append(f"line {child.lineno}: the '@' matrix-multiply operator")
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            name = getattr(child, "module", None) or ""
            names = [a.name for a in child.names]
            for candidate in [name, *names]:
                root = candidate.split(".")[0]
                if root in _DENIED_ROOTS:
                    found.append(f"line {child.lineno}: import of {candidate!r}")
            if name.split(".")[0] == "torch":
                # ``from torch import matmul`` would otherwise arrive as a bare
                # call with no ``torch.`` prefix left to recognise.
                for alias in child.names:
                    if alias.name in _TORCH_DENIED:
                        found.append(
                            f"line {child.lineno}: from {name} import {alias.name}"
                        )
            continue
        if not isinstance(child, ast.Call):
            continue
        path = _dotted(child.func)
        if path is None:
            # A call on a non-name receiver, e.g. ``f(x).matmul(y)``.
            if isinstance(child.func, ast.Attribute) and child.func.attr in _DENIED_METHODS:
                found.append(f"line {child.lineno}: .{child.func.attr}()")
            continue
        root, *rest = path
        leaf = path[-1]
        if root in _DENIED_ROOTS:
            found.append(f"line {child.lineno}: {'.'.join(path)}")
        elif root == "torch":
            if any(tuple(rest[: len(p)]) == p for p in _DENIED_TORCH_PREFIXES):
                found.append(f"line {child.lineno}: {'.'.join(path)}")
            elif leaf in _TORCH_DENIED:
                found.append(f"line {child.lineno}: {'.'.join(path)}")
        elif len(path) == 1:
            if leaf in REGISTRY or leaf in granted:
                continue
        elif leaf in _DENIED_METHODS:
            # ``x.matmul(y)`` on any receiver that is not the triton language.
            found.append(f"line {child.lineno}: {'.'.join(path)}")
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
