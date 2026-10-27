"""Assemble an evolvable autograd-pair seed from two Inductor-generated modules.

Inductor emits each graph as a standalone module: a shared import header, its
kernels registered through ``async_compile``, and a ``Runner.call`` wrapper that
allocates buffers and launches them. The forward and backward modules cannot
simply be concatenated -- both define ``call`` and their kernel symbols collide
-- so each is split into (header, kernels, call body), its generated symbols are
prefixed, and the pieces are re-emitted as one flat module.

One seed covers one dtype. Inductor specializes on dtype, and the specialized
kernels are not small variations of each other -- they differ in vector width,
inserted converts, and accumulator types -- so merging them would carry several
near-duplicate copies of the same algorithm through the search. Shapes are a
different matter: dynamic-shape capture makes a single kernel size-generic.

Everything the search may change -- kernels, buffer allocation, launch order,
and the saved-tensor contract -- lands inside the EVOLVE-BLOCK. Inductor's
library calls (extern GEMM, fused attention) are routed through the granted
trusted primitives, and the generated deployment layer follows the block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evograd.opdecl.activity import Inactive, OpDecl, format_default
from evograd.pipelines.d_inductor.capture import CapturedPair
from evograd.pipelines.shared.artifact import render_deployment_layer

_HEADER_END = re.compile(r"^async_compile\s*=\s*AsyncCompile\(\)\s*$")
_WAIT = re.compile(r"^async_compile\.wait\(globals\(\)\)\s*$")
_CALL_DEF = re.compile(r"^\s+def call\(self, args\):\s*$")
_RUNNER_INSTANCE = re.compile(r"^runner\s*=\s*Runner\(")
_KERNEL_ASSIGN = re.compile(r"^(\w+)\s*=\s*async_compile\.", re.MULTILINE)
#: Module-level handles Inductor binds for its runtime: ATen and extern-kernel
#: namespaces and the symmetric-memory allocator. Kept, they would let evolved
#: code reach library or private ATen entry points under names the static rules
#: do not recognise; the seed reaches GEMM and attention only through the
#: granted primitives instead.
_LIBRARY_ALIAS = re.compile(
    r"^(?:\w+ = torch\.(?:ops|_C\._distributed_c10d)\..*"
    r"|from torch\._inductor\.select_algorithm import extern_kernels)$"
)

_DTYPE_TAGS = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "float64": "f64",
}

_EXTRA_IMPORTS = """
# Available so evolved code can add plain Triton kernels alongside the
# Inductor-generated ones. Guarded because Triton is not installed on the
# CPU-only machines used to check seed assembly.
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU capture
    triton = None
    tl = None
"""


def dtype_tag(name: str) -> str:
    return _DTYPE_TAGS.get(name, re.sub(r"\W", "_", name))


@dataclass(frozen=True)
class _Split:
    header: list[str]
    kernels: list[str]
    call_body: list[str]
    symbols: tuple[str, ...]


def _split_module(source: str, side: str) -> _Split:
    lines = [ln for ln in source.splitlines() if not _LIBRARY_ALIAS.match(ln)]

    header_end = next((i for i, ln in enumerate(lines) if _HEADER_END.match(ln)), None)
    if header_end is None:
        raise RuntimeError(f"{side}: no 'async_compile = AsyncCompile()' in generated module")

    wait_at = next((i for i, ln in enumerate(lines) if _WAIT.match(ln)), None)
    if wait_at is None:
        raise RuntimeError(f"{side}: no 'async_compile.wait(globals())' in generated module")

    call_at = next((i for i, ln in enumerate(lines) if _CALL_DEF.match(ln)), None)
    if call_at is None:
        raise RuntimeError(f"{side}: no 'def call(self, args)' in generated module")

    runner_at = next(
        (i for i, ln in enumerate(lines[call_at:], call_at) if _RUNNER_INSTANCE.match(ln)),
        len(lines),
    )

    kernels = lines[header_end + 1 : wait_at]
    body = lines[call_at + 1 : runner_at]
    while body and not body[-1].strip():
        body.pop()

    if any(re.search(r"\bself\b", ln) for ln in body):
        raise RuntimeError(
            f"{side}: generated call() references 'self' (subgraph partitioning); "
            "this pipeline supports single-partition graphs only"
        )

    return _Split(
        header=lines[: header_end + 1],
        kernels=kernels,
        call_body=body,
        symbols=tuple(_KERNEL_ASSIGN.findall("\n".join(kernels))),
    )


_TRITON_OPEN = re.compile(r"^(\w+)\s*=\s*async_compile\.triton\(\s*['\"](\w+)['\"]\s*,\s*'''\s*$")
_TRITON_CLOSE = re.compile(r"^'''\s*,.*\)\s*$")


def _hoist_leading_imports(body: list[str]) -> tuple[list[str], list[str]]:
    """Split a kernel body into (its leading imports, the rest).

    Every kernel Inductor emits opens with the same import preamble. Repeating
    it once per kernel is noise, so it is lifted to the top of the seed.
    """
    cut = 0
    imports = []
    for index, line in enumerate(body):
        stripped = line.strip()
        if not stripped:
            cut = index + 1
            continue
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
            cut = index + 1
            continue
        break
    return imports, body[cut:]


def _inline_triton_kernels(lines: list[str]) -> tuple[list[str], list[str], int]:
    """Lift Triton kernels out of their ``async_compile.triton`` string.

    Inductor ships each kernel as source text handed to the async compiler.
    That text is already a self-contained module -- it imports triton and the
    inductor runtime helpers, and carries its own ``@triton_heuristics.*`` plus
    ``@triton.jit`` decorators -- so splicing it in at module level binds the
    same name to the same autotuner and leaves the ``.run(...)`` launch sites
    working. What changes is that the kernel becomes real code the search can
    read and edit, rather than a quoted blob.

    Returns the rewritten lines, the imports hoisted out of the kernel bodies,
    and how many kernels stayed in string form -- C++ bodies from a CPU capture
    cannot be inlined as Python.
    """
    out: list[str] = []
    hoisted: list[str] = []
    remaining = 0
    index = 0
    while index < len(lines):
        # Inductor prefixes each kernel with an absolute path into whoever's
        # cache produced it. That is machine-local and means nothing in a seed;
        # the ATen provenance comments beside it stay, since they say which ops
        # the kernel fused.
        if lines[index].lstrip().startswith("# kernel path:"):
            index += 1
            continue

        opened = _TRITON_OPEN.match(lines[index])
        if not opened:
            if re.match(r"^\w+\s*=\s*async_compile\.", lines[index]):
                remaining += 1
            out.append(lines[index])
            index += 1
            continue

        close = next(
            (j for j in range(index + 1, len(lines)) if _TRITON_CLOSE.match(lines[j])),
            None,
        )
        if close is None:
            raise RuntimeError(
                f"unterminated async_compile.triton block for {opened.group(1)!r}"
            )
        imports, body = _hoist_leading_imports(lines[index + 1 : close])
        for line in imports:
            if line not in hoisted:
                hoisted.append(line)
        out.append("")
        out.extend(body)
        out.append("")
        index = close + 1
    return out, hoisted, remaining


_HEURISTIC_OPEN = re.compile(r"^@triton_heuristics\.(\w+)\(\s*$")
_KERNEL_DEF = re.compile(r"^def (\w+)\(")


def _split_top_level(text: str) -> list[str]:
    """Split a call's argument text on commas that are not nested or quoted."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                current.append(char)
                index += 1
                if index < len(text):
                    current.append(text[index])
                index += 1
                continue
            if text.startswith(quote, index):
                current.append(quote)
                index += len(quote)
                quote = None
                continue
        elif char in "\"'":
            quote = text[index : index + 3] if text.startswith(char * 3, index) else char
            current.append(quote)
            index += len(quote)
            continue
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def _pin_kernel_configs(
    lines: list[str], configs: dict[str, dict[str, Any]], side: str
) -> list[str]:
    """Replace autotuning decorators with ``fixed_config``, one config each.

    ``triton_heuristics.pointwise``/``.reduction`` exist to *generate* candidate
    configs from ``size_hints`` and sweep them at first call. ``fixed_config``
    takes a single decided config and skips the sweep. ``triton_meta`` and
    ``inductor_meta`` are carried over untouched -- they are not tuning inputs:
    the former is what Triton needs to compile, and the latter carries
    ``grid_type``, without which there is no launch grid.
    """
    out: list[str] = []
    index = 0
    while index < len(lines):
        opened = _HEURISTIC_OPEN.match(lines[index])
        if not opened or opened.group(1) == "fixed_config":
            out.append(lines[index])
            index += 1
            continue

        depth = 0
        close = index
        for close in range(index, len(lines)):
            depth += lines[close].count("(") - lines[close].count(")")
            if depth == 0:
                break
        else:
            raise RuntimeError(f"{side}: unterminated @triton_heuristics decorator")

        body = "\n".join(lines[index : close + 1])
        body = body[body.index("(") + 1 : body.rindex(")")]
        kept = {}
        for part in _split_top_level(body):
            name, separator, value = part.partition("=")
            if separator and name.strip() in ("filename", "triton_meta", "inductor_meta"):
                kept[name.strip()] = value.strip()

        kernel = next(
            (
                _KERNEL_DEF.match(lines[j]).group(1)
                for j in range(close + 1, min(close + 6, len(lines)))
                if _KERNEL_DEF.match(lines[j])
            ),
            None,
        )
        config = configs.get(kernel)
        if config is None:
            raise RuntimeError(
                f"{side}: no autotuner config recorded for {kernel!r}; cannot pin "
                "it. Re-capture with autotuning enabled, or file this as a bug."
            )

        out.append("@triton_heuristics.fixed_config(")
        out.append(f"    config={config!r},")
        for name in ("filename", "triton_meta", "inductor_meta"):
            if name in kept:
                out.append(f"    {name}={kept[name]},")
        out.append(")")
        index = close + 1
    return out


def _prefix_symbols(text: str, symbols: tuple[str, ...], prefix: str) -> str:
    """Rename generated kernel symbols so both halves can share a module.

    Applied uniformly, including inside the kernel source strings: Inductor
    passes the kernel name both as a Python identifier and as a lookup key into
    that string, so renaming both keeps them consistent.
    """
    for symbol in sorted(symbols, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(symbol)}\b", f"{prefix}{symbol}", text)
    return text


_ARGS_UNPACK = re.compile(r"^\s*([A-Za-z_][\w, ]*?),?\s*=\s*args\s*$")


def _named_parameters(body: list[str], side: str) -> tuple[list[str], list[str]]:
    """Replace Inductor's ``call(args)`` list protocol with named parameters.

    The generated wrapper opens by unpacking a list and clearing it, so callers
    have to pack a list and the signature says nothing about what goes in it.
    Lifting the names into the signature reads the way a hand-written launcher
    does. ``args.clear()`` exists to drop the caller's references early; with
    named parameters the local names are the only ones, and the ``del`` lines
    Inductor already emits still free them at the same points.
    """
    unpack_at = next(
        (i for i, line in enumerate(body) if _ARGS_UNPACK.match(line)), None
    )
    if unpack_at is None:
        raise RuntimeError(f"{side}: generated call() does not unpack 'args'")
    names = [
        n.strip()
        for n in _ARGS_UNPACK.match(body[unpack_at]).group(1).split(",")
        if n.strip()
    ]
    rest = [
        line
        for i, line in enumerate(body)
        if i != unpack_at and line.strip() != "args.clear()"
    ]
    return names, rest


_EXTERN_MM = re.compile(r"^(\s*)extern_kernels\.mm\((.*), out=(\w+)\)$")
#: Inductor's fused-attention externs: the granted primitive that runs the same
#: kernel, how many leading tensors it takes, and where dropout_p, is_causal and
#: attn_bias sit among the captured call's positional arguments.
_ATTENTION = {
    "_scaled_dot_product_cudnn_attention": ("vendor_sdpa_forward", 3, 5, 6, 3),
    "_scaled_dot_product_flash_attention": ("vendor_sdpa_forward", 3, 3, 4, None),
    "_scaled_dot_product_cudnn_attention_backward": ("vendor_sdpa_backward", 6, 13, 14, 8),
    "_scaled_dot_product_flash_attention_backward": ("vendor_sdpa_backward", 6, 10, 11, None),
}
_ATTENTION_CALL = re.compile(
    rf"^(\s*)(\w+) = torch\.ops\.aten\.({'|'.join(_ATTENTION)})\.default\((.*)\)$"
)
_ASSERT_ON = re.compile(r"^\s*assert_(?:size_stride|alignment)\((\w+),")
#: A library entry point left in launch code once the granted calls are routed.
_LIBRARY_CALL = re.compile(r"\b(?:extern_kernels|aten|inductor_ops|torch\.ops)\.")


def _require(allowed: tuple[str, ...], primitive: str, side: str, line: str) -> None:
    if primitive not in allowed:
        raise RuntimeError(
            f"{side}: {line.strip()!r} is a library call; routing it needs the "
            f"{primitive} grant (--allow-primitive {primitive})"
        )


def _legalize_launch_code(body: list[str], allowed: tuple[str, ...], side: str) -> list[str]:
    """Make a generated call body legal inside the evolve block.

    The private ``torch.cuda._DeviceGuard`` becomes the public
    ``torch.cuda.device``. Library calls run the same computation through the
    granted trusted primitives: ``extern_kernels.mm(a, b, out=buf)`` becomes
    ``buf = vendor_gemm(a, b)`` -- cuBLAS through ``torch.mm`` -- and the line
    that planned ``buf`` (a fresh allocation, or a dead buffer reused) goes, as
    nothing writes into it any more. Fused attention becomes
    ``vendor_sdpa_forward``/``vendor_sdpa_backward`` with the save-set kept:
    the primitive returns ``(out, logsumexp)`` and runs flash under
    deterministic algorithms, whose logsumexp layout is not cuDNN's, so the
    asserts on that backend state are dropped and the rng state (no dropout)
    is ``None``. Any other library call is an error.
    """
    out: list[str] = []
    state: set[str] = set()
    result = None  # the forward attention call's result tuple
    for line in body:
        line = line.replace("torch.cuda._DeviceGuard(", "torch.cuda.device(")
        if mm := _EXTERN_MM.match(line):
            indent, args, buf = mm.groups()
            _require(allowed, "vendor_gemm", side, line)
            at = max((i for i, ln in enumerate(out)
                      if re.search(rf"\b{buf}\b", ln) and not ln.lstrip().startswith("#")),
                     default=None)
            planned = at is not None and re.match(
                rf"^(\s*){buf} = (?:empty_strided_\w+\(.*\)|.*; (del \w+)  # reuse)$", out[at]
            )
            if not planned:
                raise RuntimeError(
                    f"{side}: extern mm writes into {buf}, which aliases a live "
                    "buffer; vendor_gemm returns a fresh tensor"
                )
            if planned.group(2):
                out[at] = planned.group(1) + planned.group(2)
            else:
                del out[at]
            a, b = (part.strip() for part in _split_top_level(args))
            out.append(f"{indent}{buf} = vendor_gemm({a}, {b})")
            continue
        if call := _ATTENTION_CALL.match(line):
            indent, name, kernel, text = call.groups()
            primitive, tensors, dropout_at, causal_at, bias_at = _ATTENTION[kernel]
            _require(allowed, "vendor_sdpa", side, line)
            parts = [part.strip() for part in _split_top_level(text)]
            args = [part for part in parts if not re.match(r"^\w+=", part)]
            scale = next((p.split("=", 1)[1] for p in parts if p.startswith("scale=")), "None")
            dropout = args[dropout_at] if len(args) > dropout_at else "0.0"
            if float(dropout) != 0.0 or (bias_at is not None and args[bias_at] != "None"):
                raise RuntimeError(f"{side}: aten.{kernel} with dropout or a bias has no granted primitive")
            causal = args[causal_at] if len(args) > causal_at else "False"
            out.append(f"{indent}{name} = {primitive}("
                       f"{', '.join(args[:tensors])}, is_causal={causal}, scale={scale})")
            if primitive == "vendor_sdpa_forward":
                result = name
            else:  # logsumexp and the rng state, handed back from the forward
                state.update(args[5:])
            continue
        if result and (item := re.match(rf"^(\s*)(\w+) = {result}\[(\d+)\]$", line)):
            indent, name, index = item.groups()
            if index != "0":
                state.add(name)
            if int(index) > 1:
                line = f"{indent}{name} = None  # backend rng/sequence state; unused without dropout"
        if _LIBRARY_CALL.search(re.sub(r"'[^']*'|#.*", "", line)):
            raise RuntimeError(f"{side}: no granted primitive runs {line.strip()!r}")
        out.append(line)
    return [ln for ln in out if not ((m := _ASSERT_ON.match(ln)) and m.group(1) in state)]


def _dedent(lines: list[str], amount: int) -> list[str]:
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif line[:amount].strip():
            raise RuntimeError(f"cannot dedent line by {amount}: {line!r}")
        else:
            out.append(line[amount:])
    return out


def _merge_headers(first: list[str], second: list[str]) -> list[str]:
    merged = list(first)
    seen = set(merged)
    insert_at = len(merged) - 1  # keep 'async_compile = AsyncCompile()' last
    for line in second:
        if line not in seen:
            merged.insert(insert_at, line)
            seen.add(line)
            insert_at += 1
    return merged


def _tensor_args(op: OpDecl) -> list[str]:
    return [a.name for a in op.args if getattr(a, "shape", None) is not None]


def _scalar_inactive(op: OpDecl) -> list[Inactive]:
    return list(op.scalar_inactive_args())


def _render_wrapper(op: OpDecl, dtype: str, captured: CapturedPair) -> str:
    tensor_names = _tensor_args(op)
    scalars = _scalar_inactive(op)
    guard_on = tensor_names[0]

    scalar_sig = "".join(
        f", {c.name}" + (f"={format_default(c.default)}" if c.default is not None else "")
        for c in scalars
    )
    forward_sig = ", ".join(tensor_names) + scalar_sig
    # Inputs keep the strides they were captured with: the generated code
    # asserts them, and a declared non-contiguous input (attention's q, k, v)
    # is part of the case, not something to normalize away.
    fwd_args = ", ".join(tensor_names)
    # One upstream gradient per declared output; several arrive as one tuple.
    upstream = op.upstream_grad_parameter
    tangents = upstream if op.is_multi_output else f"({upstream},)"
    n_out = len(op.outputs)
    outputs = f"_out[:{n_out}]" if op.is_multi_output else "_out[0]"
    backward_sig = f"{upstream}, saved_tensors{scalar_sig}"
    unused = "".join(f"    _ = {c.name}\n" for c in scalars)
    picked = ", ".join(f"_grads[{i}]" for i in captured.grad_indices)

    baked_note = ""
    if captured.baked_scalars:
        pairs = ", ".join(f"{k}={v!r}" for k, v in captured.baked_scalars.items())
        baked_note = (
            f"    # Baked into the kernels at trace time ({pairs}); the parameter is\n"
            f"    # accepted for API compatibility. Changing it requires re-capturing,\n"
            f"    # or rewriting the kernels to take it as a runtime argument.\n"
        )

    return f'''

_SEED_DTYPE = torch.{dtype}

# How to build the backward's argument list: ("s", i) takes saved[i], ("t", i)
# takes tangent i. AOTAutograd orders the backward's placeholders with symbolic
# sizes first while the forward returns them in graph order, so the save-set is
# a permutation apart rather than a pass-through.
_BWD_ARG_SPEC = {captured.backward_arg_spec}


def _check_dtype(tensor):
    if tensor.dtype is not _SEED_DTYPE:
        raise NotImplementedError(
            f"{op.name}: this seed is a {{_SEED_DTYPE}} specialist, got {{tensor.dtype}}"
        )


def _build_backward_args(spec, saved, tangents):
    """Assemble the backward's argument list.

    Saved values pass through untouched: they came from this file's forward, so
    they already carry the exact strides the generated code asserts. Forcing
    them contiguous would break saved transposed views. Tangents arrive from the
    caller, so those are normalized.
    """
    return [saved[i] if kind == "s" else tangents[i].contiguous() for kind, i in spec]


def {op.forward_fn_name}({forward_sig}):
{baked_note}{unused}    _check_dtype({guard_on})
    _out = _fwd_call({fwd_args})
    return {outputs}, tuple(_out[{n_out}:])


def {op.backward_fn_name}({backward_sig}):
{unused}    _grads = _bwd_call(
        *_build_backward_args(_BWD_ARG_SPEC, saved_tensors, {tangents})
    )
    return ({picked},)
'''


def _render_banner(
    op: OpDecl, dtype: str, captured: CapturedPair, autotune: bool = True
) -> str:
    entries = "\n".join(f"    {s.describe()}" for s in captured.saved)
    shapes = (
        "generic: captured with dynamic shapes, so sizes arrive as runtime arguments"
        if captured.dynamic
        else "static: captured at the case shape, which the generated code asserts"
    )
    return f'''"""Pipeline D seed for `{op.name}` ({dtype}) -- Inductor's kernels and launch code.

Produced by AOTAutograd + min_cut_rematerialization_partition + Inductor. The
save-set below is the partitioner's choice, not a policy of this pipeline;
Pipeline B's inputs-only contract is the same decision at memory budget 0.
Inductor's library calls (extern GEMM, fused attention) run through the granted
trusted primitives defined after the evolve block.

Partitioner save-set -- {len(captured.saved)} entries,
{captured.saved_bytes_at_capture:,} bytes at the capture shape:
{entries}
(as captured; where a library call runs through a granted primitive the seed
saves what that primitive returns, e.g. no rng state for vendor_sdpa)

This is a {dtype} specialist, because Inductor specializes on dtype. Shapes are
{shapes}.

Launch configs are {"autotuned at first call" if autotune else "pinned to the config autotuning chose at capture"}.

The saved-tensor contract is a private agreement between the two functions in
this file, so the search may change it -- the forward's returns and the
backward's unpacking need only stay consistent. Correctness is checked against
`torch.autograd.grad` on the declared forward reference.

Captured on {captured.device}.
"""
'''


def generate_inductor_seed(
    op: OpDecl,
    dtype: str,
    captured: CapturedPair,
    autotune: bool = True,
    allowed_primitives: tuple[str, ...] = (),
) -> str:
    """Merge one dtype's forward and backward captures into one evolvable seed.

    With ``autotune=False`` each kernel's config is pinned to the one the
    autotuner chose during capture, so the seed keeps that performance without
    sweeping at first call -- deterministic, and the block size becomes an
    explicit constant the search can read and change.

    ``allowed_primitives`` is the run's grant: Inductor's library calls are
    routed through it, and the deployment layer records it."""
    fwd = _split_module(captured.forward_source, "forward")
    bwd = _split_module(captured.backward_source, "backward")

    fwd_lines, fwd_imports, fwd_left = _inline_triton_kernels(fwd.kernels)
    bwd_lines, bwd_imports, bwd_left = _inline_triton_kernels(bwd.kernels)
    if not autotune:
        fwd_lines = _pin_kernel_configs(
            fwd_lines, captured.kernel_configs.get("forward", {}), "forward"
        )
        bwd_lines = _pin_kernel_configs(
            bwd_lines, captured.kernel_configs.get("backward", {}), "backward"
        )
    needs_async_compile = bool(fwd_left + bwd_left)
    kernel_imports = fwd_imports + [i for i in bwd_imports if i not in fwd_imports]

    fwd_kernels = _prefix_symbols("\n".join(fwd_lines), fwd.symbols, "fwd_")
    bwd_kernels = _prefix_symbols("\n".join(bwd_lines), bwd.symbols, "bwd_")
    fwd_params, fwd_rest = _named_parameters(fwd.call_body, "forward")
    bwd_params, bwd_rest = _named_parameters(bwd.call_body, "backward")
    fwd_rest = _legalize_launch_code(fwd_rest, allowed_primitives, "forward")
    bwd_rest = _legalize_launch_code(bwd_rest, allowed_primitives, "backward")
    fwd_body = _prefix_symbols("\n".join(_dedent(fwd_rest, 4)), fwd.symbols, "fwd_")
    bwd_body = _prefix_symbols("\n".join(_dedent(bwd_rest, 4)), bwd.symbols, "bwd_")

    header = _merge_headers(fwd.header, bwd.header)
    if not needs_async_compile:
        header = [ln for ln in header if "AsyncCompile" not in ln]
    header += kernel_imports

    parts = [
        _render_banner(op, dtype, captured, autotune),
        "\n".join(header),
        # Only needed when no kernel brought a real `import triton` along.
        "" if kernel_imports else _EXTRA_IMPORTS,
        "",
        "# EVOLVE-BLOCK-START",
        "",
        "# ==== forward kernels " + "=" * 52,
        fwd_kernels,
        "",
        "# ==== backward kernels " + "=" * 51,
        bwd_kernels,
        "",
    ]
    if needs_async_compile:
        parts += ["async_compile.wait(globals())", "del async_compile", ""]
    parts += [
        "",
        f"def _fwd_call({', '.join(fwd_params)}):",
        fwd_body,
        "",
        "",
        f"def _bwd_call({', '.join(bwd_params)}):",
        bwd_body,
        "",
        _render_wrapper(op, dtype, captured).rstrip(),
        "",
        "# EVOLVE-BLOCK-END",
        render_deployment_layer(op, allowed_primitives=allowed_primitives),
    ]
    return "\n".join(parts)
