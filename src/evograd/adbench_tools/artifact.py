"""Turn an agent's module into an evograd artifact, and back.

The evaluator checks the static rules (``check_source``) only inside the
``EVOLVE-BLOCK``, because evograd's own artifacts keep fixed, trusted code --
preserved library calls, the deployment layer -- outside it. An external agent
owns no fixed code, so its *whole* module goes inside one block and every line
it wrote is checked. What follows the block is generated here from the
declaration by ``render_deployment_layer``, the function evograd's pipelines
use, with the task's granted library primitives; the agent cannot reach it.

Layout of a packaged artifact::

    <provenance docstring>
    from __future__ import ...      (hoisted from the agent's module, if any)
    import torch, triton, triton.language as tl
    # EVOLVE-BLOCK-START
    <the agent's module, verbatim>
    # EVOLVE-BLOCK-END
    <trusted primitives granted by the task>
    <declaration-generated autograd.Function, entry, module, DEPLOYMENT_ENTRY>
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evograd.adbench_tools.task import AgentTask

START = "# EVOLVE-BLOCK-START"
END = "# EVOLVE-BLOCK-END"
HEADER_MARK = "adbench-tools artifact"
#: The task pack's standalone module of granted primitives (``render_primitives_module``).
PRIMITIVES_MODULE = "adbench_primitives"

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


class PackagingError(ValueError):
    """The agent's module cannot become an artifact; the message is feedback."""


def extract_code(text: str, task: "AgentTask | None" = None) -> str:
    """The module inside an LLM response.

    Prefers a fenced block that defines both pair functions, then the longest
    Python block, then the whole text when it parses as Python.
    """
    blocks = [body for lang, body in _FENCE.findall(text)
              if lang.lower() in ("", "python", "py", "python3")]
    if task is not None:
        abi = task.spec["abi"]
        wanted = [b for b in blocks
                  if f"def {abi['forward_fn']}(" in b and f"def {abi['backward_fn']}(" in b]
        if wanted:
            return max(wanted, key=len).strip() + "\n"
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def unpackage(source: str) -> str:
    """The agent module inside a packaged artifact; other text is returned as is."""
    if HEADER_MARK not in source or source.count(START) != 1 or source.count(END) != 1:
        return source
    _, rest = source.split(START, 1)
    body, _ = rest.split(END, 1)
    return body.strip("\n") + "\n"


def _defined_names(tree: ast.Module) -> dict[str, int]:
    """Every name the module binds at any depth, with its first line."""
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names = [node.id]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [(alias.asname or alias.name).split(".")[0] for alias in node.names]
        for name in names:
            found.setdefault(name, getattr(node, "lineno", 0))
    return found


def check_module(task: "AgentTask", source: str) -> tuple[list[str], str]:
    """Validate an agent module; return ``(future_imports, body)``.

    Raises :class:`PackagingError` with a message an agent can act on.
    """
    from evograd.adbench_tools.statement import reserved_names

    if START in source or END in source:
        raise PackagingError("the module must not contain EVOLVE-BLOCK markers; "
                             "the harness adds them")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PackagingError(f"SyntaxError at line {exc.lineno}: {exc.msg}") from exc
    abi = task.spec["abi"]
    top = {node.name for node in tree.body
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [name for name in (abi["forward_fn"], abi["backward_fn"]) if name not in top]
    if missing:
        raise PackagingError(f"the module must define at top level: {missing}")

    futures: list[str] = []
    lines = source.splitlines()
    drop: set[int] = set()
    granted = set(granted_symbols(task))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.extend(alias.name for alias in node.names)
            drop.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.ImportFrom) and node.module == PRIMITIVES_MODULE:
            # The standalone stand-in for what the harness defines; the
            # artifact gets the trusted definition instead.
            names = {alias.name for alias in node.names}
            if any(alias.asname for alias in node.names) or not names <= granted:
                raise PackagingError(
                    f"line {node.lineno}: only `from {PRIMITIVES_MODULE} import "
                    f"{', '.join(sorted(granted)) or '<nothing granted>'}` is allowed")
            drop.update(range(node.lineno - 1, node.end_lineno))
    body = "\n".join(line for i, line in enumerate(lines) if i not in drop).strip("\n")
    body_tree = ast.parse(body)
    for node in ast.walk(body_tree):
        modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                   else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
        if any(m.split(".")[0] == PRIMITIVES_MODULE for m in modules):
            raise PackagingError(f"{PRIMITIVES_MODULE} may only be imported with a top-level "
                                 "`from ... import <granted names>`")
    bound = _defined_names(body_tree)
    clashes = sorted(name for name in reserved_names(task) if name in bound)
    if clashes:
        raise PackagingError(
            f"the module defines names the harness reserves: {clashes} "
            f"(first at line {bound[clashes[0]]}); remove them")
    return sorted(set(futures)), body + "\n"


def granted_symbols(task: "AgentTask") -> tuple[str, ...]:
    from evograd.pipelines.shared.primitives import granted_symbols as _granted

    return tuple(_granted(task.allowed_primitives))


def render_primitives_module(task: "AgentTask") -> str | None:
    """``adbench_primitives.py``: the granted primitives, importable standalone.

    For agents whose own harness imports the module directly (KernelAgent's
    benchmark, GEAK's workspace). Byte-identical bodies to what the packaged
    artifact defines; packaging strips the import.
    """
    from evograd.pipelines.shared.primitives import REGISTRY

    if not task.allowed_primitives:
        return None
    sources = "".join(REGISTRY[name].source for name in task.allowed_primitives)
    return (f'"""Trusted primitives granted by task {task.task_id}. Do not edit."""\n'
            f"import torch\n{sources}")


def package(task: "AgentTask", source: str, *, origin: str = "unknown") -> str:
    """The evaluable artifact for an agent module (raises :class:`PackagingError`)."""
    from evograd.pipelines.shared.artifact import render_deployment_layer

    source = unpackage(source)
    futures, body = check_module(task, source)
    header = (
        f'"""{HEADER_MARK}: task {task.task_id}, produced by {origin}.\n\n'
        "Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.\n"
        "Everything after them is generated from the declaration by evograd.\n"
        '"""\n'
    )
    future_line = f"from __future__ import {', '.join(futures)}\n" if futures else ""
    artifact = (
        f"{header}{future_line}"
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        f"{START}\n{body}{END}\n"
        f"{render_deployment_layer(task.op(), allowed_primitives=task.allowed_primitives)}"
    )
    try:
        compile(artifact, f"<adbench-tools {task.task_id}>", "exec")
    except SyntaxError as exc:  # e.g. a body that only parses at module top
        raise PackagingError(f"packaged artifact does not compile: {exc}") from exc
    return artifact


def render_skeleton(task: "AgentTask") -> str:
    """The deliverable's signatures with unimplemented bodies.

    The from-nothing starting point for an agent that needs a seed program
    (OpenEvolve), carrying the reference so the computation is in front of
    the model. It fails evaluation by construction.
    """
    abi = task.spec["abi"]
    reference = "\n".join(f"# {line}" if line else "#"
                          for line in task.reference_source.rstrip().splitlines())
    return f'''"""Unimplemented pair for task {task.task_id}; see TASK.md."""
import torch
import triton
import triton.language as tl

# The PyTorch forward reference this pair must match (do not call it):
{reference}


{abi['forward_signature']}:
    """Return {abi['forward_returns']}."""
    raise NotImplementedError("{abi['forward_fn']} is not implemented yet")


{abi['backward_signature']}:
    """Return {abi['backward_returns']}."""
    raise NotImplementedError("{abi['backward_fn']} is not implemented yet")
'''
