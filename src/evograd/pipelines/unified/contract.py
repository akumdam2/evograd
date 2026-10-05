"""Keep a unified seed's library providers and public ABI fixed during evolution."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

CONTRACT_FILE = "unified_contract.json"
CONTRACT_ENV = "EVOGRAD_UNIFIED_CONTRACT_PATH"


def _parts(source: str):
    lines = source.splitlines()
    starts = [i + 1 for i, line in enumerate(lines) if line.strip() == "# EVOLVE-BLOCK-START"]
    ends = [i + 1 for i, line in enumerate(lines) if line.strip() == "# EVOLVE-BLOCK-END"]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("unified artifacts require exactly one ordered EVOLVE-BLOCK")
    start, end = starts[0], ends[0]
    fixed, editable = [], []
    for node in ast.parse(source).body:
        if start < node.lineno < end:
            if node.end_lineno >= end:
                raise ValueError("a definition crosses the unified evolution boundary")
            editable.append(node)
        else:
            fixed.append(node)
    return fixed, ast.Module(body=editable, type_ignores=[])


def _literal(nodes, name):
    matches = [node.value for node in nodes if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"unified artifact requires one fixed {name} declaration")
    return ast.literal_eval(matches[0])


def _fingerprint(nodes) -> str:
    tree = ast.Module(body=nodes, type_ignores=[])
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _calls(tree, names):
    return Counter(node.func.id for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in names)


def build_contract(source: str) -> dict:
    """Pin fixed code and the external call sites of an accepted seed.

    This manifest is held by the evaluator, outside the evolving candidate.
    AST hashes allow formatting changes without allowing provider/ABI changes.
    """
    fixed, editable = _parts(source)
    regions = _literal(fixed, "UNIFIED_EXTERNAL_REGIONS")
    wrappers = [region["wrapper"] for region in regions]
    if len(wrappers) != len(set(wrappers)):
        raise ValueError("external regions must have distinct wrapper names")
    definitions = {n.name for n in fixed if isinstance(n, ast.FunctionDef)}
    if not set(wrappers) <= definitions:
        raise ValueError("every external region needs a fixed wrapper")
    calls = _calls(editable, wrappers)
    if any(calls[name] != 1 for name in wrappers):
        raise ValueError("the seed must invoke each preserved external region once")
    return {"version": 1, "fixed_ast_sha256": _fingerprint(fixed),
            "external_regions": list(regions), "external_call_sites": dict(calls)}


def validate_source(source: str, contract: dict) -> None:
    """Reject changes to providers/ABI or removal/duplication of library calls."""
    fixed, editable = _parts(source)
    if contract.get("version") != 1 or _fingerprint(fixed) != contract["fixed_ast_sha256"]:
        raise ValueError("unified pipeline fixed library/deployment code was changed")
    expected = contract["external_call_sites"]
    if dict(_calls(editable, expected)) != expected:
        raise ValueError("unified pipeline preserved external call sites were changed")
    # A local definition or assignment must not redirect a protected call.
    for node in ast.walk(editable):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in expected:
                raise ValueError(f"preserved external wrapper was redefined: {node.name}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in expected:
            raise ValueError(f"preserved external wrapper was reassigned: {node.id}")


def load_contract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_contract_path(seed: Path) -> Path | None:
    """Find a unified seed's evaluator-owned contract without importing its code."""
    source = seed.read_text(encoding="utf-8")
    is_unified = any(isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "UNIFIED_EXTERNAL_REGIONS"
                          for t in node.targets)
                  for node in ast.parse(source).body)
    if not is_unified:
        return None
    path = seed.parent / CONTRACT_FILE
    contract = load_contract(path)
    validate_source(source, contract)
    return path
