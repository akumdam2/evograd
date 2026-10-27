"""AtenIR graph summarization for pipeline prompts and contexts.

:func:`summarize_graph` (Pipeline A's prompts, Pipeline B's lowering context)
reads the extractor's ``args_ordered`` format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_graph(graph: dict[str, Any], max_nodes: int | None = None) -> str:
    placeholders = [node for node in graph["nodes"] if node.get("op") == "placeholder"]
    calls = [node for node in graph["nodes"] if node.get("op") == "call_function"]
    output = graph["nodes"][-1]
    selected_calls = calls if max_nodes is None else calls[:max_nodes]

    lines = ["# AtenIR Graph Summary", ""]
    lines.append(f"- placeholders: {len(placeholders)}")
    lines.append(f"- call_function nodes: {len(calls)}")
    lines.append(f"- distinct ops: {len(graph.get('distinct_ops', []))}")
    lines.append("")
    lines.append("## Placeholders")
    for node in placeholders:
        lines.append(f"- `{node['name']}`: shape={node.get('shape')} dtype={node.get('dtype')}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- `{output.get('args')}`")
    lines.append("")
    lines.append("## Call Nodes")
    for index, node in enumerate(selected_calls):
        # List operands (``cat``'s tensors, a symbolic ``view`` shape) count too.
        ao = [i for e in node.get("args_ordered") or []
              for i in (e["items"] if e.get("kind") == "shape_list" else [e])]
        tensor_inputs = [e["name"] for e in ao if e.get("kind") in ("node", "sym_node")]
        scalar_values = [
            e["value"] for e in ao if e.get("kind") == "scalar" and e.get("value") is not None
        ]
        lines.append(
            f"{index}. `{node['name']}` target=`{node.get('target')}` "
            f"inputs={tensor_inputs} scalars={scalar_values} "
            f"shape={node.get('output_shape')} dtype={node.get('output_dtype')}"
        )
    if max_nodes is not None and len(calls) > max_nodes:
        lines.append(f"... omitted {len(calls) - max_nodes} call nodes ...")
    return "\n".join(lines) + "\n"
