"""Names for harvested configurations: one distinct role per configuration of a task.

A snapshot groups a task's observed calls into configurations (shape, dtype,
layout, input adapter, static parameters). Each configuration becomes one
benchmark case, named ``task/<component>``, so the component must differ
between a task's configurations. The language-model harvests name roles by
hand (``mlp_up``, ``attn_kv_proj``); the generated and vision harvests take
the module path a configuration was first observed at, without its last
numeric index (the repeated layer), and number any remaining collision in
order of first observation. Grouping never depends on the name.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def role_of(path: str) -> str:
    """``encoder.stages.2.layers.0.pwconv1.weight`` -> ``encoder.stages.2.layers.pwconv1``."""
    parts = [p for p in path.split("#", 1)[0].split(".") if p and p not in ("weight", "bias")]
    numeric = [i for i, p in enumerate(parts) if p.isdigit()]
    if numeric:
        del parts[numeric[-1]]
    return ".".join(parts) or "root"


def group_configurations(records: Iterable[dict[str, Any]], level: int,
                         digest: Callable[[Any], str]) -> dict[str, list[dict[str, Any]]]:
    """``{task: [configuration, ...]}`` for one level, each with a distinct component.

    A configuration is identified by ``dims``, ``dtype``, ``inputs``,
    ``strides``, ``input_adapter`` and the static parameters
    (``provenance["free"]``); calls that agree on all of them are one
    configuration, whatever module made them.
    """
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        if row["level"] != level:
            continue
        provenance = dict(row.get("provenance") or {})
        identity = {k: row[k] for k in ("dims", "dtype", "inputs", "strides", "input_adapter")}
        identity["free"] = dict(provenance.get("free") or {})
        key = digest(identity)
        entry = groups.setdefault(row["task"], {})
        if key not in entry:
            entry[key] = {**{k: row[k] for k in ("dims", "dtype", "inputs", "strides",
                                                 "input_adapter")},
                          "provenance": {**provenance, "free": identity["free"]},
                          "config_id": key, "frequency": 0, "module_paths": []}
        entry[key]["frequency"] += 1
        entry[key]["module_paths"].append(row["path"])
    out: dict[str, list[dict[str, Any]]] = {}
    for task, configs in groups.items():
        seen: dict[str, int] = {}
        ordered = list(configs.values())
        for config in ordered:
            base = role_of(config["module_paths"][0])
            seen[base] = seen.get(base, 0) + 1
            config["provenance"]["component"] = base if seen[base] == 1 else f"{base}_{seen[base]}"
        out[task] = ordered
    return out
