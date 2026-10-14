"""The model's own invocation of one Level-2 site.

Preferably from the captured layer artifact of the site's block kind (the
Level-3 step's own weights, arguments and upstream gradient); the directory is
``layer_dir`` or the ``EVOGRAD_QWEN3_5_LAYER_DIR`` environment variable. Without
one, from a config-derived block (T-init weights, seeded inputs, B1 T2048
bfloat16). Either way each site's ``capture`` runs the block forward and
backward with hooks on the site's boundary and returns what crossed it.
"""

from __future__ import annotations

import os
from typing import Any

BATCH, SEQ, DTYPE, SEED = 1, 2048, "bfloat16", 0
LAYER_DIR_ENV = "EVOGRAD_QWEN3_5_LAYER_DIR"


def captured_invocation(op, device: str, *, layer_dir=None, arch=None, layers=None,
                        batch: int = BATCH, seq: int = SEQ
                        ) -> tuple[dict[str, Any], Any, str] | None:
    """``(values, workload, label)`` for a Level-2 site's task; ``None`` otherwise.

    ``arch``/``layers``/``batch``/``seq`` replace the published config-derived
    block for a test; the workload is then that block's declared dims.
    """
    from evograd.benchmark.topdown.qwen3_5_4b.architecture import (
        QWEN3_5_4B, REPRESENTATIVE_LAYERS, config_record)
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import capture_tools, manifest
    from evograd.opdecl import Workload, bind_shape

    sites = {task: site for site, task in manifest.SITE_TASKS.items()}
    site = sites.get(op.name)
    if site is None:
        return None
    kind = manifest.kind_of_site(site)
    published = arch is None
    layer_dir = layer_dir if layer_dir is not None else os.environ.get(LAYER_DIR_ENV)
    artifact = capture_tools.artifact_for(kind, layer_dir) if published else None
    if artifact is not None:
        source = capture_tools.captured_source(artifact, device=device)
    else:
        index = (layers or REPRESENTATIVE_LAYERS)[kind]
        source = capture_tools.config_source(index, arch=QWEN3_5_4B if published else arch,
                                             batch=batch, seq=seq, dtype=DTYPE, seed=SEED,
                                             device=device)
    call = manifest.captures()[site](source)
    values = {**call.inputs, **call.upstream}
    missing = [a.name for a in op.args if a.name not in values and not hasattr(a, "default")]
    missing += [g for g in op.upstream_grad_names if g not in values]
    if missing:
        raise KeyError(f"{site}: the captured call has no {missing}")
    if published:
        workload = op.benchmark_workloads(suite="qwen3_5_4b_observed")[0]
    else:
        dims = manifest.dims_for(site, config_record(arch, name="derived"), batch=batch, seq=seq)
        workload = Workload(dims=dims, dtype=DTYPE)
    for arg in op.args:  # the captured tensors must be the declared observed case
        shape = getattr(arg, "shape", None)
        value = values.get(arg.name)
        if shape not in (None, "unspecified") and hasattr(value, "shape"):
            if tuple(value.shape) != tuple(bind_shape(shape, workload.dims)):
                raise ValueError(f"{site}: captured {arg.name} {tuple(value.shape)} is not the "
                                 f"observed case's {bind_shape(shape, workload.dims)}")
    label = f"{source.label}, site {site}"
    del source
    return values, workload, label


__all__ = ["LAYER_DIR_ENV", "captured_invocation"]
