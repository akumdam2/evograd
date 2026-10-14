"""Where a Qwen3.5 site's real invocation comes from, and how a block is run for it.

A site's ``capture`` takes a :class:`BlockSource` -- one decoder layer, its call
arguments and its output cotangent -- runs one forward and one VJP with hooks on
the site's boundary, and returns what crossed it (:class:`DerivedCall`). Two
sources, in order of preference:

* **captured** (:func:`captured_source`): the layer artifact the Level-3 capture
  wrote (``layer14.pt`` / ``layer15.pt``) -- the model's own weights, arguments
  and upstream gradient, hash-verified;
* **config-derived** (:func:`config_source`): the layer built from the
  architecture with the model's initialization rules and seeded inputs, for
  before any capture exists.

:func:`best_source` picks the first that is available. The hook mechanics are
shared with every hybrid model (:mod:`....common.hybrid.capture`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ....common.hybrid.capture import DerivedCall, Recorder, hooks, keep, patched
from ...architecture import (CANONICAL_BATCH, CANONICAL_DTYPE, CANONICAL_SEQ, QWEN3_5_4B,
                             REPRESENTATIVE_LAYERS, layer_kind)


@dataclass
class BlockSource:
    """One layer and one call of it: everything a site capture needs."""

    layer: Any
    hidden_states: torch.Tensor
    kwargs: dict[str, Any]
    cotangent: torch.Tensor
    layer_index: int
    kind: str
    source: str  # "captured" or "config"
    label: str = ""


def run_layer(source: BlockSource) -> torch.Tensor:
    """One forward and one VJP of the layer, from a fresh leaf input."""
    layer = source.layer
    layer.zero_grad(set_to_none=True)
    hidden = source.hidden_states.detach().clone().requires_grad_(True)
    out = layer(hidden, **source.kwargs)
    out = out[0] if isinstance(out, tuple) else out
    torch.autograd.backward(out, source.cotangent)
    layer.zero_grad(set_to_none=True)
    return out


def config_source(layer_index: int, *, arch=QWEN3_5_4B, batch: int = CANONICAL_BATCH,
                  seq: int = CANONICAL_SEQ, dtype: str = CANONICAL_DTYPE, seed: int = 0,
                  device: str = "cpu") -> BlockSource:
    """A config-derived layer (T-init applied) and its seeded call."""
    from ...layer import prepare

    weights = prepare.initial_weights(arch, layer_index, dtype=dtype, seed=seed)
    layer = prepare.build_single_layer(arch, layer_index, device=device, dtype=dtype,
                                       weights=weights)
    del weights
    recipe = prepare.block_inputs(arch, layer_index, batch=batch, seq=seq, dtype=dtype,
                                  seed=seed)
    cos, sin = recipe["position_embeddings"]
    return BlockSource(
        layer=layer, hidden_states=recipe["hidden_states"].to(device),
        kwargs={"position_embeddings": (cos.to(device), sin.to(device)), "attention_mask": None},
        cotangent=recipe["cotangent"].to(device), layer_index=layer_index,
        kind=layer_kind(layer_index, arch), source="config",
        label=f"config-derived layer-{layer_index} block (B{batch} T{seq} {dtype}, seed {seed})")


def captured_source(artifact: str | Path, *, device: str = "cpu",
                    canonical: bool = True) -> BlockSource:
    """The layer as the Level-3 step held it, from a verified layer artifact.

    ``canonical=True`` also binds the artifact to the tracked snapshot
    (``load_canonical``); ``False`` still verifies both hashes.
    """
    from ...layer import prepare
    from ...layer.artifact import LayerArtifact, load_canonical, to_device

    loaded = load_canonical(Path(artifact)) if canonical else LayerArtifact.load(Path(artifact))
    payload = loaded.payload
    layer, args, kwargs, grad_output, _dtype = prepare.prepare_layer(payload, device)
    identity = payload["identity"]
    return BlockSource(
        layer=layer, hidden_states=args[0], kwargs=dict(kwargs), cotangent=grad_output,
        layer_index=int(identity["layer_index"]), kind=str(identity["layer_kind"]),
        source="captured",
        label=f"captured layer-{identity['layer_index']} invocation "
              f"({identity['workload_id']}, artifact {payload['artifact_hash'][:12]})")


def artifact_for(kind: str, results_dir: str | Path | None) -> Path | None:
    if results_dir is None:
        return None
    path = Path(results_dir) / f"layer{REPRESENTATIVE_LAYERS[kind]}.pt"
    return path if path.is_file() else None


def best_source(kind: str, *, results_dir: str | Path | None = None, device: str = "cpu",
                **config_kwargs) -> BlockSource:
    """The captured layer of this kind if its artifact is in ``results_dir``,
    else the config-derived one."""
    artifact = artifact_for(kind, results_dir)
    if artifact is not None:
        return captured_source(artifact, device=device)
    return config_source(REPRESENTATIVE_LAYERS[kind], device=device, **config_kwargs)


__all__ = ["BlockSource", "DerivedCall", "Recorder", "artifact_for", "best_source",
           "captured_source", "config_source", "hooks", "keep", "patched", "run_layer"]
