"""Which Transformers classes Llama-3 builds, bound to the shared builder.

Building a model from a spec, and reading back what was actually built rather
than what was asked for, is the same procedure for every decoder-only causal LM
and lives in :mod:`....common.model`. Llama's contribution is two class names.

No Hub access is needed despite Llama-3 being a gated repository: the canonical
workload constructs from the written-out configuration with random weights. A
spec whose ``weights`` pin a checkpoint loads it from a local Hugging Face cache
(:func:`pretrained_snapshot`), never from the network.
"""

from __future__ import annotations

from ....common.model import (  # noqa: F401  (re-export)
    DTYPES,
    MissingDependencyError,
    ModelClasses,
    check_effective_settings,
    effective_settings,
    make_inputs,
    rope_settings,
    training_step,
)
from ....common import model as _common
from .spec import WorkloadSpec

#: Llama has been in Transformers since 4.28, but this is the floor the rest of
#: the harness targets and the only range it has been exercised against; a lower
#: bound this repository has not run is a claim, not a guarantee.
MIN_TRANSFORMERS = (4, 51)
#: The version this workload was developed against.
TESTED_TRANSFORMERS = "5.16.1"

CLASSES = ModelClasses(
    config_class="transformers:LlamaConfig",
    model_class="transformers:LlamaForCausalLM",
    min_transformers=MIN_TRANSFORMERS,
    tested_transformers=TESTED_TRANSFORMERS,
    extra="evograd[llama3]",
    label="Llama-3",
)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    return _common.require_transformers(CLASSES)


def build_config(spec: WorkloadSpec):
    """A ``LlamaConfig`` carrying the spec's architecture and run settings."""
    return _common.build_config(spec, CLASSES)


def build_model(spec: WorkloadSpec):
    """The reference model, on ``spec.device``, in train mode.

    Random initialisation goes through the shared builder; pinned weights are
    loaded (:func:`_load_pretrained`) and left in the state the shared builder
    leaves a model in.
    """
    if not spec.pretrained:
        return _common.build_model(spec, CLASSES)
    spec.validate()
    require_transformers()
    model = _load_pretrained(spec).to(device=spec.device)
    model.train()
    model.gradient_checkpointing_disable()
    model.config.use_cache = spec.use_cache
    return model


#: Environment variable naming a Hugging Face cache that holds the gated
#: checkpoint (``models--meta-llama--Llama-3.2-1B/snapshots/<revision>``).
CACHE_ENV = "EVOGRAD_LLAMA_HF_CACHE"


def pretrained_snapshot(spec: WorkloadSpec) -> str:
    """The local directory of the pinned checkpoint, resolved without the Hub.

    Read from ``$EVOGRAD_LLAMA_HF_CACHE`` (else the benchmark cache); the
    directory is only read, so a shared, read-only cache serves.
    """
    import os
    from pathlib import Path

    from evograd.benchmark.topdown.common.data_paths import hf_cache_dir

    repo, revision = spec.pretrained_repo
    root = Path(os.environ.get(CACHE_ENV) or hf_cache_dir("llama_3_2_1b"))
    directory = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    missing = [n for n in ("config.json", "model.safetensors", "tokenizer.json")
               if not (directory / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{repo}@{revision} is not in {root} (missing {missing}); "
                                f"set {CACHE_ENV} to a cache that holds it")
    return str(directory)


def _load_pretrained(spec: WorkloadSpec):
    """The pinned checkpoint, validated against the declared architecture."""
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        pretrained_snapshot(spec), dtype=DTYPES[spec.dtype],
        attn_implementation=spec.attn_implementation, use_cache=spec.use_cache)
    loaded = model.config
    # RoPE has two spellings (Transformers 5 keeps ``rope_theta`` inside
    # ``rope_scaling``): compared as (theta, type) plus every declared factor.
    rope_keys = {"rope_theta", "rope_scaling"}
    mismatched = {key: (getattr(loaded, key, None), value)
                  for key, value in spec.arch.items()
                  if key not in rope_keys and getattr(loaded, key, None) != value}
    declared_scaling = dict(spec.arch.get("rope_scaling") or {})
    loaded_scaling = dict(getattr(loaded, "rope_scaling", None) or {})
    if rope_settings(loaded) != rope_settings(dict(spec.arch)) or any(
            loaded_scaling.get(k) != v for k, v in declared_scaling.items()):
        mismatched["rope"] = (loaded_scaling, {"rope_theta": spec.arch.get("rope_theta"),
                                               **declared_scaling})
    if mismatched:
        raise RuntimeError("the pinned checkpoint's architecture disagrees with the declared "
                           f"workload: {mismatched}")
    other = sorted({str(p.dtype) for p in model.parameters()} - {str(DTYPES[spec.dtype])})
    if other:
        raise RuntimeError(f"parameters loaded in {other}, spec says {spec.dtype}")
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    if spec.arch.get("tie_word_embeddings") and not tied:
        raise RuntimeError("declared tie_word_embeddings=True but the loaded head is untied")
    return model
