"""Which Transformers classes Llama-3 builds, bound to the shared builder.

Building a model from a spec, and reading back what was actually built rather
than what was asked for, is the same procedure for every decoder-only causal LM
and lives in :mod:`....common.model`. Llama's contribution is two class names.

No Hub access is needed for the canonical workload despite Llama-3 being a
gated repository: it constructs from the written-out configuration with random
weights, and fetches neither checkpoint nor tokenizer. The real-text identity
loads the pinned checkpoint, but only from the local cache -- a benchmark run
never reaches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from ....common.model import (  # noqa: F401  (re-export)
    DTYPES,
    MissingDependencyError,
    ModelClasses,
    check_effective_settings,
    effective_settings,
    make_inputs,
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

    Random initialisation goes through the shared builder. The real-text spec
    loads the pinned checkpoint instead (validated against the declared
    architecture) and leaves it in exactly the state the shared builder would.
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


#: Files a usable snapshot must hold: weights, config, and the tokenizer that
#: packs the real text.
SNAPSHOT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def pretrained_snapshot(spec: WorkloadSpec) -> str:
    """The local directory of the pinned checkpoint. Local-only, never fetched here."""
    from huggingface_hub import snapshot_download
    from evograd.benchmark.topdown.common.data_paths import hf_cache_dir

    repo, revision = spec.pretrained_repo
    try:
        location = snapshot_download(
            repo, revision=revision, local_files_only=True,
            cache_dir=hf_cache_dir("llama_3_2_1b"),
            allow_patterns=["*.json", "*.safetensors"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"the pinned checkpoint {repo}@{revision} is not in the local cache "
            f"({hf_cache_dir('llama_3_2_1b')}). It is gated: accept Meta's license on "
            "the Hub, log in, and run `python scripts/prepare_benchmark_inputs.py "
            "--benchmark llama_3_2_1b` once") from exc
    missing = [name for name in SNAPSHOT_FILES if not (Path(location) / name).is_file()]
    if missing:
        raise RuntimeError(f"the cached Llama snapshot at {location} is incomplete: {missing}")
    return location


def _load_pretrained(spec: WorkloadSpec):
    """The pinned checkpoint, validated against the architecture the spec declares.

    Every operator contract, patch site and calibration was derived from the
    declared architecture, so a checkpoint that disagrees with it would run and
    silently measure a different model. The comparison reads the published
    ``config.json`` itself rather than the loaded config, which Transformers
    normalises (RoPE in particular) differently across releases.
    """
    from transformers import LlamaForCausalLM

    location = pretrained_snapshot(spec)
    published = json.loads((Path(location) / "config.json").read_text())
    mismatched = {key: (published.get(key), value) for key, value in spec.arch.items()
                  if published.get(key) != value}
    if mismatched:
        raise RuntimeError(
            "the pinned checkpoint's config.json disagrees with the declared "
            f"architecture (published, declared): {mismatched}")

    model = None
    for keyword in ("dtype", "torch_dtype"):
        try:
            model = LlamaForCausalLM.from_pretrained(
                location, local_files_only=True, use_safetensors=True,
                attn_implementation=spec.attn_implementation, use_cache=spec.use_cache,
                **{keyword: DTYPES[spec.dtype]})
            break
        except TypeError:
            continue
    if model is None:  # pragma: no cover - both keywords rejected
        raise RuntimeError("LlamaForCausalLM.from_pretrained accepted neither dtype keyword")
    wrong_dtype = sorted({str(p.dtype) for p in model.parameters()} - {str(DTYPES[spec.dtype])})
    if wrong_dtype:
        raise RuntimeError(f"checkpoint loaded parameters in {wrong_dtype}, spec says {spec.dtype}")

    # Tied embeddings: the head must *be* the embedding, or the declared
    # parameter count and the embedding's two gradient paths are both wrong.
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("declared tie_word_embeddings=True but the loaded head is untied")

    # The QKV contract the patched kernel was generated against.
    declared = spec.arch
    attention = model.model.layers[0].self_attn
    head_dim, hidden = declared["head_dim"], declared["hidden_size"]
    expected = {
        "q_proj": (declared["num_attention_heads"] * head_dim, hidden),
        "k_proj": (declared["num_key_value_heads"] * head_dim, hidden),
        "v_proj": (declared["num_key_value_heads"] * head_dim, hidden),
        "o_proj": (hidden, declared["num_attention_heads"] * head_dim),
    }
    for name, shape in expected.items():
        projection = getattr(attention, name)
        if tuple(projection.weight.shape) != shape:
            raise RuntimeError(
                f"self_attn.{name}.weight is {tuple(projection.weight.shape)}, "
                f"the declared contract needs {shape}")
        if projection.bias is not None:
            raise RuntimeError(f"self_attn.{name} has a bias; the declared contract has none")
    return model
