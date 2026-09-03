"""Building the model, and then checking what was actually built.

Requesting a setting and getting it are different events. ``attn_implementation``
can be silently downgraded, a ``dtype=`` keyword renamed between Transformers
majors can be swallowed by ``**kwargs`` and leave every parameter in float32,
and ``use_cache`` set on a config can be overridden by the forward call. So this
module separates the two: :func:`build_model` asks, :func:`effective_settings`
reads back what the constructed object reports, and the smoke run compares them.

Transformers is an optional dependency. Nothing here is imported at module
import time, so ``import evograd.bench.workloads.qwen3`` works on a machine that
has never installed it, and the failure -- when it comes -- names the extra to
install.
"""

from __future__ import annotations

from typing import Any

import torch

from .spec import WorkloadSpec

#: Qwen3 landed in Transformers 4.51.0; earlier releases have no ``Qwen3Config``.
MIN_TRANSFORMERS = (4, 51)
#: The version this milestone was developed and measured against.
TESTED_TRANSFORMERS = "5.16.1"

_INSTALL_HINT = (
    "The Qwen3 Level-4 workload needs Hugging Face Transformers, an optional "
    "evograd dependency.\n"
    "    pip install 'evograd[qwen3]'    # or: pip install 'transformers>=4.51'\n"
    f"Qwen3 requires transformers >= {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]}; "
    f"tested against {TESTED_TRANSFORMERS}."
)

DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class MissingDependencyError(ImportError):
    """Transformers is absent or too old for Qwen3."""


def _parse_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
        raise MissingDependencyError(_INSTALL_HINT) from exc
    version = _parse_version(getattr(transformers, "__version__", ""))
    if version and version < MIN_TRANSFORMERS:
        raise MissingDependencyError(
            f"transformers {transformers.__version__} has no Qwen3 support.\n" + _INSTALL_HINT
        )
    return transformers


def build_config(spec: WorkloadSpec):
    """A ``Qwen3Config`` carrying the spec's architecture and run settings."""
    require_transformers()
    from transformers import Qwen3Config

    return Qwen3Config(
        **spec.arch,
        use_cache=spec.use_cache,
        attn_implementation=spec.attn_implementation,
    )


def _from_config_with_dtype(config, dtype: torch.dtype):
    """Instantiate in ``dtype`` across the Transformers 4/5 keyword rename.

    ``torch_dtype`` became ``dtype`` in Transformers 5. Both are forwarded
    through ``**kwargs``, so passing the wrong one does not raise -- it just
    leaves the model in float32. Hence the explicit check afterwards: a
    keyword that was silently ignored becomes a loud error here rather than a
    surprising float32 measurement later.
    """
    from transformers import Qwen3ForCausalLM

    last_error: Exception | None = None
    for keyword in ("dtype", "torch_dtype"):
        try:
            model = Qwen3ForCausalLM._from_config(config, **{keyword: dtype})
        except TypeError as exc:  # pragma: no cover - version dependent
            last_error = exc
            continue
        if next(model.parameters()).dtype is dtype:
            return model
        del model
    raise RuntimeError(
        f"could not construct Qwen3ForCausalLM in {dtype}: neither dtype= nor "
        f"torch_dtype= took effect (transformers "
        f"{require_transformers().__version__}, tested {TESTED_TRANSFORMERS})."
        + (f" Last error: {last_error}" if last_error else "")
    )


def build_model(spec: WorkloadSpec):
    """The randomly-initialised reference model, on ``spec.device``, in train mode.

    Initialisation happens on CPU under ``spec.seed`` and the weights are then
    moved, so the same seed gives the same weights whatever device runs them --
    CUDA's RNG stream is not the CPU's. For 0.6B parameters the extra copy costs
    about a second and buys device-independent reproducibility.

    No pretrained weights and no tokenizer are fetched: the workload is a shape
    and a graph, not a trained model, and a reference execution must not depend
    on network access.
    """
    spec.validate()
    require_transformers()
    from transformers import Qwen3ForCausalLM

    config = build_config(spec)
    torch.manual_seed(spec.seed)
    model = _from_config_with_dtype(config, DTYPES[spec.dtype])
    if not isinstance(model, Qwen3ForCausalLM):  # pragma: no cover - defensive
        raise RuntimeError(f"expected Qwen3ForCausalLM, built {type(model).__name__}")
    model = model.to(device=spec.device)
    model.train()
    # Assert the state rather than assume the default: a future Transformers
    # release that enables checkpointing by default would otherwise change the
    # workload without changing this file.
    model.gradient_checkpointing_disable()
    model.config.use_cache = spec.use_cache
    return model


def make_inputs(spec: WorkloadSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic ``input_ids`` and ``labels = input_ids.clone()``.

    Drawn from an explicit CPU generator rather than the global RNG so the token
    stream does not depend on how many random numbers model construction
    consumed, and is identical on CPU and CUDA.
    """
    spec.validate()
    generator = torch.Generator().manual_seed(spec.seed)
    ids = torch.randint(
        low=0,
        high=spec.arch["vocab_size"],
        size=(spec.batch_size, spec.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    input_ids = ids.to(spec.device)
    return input_ids, input_ids.clone()


def training_step(model, input_ids: torch.Tensor, labels: torch.Tensor):
    """The canonical step: causal-LM loss, then backward. No optimizer step.

    ``use_cache=False`` is passed at the call site as well as set on the config,
    because the forward argument is what actually decides.
    """
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    return outputs


def effective_settings(model, spec: WorkloadSpec) -> dict[str, Any]:
    """What the constructed model reports, as distinct from what was requested."""
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    param_devices = sorted({str(p.device).split(":")[0] for p in model.parameters()})
    buffer_dtypes = sorted({str(b.dtype) for b in model.buffers()})
    # Read the backend off every submodule that carries a config, not just the
    # top one: a partial downgrade would leave the root claiming sdpa.
    per_module = sorted(
        {
            impl
            for module in model.modules()
            if (impl := getattr(getattr(module, "config", None), "_attn_implementation", None))
        }
    )
    return {
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "attn_implementation_per_module": per_module,
        "use_cache": bool(model.config.use_cache),
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "training_mode": bool(model.training),
        "param_dtypes": param_dtypes,
        "param_devices": param_devices,
        "buffer_dtypes": buffer_dtypes,
        "requested_dtype": spec.dtype,
        "requested_device": spec.device,
        "requested_attn_implementation": spec.attn_implementation,
    }


def check_effective_settings(effective: dict[str, Any], spec: WorkloadSpec) -> list[str]:
    """Differences between what was asked for and what exists. Empty means clean."""
    problems: list[str] = []
    want_dtype = str(DTYPES[spec.dtype])
    if effective["param_dtypes"] != [want_dtype]:
        problems.append(
            f"parameters are {effective['param_dtypes']}, expected [{want_dtype!r}]"
        )
    if effective["param_devices"] != [spec.device]:
        problems.append(
            f"parameters are on {effective['param_devices']}, expected [{spec.device!r}]"
        )
    if effective["attn_implementation"] != spec.attn_implementation:
        problems.append(
            f"attention backend is {effective['attn_implementation']!r}, "
            f"requested {spec.attn_implementation!r}"
        )
    extra = [i for i in effective["attn_implementation_per_module"] if i != spec.attn_implementation]
    if extra:
        problems.append(f"some submodules report a different attention backend: {extra}")
    if effective["use_cache"] != spec.use_cache:
        problems.append(f"use_cache is {effective['use_cache']}, expected {spec.use_cache}")
    if effective["gradient_checkpointing"] != spec.gradient_checkpointing:
        problems.append(
            f"gradient checkpointing is {effective['gradient_checkpointing']}, "
            f"expected {spec.gradient_checkpointing}"
        )
    if effective["training_mode"] != spec.training:
        problems.append(f"model.training is {effective['training_mode']}, expected {spec.training}")
    return problems
