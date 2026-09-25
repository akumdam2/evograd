"""What the tier-3 CLI needs to know about MegaFold, stated by MegaFold.

The runner never asks what model it is driving, and neither does the CLI: it
looks a workload up by name in
``evaluation.tier3.workloads.TIER3_ADAPTERS`` and asks the adapter it finds.
This is MegaFold's side of that.

Three things are declared here, and each would otherwise be a branch in the
tier-3 CLI:

* how to build the workload from parsed arguments -- including that a crop
  length is this model's analogue of a sequence length, and that its canonical
  dtype is float32 rather than the bfloat16 the language models train in;
* the four providers MegaFold has that no other workload does -- ``eager``,
  ``megafold``, ``compile`` and ``compile_clean`` -- which are four different
  *trees* rather than four kernel sets: MegaFold builds its fused Triton modules straight into the
  module tree and ``torch.compile`` wraps a whole model, so the choice between
  them is made at construction and the runner passes the provider's name to the
  workload;
* which optional flags mean anything here, so the parser can refuse the rest by
  name instead of accepting them silently.

``--structural-identity`` is deliberately not among them: MegaFold patches by
module surgery and has no second spelling to compare bitwise against.

The workload itself is :mod:`.workload`; nothing about the harness lives there.
"""

from __future__ import annotations

from typing import Any

from evograd.evaluation.tier3.workloads import Tier3Adapter

#: The crop the benchmark measures when none is given. AlphaFold3's cost is
#: quadratic in residues, so this is the smallest crop that still exercises the
#: pair stack rather than a number chosen for speed.
DEFAULT_RESIDUES = 128

#: MegaFold holds its weights in float32 and gets mixed precision from autocast
#: rather than from the parameter dtype, and the declared operator tolerances
#: were calibrated there. The language-model workloads default to bfloat16,
#: which is why ``--dtype`` carries no parser-level default: each workload's
#: canonical dtype is part of its identity, not the CLI's to choose.
DEFAULT_DTYPE = "float32"


def _config(name: str):
    from evograd.opdecl import models as model_registry

    return {
        "megafold_2l": model_registry.ALPHAFOLD3_2L,
        "megafold": model_registry.ALPHAFOLD3,
    }[name]


def _settings(args) -> dict[str, Any]:
    """The workload's own knobs, from the flags that name them.

    Only a flag that was actually given appears here, so an unset flag leaves
    the workload's own default in place rather than overwriting it with the
    parser's idea of one.
    """
    settings: dict[str, Any] = {}
    evo = getattr(args, "evo", None)
    if evo is not None:
        settings["use_optimized_evo"] = None if evo == "none" else evo
    checkpointing = getattr(args, "checkpointing", None)
    if checkpointing is not None:
        settings["checkpointing"] = checkpointing == "on"
    recycling = getattr(args, "recycling", None)
    if recycling is not None:
        settings["num_recycling_steps"] = recycling
    if getattr(args, "confidence_head", False):
        settings["include_confidence_head"] = True
    return settings


def build(args) -> Any:
    """The MegaFold training step at the requested crop.

    Every value comes off the command line, which is what lets a child process
    reconstruct an identical workload from the same argv rather than inheriting
    an object it cannot pickle.

    ``train_step_dims`` derives the case rather than accepting one: the dims a
    crop implies are a property of the configuration, and ``make_workload``
    re-derives them a second time and refuses a case that does not match, so a
    run cannot quietly claim an architecture it does not describe.
    """
    from evograd.opdecl.activity import Workload
    from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

    config = _config(args.model)
    case = Workload(
        dims=config.train_step_dims(
            batch=args.batch if args.batch is not None else 1,
            residues=args.residues if args.residues is not None else DEFAULT_RESIDUES,
        ),
        dtype=args.dtype or DEFAULT_DTYPE,
    )
    return make_workload(
        case, device=args.device, seed=args.seed, config=config, **_settings(args)
    )


def providers(args, registry) -> dict[str, Any]:
    """MegaFold's four trees, declared by name.

    These providers differ by *construction* rather than by kernel: MegaFold
    builds its fused Triton modules straight into the module tree, and
    ``torch.compile`` wraps a whole model rather than patching a site, so all
    three carry the same empty kernel set and the runner tells the workload
    which one it is building.

    ``eager``
        Every fused module replaced by the plain PyTorch mathematics it stands
        for. The reference, and the base the other two are read against.
    ``megafold``
        The tree as MegaFold builds it, with its Triton EvoAttention on.
    ``compile``
        ``torch.compile`` of the eager model, with no module surgery at all.
        The whole-model compiler baseline: what a general-purpose compiler
        gets on this architecture without anyone writing a kernel.
    ``compile_clean``
        The same, with MegaFold's scaffolding rebound out of the compiler's
        way first: memory probes, the rank-0 print, sequence-parallel
        identities, one decorator's generator, one dtype branch. No
        mathematics changes. What the compiler gets when the code stops
        breaking its graph for reasons that are not the model.

    ``eager`` deliberately replaces the entry the CLI creates for every
    workload. It builds the same thing either way; it is stated here so that
    this workload's reference provider is declared rather than inherited.

    There is no ``evolved`` entry, because the CLI already builds one from
    ``--candidate`` and names it ``candidate``. A provider name this workload
    does not recognize gets the eager tree and then its own kernels on top, so
    the evolved provider differs from the reference in its kernels and nothing
    else.
    """
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.megafold.workload import BUILD_MODES

    from evograd.evaluation.tier3.providers import compile_and_patch_set_providers
    providers = {mode: KernelSet(registry=registry) for mode in BUILD_MODES}
    # Per-site torch.compile (--compile-site) and composed patch sets
    # (--patch-set): kernel-level providers on the eager tree, the route
    # Llama-3 and Qwen3 offer. Distinct from the whole-model compile trees.
    providers.update(compile_and_patch_set_providers(args, registry))
    return providers


#: The workload-specific flags this adapter understands. Anything else the
#: parser carries is refused by name for this model rather than ignored.
OPTIONS = frozenset(
    {"residues", "evo", "checkpointing", "recycling", "confidence_head",
     "compile_site", "patch_set"}
)


def _adapter(name: str, summary: str) -> Tier3Adapter:
    return Tier3Adapter(
        name=name,
        build=build,
        providers=providers,
        options=OPTIONS,
        summary=summary,
    )


ADAPTER = _adapter(
    "megafold",
    "MegaFold as its own 1x1 config builds it; float32 weights, crop-length scaled",
)

ADAPTER_2L = _adapter(
    "megafold_2l",
    "MegaFold reduced to 2 pairformer blocks; the one to iterate on",
)
