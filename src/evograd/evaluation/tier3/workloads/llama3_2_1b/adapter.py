"""What the tier-3 CLI needs to know about Llama-3, stated by Llama-3.

The runner never asks what model it is driving, and neither does the CLI: it
looks up an adapter by name in ``bench.workloads.TIER3_ADAPTERS`` and asks it.
This is Llama's side of that contract.

Three things are declared here:

* how to build the workload from parsed arguments -- including that the
  canonical batch and sequence are the *workload's*, not the CLI's, so only an
  explicit flag overrides them;
* the ``structural_identity`` provider, which exists because Llama's adapters
  can call the exact Transformers spellings through native autograd;
* the ``torch.compile`` providers, which are generic and only need declaring;
* the block-scope adapter (:mod:`.block`), one decoder layer for ``--scope block``;
* which optional flags mean anything here, so the parser can refuse the rest by
  name instead of accepting them silently.
"""

from __future__ import annotations

from typing import Any

from evograd.evaluation.tier3.workloads import Tier3Adapter


def build(args) -> Any:
    """The canonical Llama-3.2-1B training step, shrunk only where asked.

    Every value comes off the command line, which is what lets a child process
    reconstruct an identical workload from the same argv rather than inheriting
    an object it cannot pickle.
    """
    from .workload import Llama3Workload

    config: dict[str, Any] = {
        "device": args.device,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }
    # The canonical batch, sequence and dtype are part of the workload's
    # identity, so a CLI default would silently replace them. ``--dtype`` has no
    # default for exactly that reason. Only an explicit flag overrides any.
    if args.dtype is not None:
        config["dtype"] = args.dtype
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.tokens is not None:
        config["seq_len"] = args.tokens
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    # A calibration is bound to the workload id it was measured for, so a
    # shrunk run cannot borrow the canonical one -- the gate refuses it by
    # name. Pointing at a matching artifact is how a debug run gets a gate at
    # all, rather than having to overwrite the canonical file.
    if getattr(args, "calibration", None) is not None:
        config["calibration_path"] = str(args.calibration)
    return Llama3Workload.from_config(config)


def providers(args, registry) -> dict[str, Any]:
    """Llama's extra providers: the structural-identity control, and compiled ones.

    Patching every site with an adapter that calls the exact Transformers
    spelling changes the module structure and no arithmetic, so the result must
    be *bitwise* identical to the unmodified model.

    ``--compile-site`` and ``--patch-set`` are not Llama's -- they mean the same
    thing for any workload with a site registry -- so they are built by
    :func:`evograd.evaluation.tier3.providers.compile_and_patch_set_providers`
    and this only has to say that Llama offers them.
    """
    from evograd.evaluation.tier3.providers import compile_and_patch_set_providers

    providers: dict[str, Any] = {}
    if getattr(args, "structural_identity", False):
        from .sites import structural_identity_kernels

        providers["structural_identity"] = structural_identity_kernels(registry)
    if getattr(args, "whole_model_compile", False):
        from .workload import whole_model_compile_kernels

        providers["torch_compile_model"] = whole_model_compile_kernels(registry)
    providers.update(compile_and_patch_set_providers(args, registry))
    return providers


def block(args):
    """The block-scope adapter: one decoder layer, captured or config-derived."""
    from .block import from_args

    return from_args(args)


ADAPTER = Tier3Adapter(
    name="llama_3_2_1b",
    build=build,
    providers=providers,
    block=block,
    options=frozenset({"structural_identity", "layers", "data_seed", "calibration",
                       "compile_site", "patch_set", "whole_model_compile"}),
    summary=(
        "Llama-3.2-1B, 16 layers, the canonical training step "
        "(~9 GiB of weights, grads and AdamW moments; fits at full depth)"
    ),
)
