"""What the tier-3 CLI needs to know about OLMoE-1B-7B, stated by OLMoE.

The whole-model training step (benchmark Level 3):

    python -m evograd.evaluation.tier3.cli --model olmoe_1b_7b --structural-identity

The canonical batch, sequence and dtype are the workload's (B2 x T2048 bf16);
only an explicit flag overrides them. ``--layers N`` cuts the depth. The
experts kernel is the spec's (grouped GEMM) and has no flag in this step.
"""

from __future__ import annotations

from typing import Any

from evograd.evaluation.tier3.workloads import Tier3Adapter


def build(args) -> Any:
    """The canonical OLMoE-1B-7B training step, shrunk only where asked."""
    from .workload import OlmoeWorkload

    config: dict[str, Any] = {"device": args.device, "seed": args.seed,
                              "data_seed": args.data_seed}
    if args.dtype is not None:
        config["dtype"] = args.dtype
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.tokens is not None:
        config["seq_len"] = args.tokens
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    return OlmoeWorkload.from_config(config)


def providers(args, registry) -> dict[str, Any]:
    """The structural-identity control, whole-model compile, and the generic
    compile / patch-set providers."""
    out: dict[str, Any] = {}
    if getattr(args, "structural_identity", False):
        from .sites import structural_identity_kernels

        out["structural_identity"] = structural_identity_kernels(registry)
    if getattr(args, "whole_model_compile", False):
        from .workload import whole_model_compile_kernels

        out["torch_compile_model"] = whole_model_compile_kernels(registry)
    from evograd.evaluation.tier3.providers import compile_and_patch_set_providers

    out.update(compile_and_patch_set_providers(args, registry))
    return out


ADAPTER = Tier3Adapter(
    name="olmoe_1b_7b",
    build=build,
    providers=providers,
    options=frozenset({"structural_identity", "compile_site", "patch_set",
                       "whole_model_compile", "layers", "data_seed"}),
    summary=("OLMoE-1B-7B, 16 layers of multi-head attention and 64 routed SwiGLU experts "
             "(8 active), B2 x T2048, grouped-GEMM experts"),
)
