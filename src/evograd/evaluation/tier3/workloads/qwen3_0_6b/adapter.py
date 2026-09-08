"""What the tier-3 CLI needs to know about Qwen3, stated by Qwen3.

The runner never asks what model it is driving. The CLI used to, because a
``--model qwen3_0_6b`` string had to become a workload, and ``--structural-identity``
had to become a provider, and both were ``if args.model == "qwen3_0_6b"``
branches inside ``tier3_cli``. This module is the other side of that: the CLI
looks up an adapter by name and asks it, so the evaluation half holds no
knowledge of any particular architecture.

Three things are declared here, and each was previously a branch:

* how to build the workload from parsed arguments -- including that the
  canonical batch and sequence are the *workload's*, not the CLI's, so only an
  explicit flag overrides them;
* the ``structural_identity`` provider, which exists for Qwen3 because its
  adapters can call the exact Transformers spellings through native autograd;
* which optional flags mean anything here, so the parser can refuse the rest by
  name instead of accepting them silently.
"""

from __future__ import annotations

from typing import Any

from evograd.evaluation.tier3.workloads import Tier3Adapter


def build(args) -> Any:
    """The canonical Qwen3-0.6B training step, shrunk only where asked.

    Every value comes off the command line, which is what lets a child process
    reconstruct an identical workload from the same argv rather than inheriting
    an object it cannot pickle.
    """
    from .workload import Qwen3Workload

    config: dict[str, Any] = {
        "device": args.device,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }
    # The canonical batch, sequence and dtype are part of the workload's
    # identity, so a CLI default would silently replace them. ``--dtype`` has no
    # default for exactly that reason -- workloads disagree about theirs, and
    # the canonical value belongs in the spec rather than in the parser. Only an
    # explicit flag overrides any of them.
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
    if getattr(args, "simple_calibration", None) is not None:
        config["simple_calibration_path"] = str(args.simple_calibration)
    # Real text is a different workload identity: pinned pretrained weights and
    # pinned WikiText-2, so no synthetic calibration can bind to it.
    if getattr(args, "real_text", False):
        from .protocol4_cli import PRETRAINED, REAL_TEXT

        config["weights"], config["data"] = PRETRAINED, REAL_TEXT
    if getattr(args, "protocol4_calibration", None) is not None:
        config["protocol4_calibration_path"] = str(args.protocol4_calibration)
    if getattr(args, "protocol4_verdict", None) is not None:
        config["protocol4_verdict_path"] = str(args.protocol4_verdict)
    if getattr(args, "protocol4_diagnostic_timing", False):
        config["protocol4_diagnostic_timing"] = True
    return Qwen3Workload.from_config(config)


def providers(args, registry) -> dict[str, Any]:
    """Qwen3's own extra provider: the structural-identity control.

    Patching every site with an adapter that calls the exact Transformers
    spelling changes the module structure and no arithmetic, so the result must
    be *bitwise* identical to the unmodified model. That is a control no other
    workload in this repository can offer, which is why it is declared here
    rather than in the CLI.
    """
    providers: dict[str, Any] = {}
    if getattr(args, "structural_identity", False):
        from .sites import structural_identity_kernels

        providers["structural_identity"] = structural_identity_kernels(registry)

    # One provider per --compile-site: only that site's arithmetic is replaced,
    # by torch.compile of the *declared* runtime_forward the unpatched model
    # already calls. Compiled autograd is preserved -- a plain callable patched
    # into the site, not a pair wrapper that would recompute backward.
    for site in getattr(args, "compile_site", []) or []:
        from .simple import PatchSet, compiled_trusted_kernels

        if site not in registry:
            raise ValueError(f"--compile-site {site!r} is not a site of {registry.name!r}; "
                             f"known: {sorted(registry.names)}")
        providers[f"torch_compile_{site}"] = compiled_trusted_kernels(
            PatchSet((site,), (), {}), registry)

    # --patch-set NAME:SITE=SPEC,...: one provider patching several sites at once,
    # each by its real route (compile | liger | an evolved program).
    for entry in getattr(args, "patch_set", []) or []:
        from .simple import kernels_from_patches, parse_patch_set_spec

        name, patches = parse_patch_set_spec(entry)
        if name in providers or name == "eager":
            raise ValueError(f"--patch-set name {name!r} is already a provider")
        unknown = [s for s in patches if s not in registry]
        if unknown:
            raise ValueError(f"--patch-set {name!r}: unknown sites {unknown}; "
                             f"known: {sorted(registry.names)}")
        providers[name] = kernels_from_patches(patches, registry)
    return providers


ADAPTER = Tier3Adapter(
    name="qwen3_0_6b",
    build=build,
    providers=providers,
    options=frozenset({"structural_identity", "layers", "data_seed", "calibration",
                       "simple_calibration", "real_text", "protocol4_calibration",
                       "protocol4_verdict", "protocol4_diagnostic_timing",
                       "compile_site", "patch_set"}),
    summary="Qwen3-0.6B, 28 layers, the harvested canonical training step",
)
