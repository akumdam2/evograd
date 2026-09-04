"""What the tier-3 CLI needs to know about AlphaFold3, stated by AlphaFold3.

The runner never asks what model it is driving, and neither does the CLI: it
looks a workload up by name in ``bench.workloads.TIER3_ADAPTERS`` and asks the
adapter it finds. This is AlphaFold3's side of that.

Three things are declared here, and each would otherwise be a branch in
``tier3_cli``:

* how to build the workload from parsed arguments -- including that a crop
  length is this model's analogue of a sequence length, and that its canonical
  dtype is float32 rather than the bfloat16 the language models train in;
* which optional flags mean anything here, so the parser can refuse the rest by
  name instead of accepting them silently;
* that ``--structural-identity`` is not among them: AlphaFold3 patches by module
  surgery and has no second spelling to compare against.

The workload itself is :mod:`.workload`; nothing about the harness lives there
either.
"""

from __future__ import annotations

from typing import Any

from evograd.bench.workloads import Tier3Adapter

#: The crop the benchmark measures when none is given. AlphaFold3's cost is
#: quadratic in residues, so this is the smallest crop that still exercises the
#: pair stack rather than a number chosen for speed.
DEFAULT_RESIDUES = 128

#: MegaFold trains AlphaFold3 in float32, and the declared operator tolerances
#: were calibrated there. The language-model workloads default to bfloat16, which
#: is why ``--dtype`` carries no parser-level default: each workload's canonical
#: dtype is part of its identity, not the CLI's to choose.
DEFAULT_DTYPE = "float32"


def _config(name: str):
    from evograd.opdecl import models as model_registry

    return {
        "alphafold3_2l": model_registry.ALPHAFOLD3_2L,
        "alphafold3": model_registry.ALPHAFOLD3,
    }[name]


def build(args) -> Any:
    """The AlphaFold3 training step at the requested crop.

    Every value comes off the command line, which is what lets a child process
    reconstruct an identical workload from the same argv rather than inheriting
    an object it cannot pickle.

    ``train_step_dims`` derives the case rather than accepting one: the dims a
    crop implies are a property of the configuration, and ``make_workload``
    re-derives them a second time and refuses a case that does not match, so a
    run cannot quietly claim an architecture it does not describe.
    """
    from evograd.opdecl.activity import Workload
    from evograd.ops.level4.alphafold3.workload import make_workload

    config = _config(args.model)
    case = Workload(
        dims=config.train_step_dims(
            batch=args.batch if args.batch is not None else 1,
            residues=args.residues if args.residues is not None else DEFAULT_RESIDUES,
        ),
        dtype=args.dtype or DEFAULT_DTYPE,
    )
    return make_workload(case, device=args.device, seed=args.seed, config=config)


def _adapter(name: str, summary: str) -> Tier3Adapter:
    return Tier3Adapter(
        name=name,
        build=build,
        # No extra providers: patching is by module surgery, and `eager`, the
        # identity control and any declared pair baseline already cover it.
        # There is no second production spelling to compare bitwise against, so
        # `--structural-identity` is deliberately absent from `options` below.
        providers=None,
        options=frozenset({"residues"}),
        summary=summary,
    )


ADAPTER = _adapter(
    "alphafold3",
    "AlphaFold3 as MegaFold configures it; float32, crop-length scaled",
)

ADAPTER_2L = _adapter(
    "alphafold3_2l",
    "AlphaFold3 reduced to 2 blocks; the one to iterate on",
)
