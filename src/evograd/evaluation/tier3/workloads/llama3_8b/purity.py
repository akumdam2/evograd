"""Llama's half of the purity gate: how many calls, and how to rebuild a provider.

    PYTHONPATH=src python -m evograd.evaluation.tier3.workloads.llama3_8b.purity \
        --provider structural

The question the gate asks -- *is this provider a function of its arguments, or
does it remember?* -- is not about Llama-3, and neither is the machinery that
asks it. That lives in :mod:`evograd.evaluation.tier3.gate.purity`.

Three things are Llama-specific, and they are all that is left here:

* **How many calls each site is worth.** Twice the canonical invocation count,
  so a provider that only misbehaves after "more calls than preflight makes"
  has nowhere to hide. Those counts come from this model's 16 layers.
* **Which benchmark suite carries the production shapes.** Purity is a question
  about the provider at the width the model runs, not on the declaration's small
  correctness grid. That suite does not exist until a harvest has been run, and
  the shared gate falls back to the declaration's own benchmark when it is
  missing rather than refusing.
* **How to rebuild a named provider in a fresh interpreter.** ``structural`` and
  ``bound`` mean something only once you know which registry and which adapters
  they refer to, and only this package does.

The child process is the point of the gate, not an implementation detail: a
second Python object is not a reset -- module-level counters, lazily built
caches and autotuner memos all survive one. So the provider the model
validation later runs is constructed in a new interpreter, here.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from evograd.evaluation.tier3.gate.purity import (  # noqa: F401  (re-export)
    SCHEMA_VERSION,
    checkpoints,
    child_main,
    child_parser,
    run_isolated,
    spec_for,
)
from evograd.evaluation.tier3.gate import purity as _gate

from .sites import (
    OBSERVED_SUITE,
    SITE_ATTENTION,
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
)
from evograd.benchmark.topdown.llama3_8b.levels.level4.spec import LLAMA_3_8B

_LAYERS = LLAMA_3_8B["num_hidden_layers"]

#: This module, as ``python -m`` names it. The isolated run re-enters here.
CHILD_MODULE = "evograd.evaluation.tier3.workloads.llama3_8b.purity"

#: Twice the canonical invocation count of each site: 16 layers give 16
#: invocations of the three per-layer sites and 64 residual fusions, so a
#: provider is called about twice as often here as the model will call it.
#: Derived rather than written out, so a reduced-layer spec cannot make the
#: constant and the architecture disagree.
MIN_CALLS = {
    SITE_QKV: 2 * _LAYERS,
    SITE_ATTENTION: 2 * _LAYERS,
    SITE_MLP: 2 * _LAYERS,
    SITE_RESIDUAL: 4 * _LAYERS,
}


def check_site(site: str, op_name: str, kernel, *, registry=None,
               suite: str = OBSERVED_SUITE, device: str = "cuda",
               calls: int | None = None) -> dict[str, Any]:
    """One Llama site through the shared purity gate."""
    from .sites import llama3_sites

    return _gate.check_site(
        site, op_name, kernel,
        registry=llama3_sites() if registry is None else registry,
        suite=suite, device=device, calls=calls or MIN_CALLS.get(site),
    )


def check_kernels(kernels, *, suite: str = OBSERVED_SUITE, device: str = "cuda",
                  calls: dict[str, int] | None = None) -> dict[str, Any]:
    """Every patched site of one Llama kernel set, in this process."""
    return _gate.check_kernels(
        kernels, suite=suite, device=device, calls=MIN_CALLS if calls is None else calls,
    )


def run_for(kernels, workload, *, device: str = "cuda") -> dict[str, Any]:
    """The purity gate for one Llama kernel set, isolated where it can be."""
    return _gate.run_for(
        kernels, workload, module=CHILD_MODULE, suite=OBSERVED_SUITE,
        calls=MIN_CALLS, device=device,
    )


def kernels_for(provider: str, workload, fault: dict[str, Any] | None):
    """Turn a provider name back into a :class:`KernelSet` in a fresh process.

    This is the step that cannot be shared: ``structural`` and ``bound`` are
    Llama's two identity controls, and reaching them means reaching Llama's
    adapters.
    """
    from evograd.benchmark import TASKS

    from .sites import bound_pair_identity_kernels, structural_identity_kernels

    if provider == "structural":
        kernels = structural_identity_kernels(workload.site_registry)
    elif provider == "bound":
        kernels = bound_pair_identity_kernels(TASKS, None, workload.site_registry)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    if fault:
        from .faults import Fault

        kernels = Fault(fault["name"], fault["site"], fault["magnitude"],
                        fault.get("describes", "")).apply(workload, kernels)
    return kernels


def main(argv: list[str] | None = None) -> int:
    parser = child_parser(f"python -m {CHILD_MODULE}", __doc__)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from evograd.evaluation.tier3.patch import restrict

    from .workload import Llama3Workload

    config = json.loads(args.workload) if args.workload else {"device": args.device}
    workload = Llama3Workload.from_config(config)
    kernels = kernels_for(args.provider, workload,
                          json.loads(args.fault) if args.fault else None)
    if args.sites:
        kernels = restrict(kernels, tuple(s.strip() for s in args.sites.split(",")))
    return child_main(args, kernels, suite=OBSERVED_SUITE, calls=MIN_CALLS)


if __name__ == "__main__":
    sys.exit(main())
