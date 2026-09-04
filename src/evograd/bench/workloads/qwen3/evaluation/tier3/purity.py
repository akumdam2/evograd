"""Qwen3's half of the purity gate: how many calls, and how to rebuild a provider.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.evaluation.tier3.purity \
        --provider structural

The question the gate asks -- *is this provider a function of its arguments, or
does it remember?* -- is not about Qwen3, and neither is the machinery that asks
it. That lives in :mod:`evograd.bench.tier3_gate.purity`.

Three things are Qwen-specific, and they are all that is left here:

* **How many calls each site is worth.** Twice the canonical invocation count,
  so a provider that only misbehaves after "more calls than preflight makes"
  has nowhere to hide. Those counts come from Qwen3's 28 layers.
* **Which benchmark suite carries the production shapes.** Purity is a question
  about the provider at the width the model runs, not on the declaration's small
  correctness grid.
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

from evograd.bench.tier3_gate.purity import (  # noqa: F401  (re-export)
    SCHEMA_VERSION,
    checkpoints,
    child_main,
    child_parser,
    run_isolated,
    spec_for,
)
from evograd.bench.tier3_gate import purity as _gate

#: This module, as ``python -m`` names it. The isolated run re-enters here.
CHILD_MODULE = "evograd.bench.workloads.qwen3.evaluation.tier3.purity"

#: The benchmark suite carrying the shapes the canonical step actually runs.
OBSERVED_SUITE = "qwen3_0_6b_observed"

#: Twice the canonical invocation count of each site: 28 layers give 28
#: invocations of the three per-layer sites and 56 residual fusions, so a
#: provider is called about twice as often here as the model will call it.
MIN_CALLS = {
    "qkv_norm_rope": 56,
    "attention": 56,
    "swiglu_mlp": 56,
    "residual_rmsnorm": 112,
}


def check_site(site: str, op_name: str, kernel, *, registry=None,
               suite: str = OBSERVED_SUITE, device: str = "cuda",
               calls: int | None = None) -> dict[str, Any]:
    """One Qwen3 site through the shared purity gate.

    The generic gate asks which registry and which suite; this fills in Qwen3's
    answers so a caller inside this package does not repeat them. Both stay
    overridable, because a test that wants a different registry should be able
    to say so rather than monkey-patching one in.
    """
    from .sites import qwen3_sites

    return _gate.check_site(
        site, op_name, kernel,
        registry=qwen3_sites() if registry is None else registry,
        suite=suite, device=device, calls=calls or MIN_CALLS.get(site),
    )


def check_kernels(kernels, *, suite: str = OBSERVED_SUITE, device: str = "cuda",
                  calls: dict[str, int] | None = None) -> dict[str, Any]:
    """Every patched site of one Qwen3 kernel set, in this process."""
    return _gate.check_kernels(
        kernels, suite=suite, device=device, calls=MIN_CALLS if calls is None else calls,
    )


def run_for(kernels, workload, *, device: str = "cuda") -> dict[str, Any]:
    """The purity gate for one Qwen3 kernel set, isolated where it can be."""
    return _gate.run_for(
        kernels, workload, module=CHILD_MODULE, suite=OBSERVED_SUITE,
        calls=MIN_CALLS, device=device,
    )


def kernels_for(provider: str, workload, fault: dict[str, Any] | None):
    """Turn a provider name back into a :class:`KernelSet` in a fresh process.

    This is the step that cannot be shared: ``structural`` and ``bound`` are
    Qwen3's two identity controls, and reaching them means reaching Qwen3's
    adapters.
    """
    from evograd.ops import OPS

    from .sites import bound_pair_identity_kernels, structural_identity_kernels

    if provider == "structural":
        kernels = structural_identity_kernels(workload.site_registry)
    elif provider == "bound":
        kernels = bound_pair_identity_kernels(OPS, None, workload.site_registry)
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

    from evograd.bench.tier3_patch import restrict

    from .workload import Qwen3Workload

    config = json.loads(args.workload) if args.workload else {"device": args.device}
    workload = Qwen3Workload.from_config(config)
    kernels = kernels_for(args.provider, workload,
                          json.loads(args.fault) if args.fault else None)
    if args.sites:
        kernels = restrict(kernels, tuple(s.strip() for s in args.sites.split(",")))
    return child_main(args, kernels, suite=OBSERVED_SUITE, calls=MIN_CALLS)


if __name__ == "__main__":
    sys.exit(main())
