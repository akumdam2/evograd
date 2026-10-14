"""Qwen3.5's half of the purity gate: how many calls, and how to rebuild a provider.

    PYTHONPATH=src python -m evograd.evaluation.tier3.workloads.qwen3_5_4b.purity \
        --provider structural

The question -- *is this provider a function of its arguments?* -- and its
machinery are shared (:mod:`evograd.evaluation.tier3.gate.purity`). Qwen3.5's
answers: twice each site's whole-model invocation count *by layer type* (24 GDN
layers, 8 full-attention layers, 32 of the rest), the suite carrying the
production shapes, and how to rebuild ``structural`` / ``bound`` in a fresh
interpreter.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from evograd.evaluation.tier3.gate import purity as _gate
from evograd.evaluation.tier3.gate.purity import child_main, child_parser  # noqa: F401

CHILD_MODULE = "evograd.evaluation.tier3.workloads.qwen3_5_4b.purity"
OBSERVED_SUITE = "qwen3_5_4b_observed"


def min_calls(layer_types=None) -> dict[str, int]:
    """Twice each site's invocation count in one step of a model with these layer types."""
    from evograd.benchmark.topdown.qwen3_5_4b.architecture import layer_types as published

    from .sites import expected_counts

    counts = expected_counts(layer_types if layer_types is not None else published())
    return {site: 2 * count for site, count in counts.items()}


#: 48 for the GDN sites, 16 for the attention sites, 64 for residual and MLP.
MIN_CALLS = min_calls()


def check_kernels(kernels, *, suite: str = OBSERVED_SUITE, device: str = "cuda",
                  calls: dict[str, int] | None = None) -> dict[str, Any]:
    return _gate.check_kernels(kernels, suite=suite, device=device,
                               calls=MIN_CALLS if calls is None else calls)


def run_for(kernels, workload, *, device: str = "cuda") -> dict[str, Any]:
    """The purity gate for one kernel set, isolated in a child where it can be."""
    return _gate.run_for(kernels, workload, module=CHILD_MODULE, suite=OBSERVED_SUITE,
                         calls=min_calls(workload.layer_types), device=device)


def kernels_for(provider: str, workload, fault: dict[str, Any] | None):
    """A provider name back into a :class:`KernelSet`, in a fresh process."""
    from evograd.benchmark import TASKS

    from .sites import bound_pair_identity_kernels, structural_identity_kernels

    if fault:
        raise ValueError("Qwen3.5 has no fault catalogue yet; a faulted provider cannot be "
                         "rebuilt")
    if provider == "structural":
        return structural_identity_kernels(workload.site_registry)
    if provider == "bound":
        return bound_pair_identity_kernels(TASKS, None, workload.site_registry)
    raise ValueError(f"unknown provider {provider!r}")


def main(argv: list[str] | None = None) -> int:
    parser = child_parser(f"python -m {CHILD_MODULE}", __doc__)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from evograd.evaluation.tier3.patch import restrict

    from .workload import Qwen3_5Workload

    config = json.loads(args.workload) if args.workload else {"device": args.device}
    workload = Qwen3_5Workload.from_config(config)
    kernels = kernels_for(args.provider, workload,
                          json.loads(args.fault) if args.fault else None)
    if args.sites:
        kernels = restrict(kernels, tuple(s.strip() for s in args.sites.split(",")))
    return child_main(args, kernels, suite=OBSERVED_SUITE, calls=min_calls(workload.layer_types))


if __name__ == "__main__":
    sys.exit(main())
