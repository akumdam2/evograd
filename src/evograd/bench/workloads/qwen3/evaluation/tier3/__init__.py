"""Tier 3 for Qwen3-0.6B: replace an operator inside the real model.

Four sites in the live ``Qwen3ForCausalLM`` can be swapped without touching the
call site, the model state, the parameter names, or the training loop. What
makes that measurable rather than merely possible is the untimed gate in
:mod:`.gate`, which runs in a fixed order and refuses to time anything that
fails it:

1. :mod:`.validate` -- the sites hold at the shapes this model supplies;
2. :mod:`.purity` -- the provider is a function of its arguments;
3. :mod:`.boundary` -- all 140 invocations match their declaration;
4. :mod:`.numerics` -- the whole model stays inside a calibrated envelope;
5. the calibrated loss trajectory;
6. invocation counts and patch provenance.

:mod:`.calibrate` measures the envelope on the machine it will be enforced on,
:mod:`.faults` and :mod:`.controls` show it still rejects a wrong provider.
"""

from .workload import Qwen3Workload

__all__ = ["Qwen3Workload"]
