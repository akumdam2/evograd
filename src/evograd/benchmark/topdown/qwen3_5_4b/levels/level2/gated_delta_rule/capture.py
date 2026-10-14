"""One ``gated_delta_rule`` invocation, taken from a Gated DeltaNet block.

The boundary is the module-level ``torch_chunk_gated_delta_rule`` call inside
``linear_attn``: its five tensor arguments are the inputs, its first return the
output, and the gradient arriving at that return the upstream gradient.
"""

from __future__ import annotations

from .. import hf_modules
from ..capture_tools import BlockSource, DerivedCall, Recorder, patched, run_layer

SITE = "gated_delta_rule"
TASK = "qwen3_5_gated_delta_rule"


def capture(source: BlockSource) -> DerivedCall:
    rec = Recorder()

    def delta_rule(original):
        def wrapper(query, key, value, g=None, beta=None, **kwargs):
            for name, tensor in (("q", query), ("k", key), ("v", value), ("g", g), ("beta", beta)):
                rec.value(name, tensor)
            out, state = original(query, key, value, g=g, beta=beta, **kwargs)
            rec.value("o", out)
            rec.grad_of("do", out)
            return out, state
        return wrapper

    with patched(hf_modules.modeling, "torch_chunk_gated_delta_rule", delta_rule):
        run_layer(source)
    rec.require("q", "k", "v", "g", "beta", "o", "do")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={name: rec.values[name] for name in ("q", "k", "v", "g", "beta")},
        upstream={"do": rec.grads["do"]},
        outputs={"o": rec.values["o"]},
    )


__all__ = ["SITE", "TASK", "capture"]
