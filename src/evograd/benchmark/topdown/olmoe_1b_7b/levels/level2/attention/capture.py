"""One ``attention`` invocation, taken from a decoder layer.

Inputs are what SDPA received; the output is ``self_attn``'s, whose gradient is
the upstream.
"""

from __future__ import annotations

import torch

from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, patched, run_layer

SITE = "attention"
TASK = "olmoe_attention"


def capture(source: BlockSource) -> DerivedCall:
    attn = source.layer.self_attn
    rec = Recorder()

    def attn_post(module, args, kwargs, output):
        rec.value("y", output[0])
        rec.grad_of("dy", output[0])

    def sdpa(original):
        def wrapper(query, key, value, *args, **kwargs):
            for name, tensor in (("q", query), ("k", key), ("v", value)):
                rec.value(name, tensor)
            return original(query, key, value, *args, **kwargs)
        return wrapper

    with hooks(attn.register_forward_hook(attn_post, with_kwargs=True)), \
            patched(torch.nn.functional, "scaled_dot_product_attention", sdpa):
        run_layer(source)
    rec.require("q", "k", "v", "y", "dy")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={"q": rec.values["q"], "k": rec.values["k"], "v": rec.values["v"],
                "o_weight": keep(attn.o_proj.weight)},
        upstream={"dy": rec.grads["dy"]},
        outputs={"y": rec.values["y"]},
    )


__all__ = ["SITE", "TASK", "capture"]
