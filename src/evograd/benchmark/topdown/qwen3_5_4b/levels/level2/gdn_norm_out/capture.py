"""One ``gdn_norm_out`` invocation, taken from a Gated DeltaNet block.

Inputs are the delta rule's output and the ``z`` the gated norm receives; the
output is ``linear_attn``'s, whose gradient is the upstream gradient.
"""

from __future__ import annotations

from .. import hf_modules
from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, patched, run_layer

SITE = "gdn_norm_out"
TASK = "qwen3_5_gdn_norm_out"


def capture(source: BlockSource) -> DerivedCall:
    gdn = source.layer.linear_attn
    rec = Recorder()

    def delta_rule(original):
        def wrapper(*args, **kwargs):
            out, state = original(*args, **kwargs)
            rec.value("o", out)
            return out, state
        return wrapper

    def norm_pre(module, args):
        rec.value("z2d", args[1])

    def gdn_post(module, args, kwargs, output):
        rec.value("y", output)
        rec.grad_of("dy", output)

    with hooks(gdn.norm.register_forward_pre_hook(norm_pre),
               gdn.register_forward_hook(gdn_post, with_kwargs=True)), \
            patched(hf_modules.modeling, "torch_chunk_gated_delta_rule", delta_rule):
        run_layer(source)
    rec.require("o", "z2d", "y", "dy")
    o = rec.values["o"]
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={"o": o, "z": rec.values["z2d"].reshape(o.shape),
                "w_norm": keep(gdn.norm.weight), "w_out": keep(gdn.out_proj.weight),
                "eps": float(gdn.norm.variance_epsilon)},
        upstream={"dy": rec.grads["dy"]},
        outputs={"y": rec.values["y"]},
    )


__all__ = ["SITE", "TASK", "capture"]
