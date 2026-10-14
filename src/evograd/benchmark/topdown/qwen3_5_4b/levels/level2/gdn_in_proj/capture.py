"""One ``gdn_in_proj`` invocation, taken from a Gated DeltaNet block.

The site starts at ``linear_attn``'s input and ends where its six outputs are
consumed: q, k, v, beta and g by the gated delta rule, z by the gated norm. Each
of these tensors has exactly one consumer, so a gradient hook on it is that
consumer's gradient.
"""

from __future__ import annotations

from .. import hf_modules
from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, patched, run_layer

SITE = "gdn_in_proj"
TASK = "qwen3_5_gdn_in_proj"


def capture(source: BlockSource) -> DerivedCall:
    gdn = source.layer.linear_attn
    rec = Recorder()

    def gdn_pre(module, args, kwargs):
        rec.value("x", kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])

    def norm_pre(module, args):
        rec.value("z2d", args[1])
        rec.grad_of("dz2d", args[1])

    def delta_rule(original):
        def wrapper(query, key, value, g=None, beta=None, **kwargs):
            for name, tensor in (("q", query), ("k", key), ("v", value), ("beta", beta), ("g", g)):
                rec.value(name, tensor)
                rec.grad_of("d" + name, tensor)
            return original(query, key, value, g=g, beta=beta, **kwargs)
        return wrapper

    with hooks(gdn.register_forward_pre_hook(gdn_pre, with_kwargs=True),
               gdn.norm.register_forward_pre_hook(norm_pre)), \
            patched(hf_modules.modeling, "torch_chunk_gated_delta_rule", delta_rule):
        run_layer(source)

    rec.require("x", "q", "k", "v", "z2d", "beta", "g", "dq", "dk", "dv", "dz2d", "dbeta", "dg")
    shape = rec.values["q"].shape
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={
            "x": rec.values["x"],
            "w_qkv": keep(gdn.in_proj_qkv.weight),
            "w_z": keep(gdn.in_proj_z.weight),
            "w_b": keep(gdn.in_proj_b.weight),
            "w_a": keep(gdn.in_proj_a.weight),
            "conv_w": keep(gdn.conv1d.weight.squeeze(1)),
            "A_log": keep(gdn.A_log),
            "dt_bias": keep(gdn.dt_bias),
        },
        upstream={
            "dq": rec.grads["dq"], "dk": rec.grads["dk"], "dv": rec.grads["dv"],
            "dz": rec.grads["dz2d"].reshape(shape), "dbeta": rec.grads["dbeta"],
            "dg": rec.grads["dg"],
        },
        outputs={"q": rec.values["q"], "k": rec.values["k"], "v": rec.values["v"],
                 "z": rec.values["z2d"].reshape(shape), "beta": rec.values["beta"],
                 "g": rec.values["g"]},
    )


__all__ = ["SITE", "TASK", "capture"]
