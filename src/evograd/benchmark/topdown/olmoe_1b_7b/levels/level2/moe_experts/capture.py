"""One ``moe_experts`` invocation, taken from a decoder layer.

The experts module receives the token-flat hidden states and the router's
choice; its output's gradient is the upstream. The router itself stays outside
the boundary: its index and weights are captured as the site's routing inputs.
"""

from __future__ import annotations

from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, run_layer

SITE = "moe_experts"
TASK = "olmoe_moe_experts"


def capture(source: BlockSource) -> DerivedCall:
    experts = source.layer.mlp.experts
    rec = Recorder()

    def experts_pre(module, args):
        hidden_states, top_k_index, top_k_weights = args
        rec.value("x", hidden_states)
        rec.value("top_k_index", top_k_index)
        rec.value("top_k_weights", top_k_weights)

    def experts_post(module, args, output):
        rec.value("out", output)
        rec.grad_of("dout", output)

    with hooks(experts.register_forward_pre_hook(experts_pre),
               experts.register_forward_hook(experts_post)):
        run_layer(source)
    rec.require("x", "top_k_index", "top_k_weights", "out", "dout")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={"x": rec.values["x"], "gate_up_proj": keep(experts.gate_up_proj),
                "down_proj": keep(experts.down_proj), "top_k_index": rec.values["top_k_index"],
                "top_k_weights": rec.values["top_k_weights"]},
        upstream={"dout": rec.grads["dout"]},
        outputs={"out": rec.values["out"]},
    )


__all__ = ["SITE", "TASK", "capture"]
