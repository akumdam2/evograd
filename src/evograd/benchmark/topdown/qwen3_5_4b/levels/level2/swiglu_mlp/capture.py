"""One ``swiglu_mlp`` invocation, taken from a block of either kind: the MLP's input,
its three weights, its output and the gradient the closing residual add returned."""

from __future__ import annotations

from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, run_layer

SITE = "swiglu_mlp"
TASK = "qwen3_5_swiglu_mlp"


def capture(source: BlockSource) -> DerivedCall:
    mlp = source.layer.mlp
    rec = Recorder()

    def mlp_pre(module, args):
        rec.value("x", args[0])

    def mlp_post(module, args, output):
        rec.value("out", output)
        rec.grad_of("dout", output)

    with hooks(mlp.register_forward_pre_hook(mlp_pre), mlp.register_forward_hook(mlp_post)):
        run_layer(source)
    rec.require("x", "out", "dout")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={"x": rec.values["x"], "gate_weight": keep(mlp.gate_proj.weight),
                "up_weight": keep(mlp.up_proj.weight), "down_weight": keep(mlp.down_proj.weight)},
        upstream={"dout": rec.grads["dout"]},
        outputs={"out": rec.values["out"]},
    )


__all__ = ["SITE", "TASK", "capture"]
