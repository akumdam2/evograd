"""One ``residual_rmsnorm`` invocation, taken from a block of either kind.

``residual`` is the layer's input, ``branch`` the token mixer's output, and the
two outputs are ``post_attention_layernorm``'s input (``summed``) and output
(``normed``). ``dsummed`` is the gradient of the layer's output -- the closing
``summed + mlp`` add delivers exactly that to ``summed`` -- and ``dnormed`` the
gradient the MLP returned.
"""

from __future__ import annotations

from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, run_layer

SITE = "residual_rmsnorm"
TASK = "qwen3_5_residual_rmsnorm"


def capture(source: BlockSource) -> DerivedCall:
    layer = source.layer
    mixer = getattr(layer, "linear_attn", None) or layer.self_attn
    norm = layer.post_attention_layernorm
    rec = Recorder()

    def layer_pre(module, args, kwargs):
        rec.value("residual", args[0] if args else kwargs["hidden_states"])

    def layer_post(module, args, kwargs, output):
        rec.grad_of("dsummed", output)

    def mixer_post(module, args, kwargs, output):
        rec.value("branch", output[0] if isinstance(output, tuple) else output)

    def norm_pre(module, args):
        rec.value("summed", args[0])

    def norm_post(module, args, output):
        rec.value("normed", output)
        rec.grad_of("dnormed", output)

    with hooks(layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
               layer.register_forward_hook(layer_post, with_kwargs=True),
               mixer.register_forward_hook(mixer_post, with_kwargs=True),
               norm.register_forward_pre_hook(norm_pre),
               norm.register_forward_hook(norm_post)):
        run_layer(source)
    rec.require("residual", "branch", "summed", "normed", "dsummed", "dnormed")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={"branch": rec.values["branch"], "residual": rec.values["residual"],
                "weight": keep(norm.weight), "eps": float(norm.eps)},
        upstream={"dnormed": rec.grads["dnormed"], "dsummed": rec.grads["dsummed"]},
        outputs={"normed": rec.values["normed"], "summed": rec.values["summed"]},
    )


__all__ = ["SITE", "TASK", "capture"]
