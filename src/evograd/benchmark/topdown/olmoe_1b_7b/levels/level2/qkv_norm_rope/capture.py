"""One ``qkv_norm_rope`` invocation, taken from a decoder layer.

The input is ``self_attn``'s hidden state and the rotary tables it was handed;
the outputs are what reached SDPA (q and k from ``apply_rotary_pos_emb``, v at
the SDPA call).
"""

from __future__ import annotations

import torch

from .. import hf_modules
from ..capture_tools import BlockSource, DerivedCall, Recorder, hooks, keep, patched, run_layer

SITE = "qkv_norm_rope"
TASK = "olmoe_qkv_norm_rope"


def _one_row(table: torch.Tensor) -> torch.Tensor:
    """The declared ``[1, T, D]`` table: a batch's identical rows reduced to one
    (exact -- an unpadded batch has one ``position_ids`` row repeated)."""
    if table.shape[0] == 1:
        return table
    if not bool((table == table[:1]).all()):
        raise RuntimeError("the rotary tables differ across the batch; the declared "
                           "[1, T, D] cos/sin cannot represent them")
    return table[:1]


def capture(source: BlockSource) -> DerivedCall:
    attn = source.layer.self_attn
    if getattr(attn.config, "clip_qkv", None) is not None:
        raise RuntimeError("the qkv_norm_rope site is declared without clip_qkv")
    rec = Recorder()

    def attn_pre(module, args, kwargs):
        rec.value("x", kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])
        cos, sin = kwargs["position_embeddings"]
        rec.value("cos", _one_row(cos))
        rec.value("sin", _one_row(sin))

    def rope(original):
        def wrapper(q, k, cos, sin, *args, **kwargs):
            q_out, k_out = original(q, k, cos, sin, *args, **kwargs)
            for name, tensor in (("q", q_out), ("k", k_out)):
                rec.value(name, tensor)
                rec.grad_of("d" + name, tensor)
            return q_out, k_out
        return wrapper

    def sdpa(original):
        def wrapper(query, key, value, *args, **kwargs):
            rec.value("v", value)
            rec.grad_of("dv", value)
            return original(query, key, value, *args, **kwargs)
        return wrapper

    with hooks(attn.register_forward_pre_hook(attn_pre, with_kwargs=True)), \
            patched(hf_modules.modeling, "apply_rotary_pos_emb", rope), \
            patched(torch.nn.functional, "scaled_dot_product_attention", sdpa):
        run_layer(source)
    rec.require("x", "cos", "sin", "q", "k", "v", "dq", "dk", "dv")
    return DerivedCall(
        site=SITE, task=TASK, layer_index=source.layer_index, source=source.source,
        inputs={
            "x": rec.values["x"],
            "q_weight": keep(attn.q_proj.weight), "k_weight": keep(attn.k_proj.weight),
            "v_weight": keep(attn.v_proj.weight),
            "q_norm_weight": keep(attn.q_norm.weight), "k_norm_weight": keep(attn.k_norm.weight),
            "cos": rec.values["cos"], "sin": rec.values["sin"],
            "eps": float(attn.q_norm.variance_epsilon),
        },
        upstream={name: rec.grads[name] for name in ("dq", "dk", "dv")},
        outputs={name: rec.values[name] for name in ("q", "k", "v")},
    )


__all__ = ["SITE", "TASK", "capture"]
