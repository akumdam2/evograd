"""Llama-3.2-1B: what its real-training-payload capture records, and how.

One native training step of the pinned model (pretrained checkpoint, a real
WikiText-2 TRAIN batch, B=2 x T=2048, BF16, SDPA, train mode, the HF causal-LM
loss, deterministic settings) runs per data order with recorders attached; the
shared loop is :func:`evograd.verification.integrations.base.capture`. Every
Level-1 call and Level-2 boundary of the step becomes one payload of the
observed case it belongs to. This is Qwen3-0.6B's capture without the per-head
q/k norms; ``q_proj`` and ``o_proj`` share one linear configuration.

* Arguments are ``detach()``-ed views of what the model passed (exact storage,
  strides and offset). 2-D task ABIs receive ``reshape(-1, k)`` of them.
* Output cotangents come from tensor hooks on the call's outputs. A hook on a
  non-leaf output fires once with the total gradient of that output, which is
  the call's output cotangent. Gradients are cloned in the hook (strides kept).
* Input gradients recorded by hooks are model-level totals, so they are not
  stored; the gates recompute each payload's local VJP from the reference.
* Residual fusion sites (``llama3_residual_rmsnorm``): 16 within-layer, 15
  cross-layer and the final norm's sum, 32 in all. The final sum has no
  consumer; its cotangent is supplied as zeros and recorded as such.
* CE: ``fixed_cross_entropy``'s own ``(source, target)``, labels already
  shifted by ``ForCausalLMLoss``. ``lm_head.weight`` is the tied embedding.
* Recorders change no arithmetic (checked per order by the shared loop).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator, Sequence

from ..base import Case, invocation, layer_index, shared_row

from ..qwen3_0_6b.capture import REAL_TEXT, TrainOnly  # the same text pins

#: The pinned checkpoint (gated; read from ``$EVOGRAD_LLAMA_HF_CACHE``).
PRETRAINED = "pretrained:meta-llama/Llama-3.2-1B@4e20de362430cd3b72f300e6b0f18e50e7166e08"
WORKLOAD_CONFIG = {
    "model": "llama_3_2_1b", "device": "cuda", "dtype": "bfloat16", "batch_size": 2,
    "seq_len": 2048, "seed": 0, "data_seed": 0, "attn_implementation": "sdpa",
    "weights": PRETRAINED, "data": REAL_TEXT,
}


def load_workload():
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    return Llama3Workload.from_config(dict(WORKLOAD_CONFIG))


def train_batches(workload, orders) -> tuple[dict[int, tuple[Any, Any]], dict[str, Any]]:
    """The first batch of each data order over the packed TRAIN split, with identities."""
    from transformers import AutoTokenizer

    from evograd.benchmark.topdown.llama3_2_1b.levels.level3.model import pretrained_snapshot
    from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata
    from evograd.verification.common.tensors import tensor_digest

    with TrainOnly() as guard:
        snapshot = pretrained_snapshot(workload.spec)
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        packed = textdata.pack_split("train", tokenizer, block_len=workload.seq_len,
                                     cache_dir=textdata.hf_cache_dir())
    out, per_order = {}, {}
    for order in orders:
        batches = textdata.TextBatches(packed, batch_size=workload.batch_size, seed=order, device="cpu")
        ids, labels = batches.batch(0)
        out[order] = (ids, labels)
        per_order[str(order)] = {"order": order, "input_ids_sha256": tensor_digest(ids),
                                 "labels_sha256": tensor_digest(labels), "shape": list(ids.shape),
                                 "rule": "TextBatches(train, batch_size, seed=order).batch(0)",
                                 "order_sha256": batches.describe()["order_sha256"]}
    identity = {"dataset": textdata.DATASET_REPO, "config": textdata.DATASET_CONFIG,
                "revision": textdata.DATASET_REVISION, "split": "train",
                "tokenizer_sha256": textdata.file_sha256(os.path.join(snapshot, "tokenizer.json")),
                "block_len": workload.seq_len, "batch_size": workload.batch_size,
                "packing": packed.describe(), "labels": "labels = input_ids (shifted once by the HF loss)",
                "splits_requested": sorted(set(guard.requested)), "batches": per_order}
    return out, identity


class LiveCapture:
    """Hooks and pass-through recorders on an unmodified ``LlamaForCausalLM``.

    Module forward hooks and tensor gradient hooks change no arithmetic; the
    three function recorders (``apply_rotary_pos_emb``,
    ``F.scaled_dot_product_attention``, ``fixed_cross_entropy``) call the
    original function with the original arguments and return its result.
    """

    def __init__(self, model):
        self.model = model
        self.linear: list[dict] = []
        self.rmsnorm: list[dict] = []
        self.layers: list[dict] = []
        self.attn: list[dict] = []
        self.mlp: list[dict] = []
        self.rope: list[dict] = []
        self.sdpa: list[dict] = []
        self.ce: list[dict] = []
        self._handles: list = []
        self._restore: list[Callable[[], None]] = []

    @staticmethod
    def _grad(record: dict, key: str, tensor) -> None:
        if tensor is not None and getattr(tensor, "requires_grad", False):
            tensor.register_hook(lambda g, r=record, k=key: r.__setitem__(k, g.detach().clone()))

    def install(self) -> "LiveCapture":
        import torch
        import torch.nn.functional as F
        import transformers.loss.loss_utils as loss_utils
        import transformers.models.llama.modeling_llama as modeling
        from transformers.models.llama.modeling_llama import (
            LlamaAttention, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm)

        for path, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._module(path, module, self.linear)
            elif isinstance(module, LlamaRMSNorm):
                self._module(path, module, self.rmsnorm, extra={"eps": module.variance_epsilon})
            elif isinstance(module, LlamaDecoderLayer):
                self._module(path, module, self.layers, grads=False)
            elif isinstance(module, LlamaAttention):
                self._attention(path, module)
            elif isinstance(module, LlamaMLP):
                self._module(path, module, self.mlp)

        capture = self
        original_rope = modeling.apply_rotary_pos_emb

        def rope_recorder(q, k, cos, sin, unsqueeze_dim=1):
            qe, ke = original_rope(q, k, cos, sin, unsqueeze_dim)
            record = {"q": q.detach(), "k": k.detach(), "cos": cos.detach(), "sin": sin.detach(),
                      "q_embed": qe.detach(), "k_embed": ke.detach()}
            for key, tensor in (("dq_embed", qe), ("dk_embed", ke)):
                capture._grad(record, key, tensor)
            capture.rope.append(record)
            return qe, ke

        modeling.apply_rotary_pos_emb = rope_recorder
        self._restore.append(lambda: setattr(modeling, "apply_rotary_pos_emb", original_rope))

        original_sdpa = F.scaled_dot_product_attention

        def sdpa_recorder(query, key, value, *args, **kwargs):
            out = original_sdpa(query, key, value, *args, **kwargs)
            record = {"q": query.detach(), "k": key.detach(), "v": value.detach(), "o": out.detach(),
                      "attn_mask_is_none": kwargs.get("attn_mask", args[0] if args else None) is None}
            for name, tensor in (("dq", query), ("dk", key), ("dv", value), ("do", out)):
                capture._grad(record, name, tensor)
            capture.sdpa.append(record)
            return out

        F.scaled_dot_product_attention = sdpa_recorder
        self._restore.append(lambda: setattr(F, "scaled_dot_product_attention", original_sdpa))

        original_ce = loss_utils.fixed_cross_entropy

        def ce_recorder(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
            loss = original_ce(source, target, num_items_in_batch, ignore_index, **kwargs)
            record = {"source": source.detach(), "target": target.detach(),
                      "num_items_in_batch": num_items_in_batch, "ignore_index": ignore_index,
                      "loss": loss.detach()}
            capture._grad(record, "dloss", loss)
            capture.ce.append(record)
            return loss

        loss_utils.fixed_cross_entropy = ce_recorder
        self._restore.append(lambda: setattr(loss_utils, "fixed_cross_entropy", original_ce))
        return self

    def _module(self, path, module, sink, *, extra=None, grads=True):
        state: dict = {}

        def pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            state["record"] = record = {"path": path, "x": x.detach(), **(extra or {})}
            if grads:
                self._grad(record, "dx", x)

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            y = output[0] if isinstance(output, tuple) else output
            record["y"] = y.detach()
            if grads:
                self._grad(record, "dy", y)
            sink.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _attention(self, path, module):
        state: dict = {}

        def pre(mod, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            state["record"] = {"path": path, "x": hidden.detach(), "cos": cos.detach(),
                               "sin": sin.detach()}

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            record["y"] = output[0].detach()
            self._grad(record, "dy", output[0])
            self.attn.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        for restore in reversed(self._restore):
            restore()
        self._handles.clear()
        self._restore.clear()


# ── calls -> payloads in each task's ABI ─────────────────────────────────────


def invocations(capture: LiveCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 boundary of one step, in its case's task ABI."""
    import torch

    modules = dict(model.named_modules())
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    linear_by_path = {r["path"]: r for r in capture.linear}

    def emit(*parts):
        return invocation(cases, *parts)

    for r in capture.linear:
        module = modules[r["path"]]
        x, y, dy = r["x"], r["y"], r["dy"]
        k, n = x.shape[-1], y.shape[-1]
        notes = {"tied_weight": "model.embed_tokens.weight"} if r["path"] == "lm_head" and tied else {}
        yield emit("linear_no_bias", 1,
                   {"path": r["path"], "layer": layer_index(r["path"]), "role": r["path"].rsplit(".", 1)[-1],
                    "kind": "module:nn.Linear"},
                   {"x": x.reshape(-1, k), "weight": module.weight},
                   {"dy": dy.reshape(-1, n)}, {"y": y.reshape(-1, n)}, notes)

    for r in capture.rmsnorm:
        module = modules[r["path"]]
        x, y, dy, h = r["x"], r["y"], r["dy"], r["x"].shape[-1]
        role = ("first_input_norm" if r["path"] == "model.layers.0.input_layernorm"
                else "final_norm" if r["path"] == "model.norm" else r["path"].rsplit(".", 1)[-1])
        yield emit("rmsnorm", 1,
                   {"path": r["path"], "layer": layer_index(r["path"]), "role": role,
                    "kind": "module:LlamaRMSNorm"},
                   {"x": x.reshape(-1, h), "weight": module.weight, "eps": r["eps"]},
                   {"dy": dy.reshape(-1, h)}, {"y": y.reshape(-1, h)})

    for layer, r in enumerate(capture.rope):
        cos, sin = shared_row(r["cos"])[0], shared_row(r["sin"])[0]
        for name, x, out, dout in (("q", r["q"], r["q_embed"], r["dq_embed"]),
                                   ("k", r["k"], r["k_embed"], r["dk_embed"])):
            yield emit("rope", 1,
                       {"path": f"model.layers.{layer}.self_attn#rope_{name}", "layer": layer,
                        "role": f"rope_{name}", "kind": "function:apply_rotary_pos_emb"},
                       {"x": x, "cos": cos, "sin": sin}, {"dy": dout}, {"y": out},
                       {"cos_sin": "first row of the model's [B, T, D] table (rows verified identical)"})

    for layer in range(len(capture.mlp)):
        prefix = f"model.layers.{layer}.mlp"
        gate, up, down = (linear_by_path[f"{prefix}.{n}"] for n in ("gate_proj", "up_proj", "down_proj"))
        cols = gate["y"].shape[-1]
        yield emit("swiglu", 1,
                   {"path": f"{prefix}#act_mul", "layer": layer, "role": "act_mul",
                    "kind": "inline:act_fn(gate)*up"},
                   {"a": gate["y"].reshape(-1, cols), "b": up["y"].reshape(-1, cols)},
                   {"dc": down["dx"].reshape(-1, cols)}, {"c": down["x"].reshape(-1, cols)},
                   {"dc": "total gradient of down_proj's input (its only consumer is down_proj)"})

    for layer, r in enumerate(capture.sdpa):
        yield emit("causal_gqa_attention", 1,
                   {"path": f"model.layers.{layer}.self_attn#sdpa", "layer": layer, "role": "sdpa",
                    "kind": "function:F.scaled_dot_product_attention"},
                   {"q": r["q"], "k": r["k"], "v": r["v"]}, {"do": r["do"]}, {"o": r["o"]},
                   {"attn_mask_is_none": r["attn_mask_is_none"]})

    for r in capture.ce:
        if r["num_items_in_batch"] is not None or r["ignore_index"] != -100:
            raise ValueError("the CE call is outside the declared case")
        yield emit("cross_entropy", 1,
                   {"path": "loss_function#fixed_cross_entropy", "layer": None, "role": "loss",
                    "kind": "function:fixed_cross_entropy"},
                   {"logits": r["source"], "target": r["target"]},
                   {"dloss": r["dloss"]}, {"loss": r["loss"]},
                   {"target": "labels as ForCausalLMLoss passes them (already shifted once)",
                    "ignored_rows": int((r["target"] == -100).sum())})

    for layer, r in enumerate(capture.attn):
        attn, sdpa = modules[r["path"]], capture.sdpa[layer]
        yield emit("llama3_qkv_rope", 2,
                   {"path": f"model.layers.{layer}.self_attn#qkv_rope", "layer": layer,
                    "role": "qkv_rope", "kind": "boundary:LlamaAttention[:3]"},
                   {"x": r["x"], "q_weight": attn.q_proj.weight, "k_weight": attn.k_proj.weight,
                    "v_weight": attn.v_proj.weight, "cos": shared_row(r["cos"]),
                    "sin": shared_row(r["sin"])},
                   {"dq": sdpa["dq"], "dk": sdpa["dk"], "dv": sdpa["dv"]},
                   {"q": sdpa["q"], "k": sdpa["k"], "v": sdpa["v"]},
                   {"cotangents": "SDPA input gradients (head-major views, as the model delivers them)"})
        yield emit("llama3_attention", 2,
                   {"path": f"model.layers.{layer}.self_attn#attention", "layer": layer,
                    "role": "attention", "kind": "boundary:sdpa+o_proj"},
                   {"q": sdpa["q"], "k": sdpa["k"], "v": sdpa["v"], "o_weight": attn.o_proj.weight},
                   {"dout": r["dy"]}, {"out": r["y"]})

    for r in capture.mlp:
        mlp, layer = modules[r["path"]], layer_index(r["path"])
        yield emit("llama3_swiglu_mlp", 2,
                   {"path": f"model.layers.{layer}.mlp#swiglu_mlp", "layer": layer,
                    "role": "swiglu_mlp", "kind": "boundary:LlamaMLP"},
                   {"x": r["x"], "gate_weight": mlp.gate_proj.weight, "up_weight": mlp.up_proj.weight,
                    "down_weight": mlp.down_proj.weight},
                   {"dout": r["dy"]}, {"out": r["y"]})

    norms = {r["path"]: r for r in capture.rmsnorm}
    n_layers = len(capture.layers)
    for layer in range(n_layers):
        attn, mlp = capture.attn[layer], capture.mlp[layer]
        post = norms[f"model.layers.{layer}.post_attention_layernorm"]
        nxt = (norms[f"model.layers.{layer + 1}.input_layernorm"] if layer + 1 < n_layers
               else norms["model.norm"])
        sites = [(f"model.layers.{layer}#residual_post_attention", "within_layer", attn["y"],
                  capture.layers[layer]["x"], post, nxt["dx"],
                  modules[f"model.layers.{layer}.post_attention_layernorm"], "next residual sum")]
        if layer + 1 < n_layers:
            sites.append((f"model.layers.{layer}#residual_mlp_to_next", "cross_layer", mlp["y"],
                          post["x"], nxt, norms[f"model.layers.{layer + 1}.post_attention_layernorm"]["dx"],
                          modules[f"model.layers.{layer + 1}.input_layernorm"], "next residual sum"))
        else:
            sites.append((f"model.layers.{layer}#residual_mlp_to_final", "final_model_norm", mlp["y"],
                          post["x"], nxt, None, modules["model.norm"], None))
        for call, role, x, r_in, norm, external, module, origin in sites:
            cols = x.shape[-1]
            dsummed = external.reshape(-1, cols) if external is not None else \
                torch.zeros_like(norm["x"].reshape(-1, cols))
            yield emit("llama3_residual_rmsnorm", 2,
                       {"path": call, "layer": layer, "role": role, "norm": norm["path"],
                        "kind": "boundary:residual add + LlamaRMSNorm"},
                       {"x": x.reshape(-1, cols), "r": r_in.reshape(-1, cols), "weight": module.weight,
                        "eps": norm["eps"]},
                       {"dout": norm["dy"].reshape(-1, cols), "dsummed": dsummed},
                       {"out": norm["y"].reshape(-1, cols), "summed": norm["x"].reshape(-1, cols)},
                       {"dsummed_origin": "total gradient of the " + origin} if origin else
                       {"dsummed_origin": "absent in the model (the final sum has no consumer); "
                                          "zeros supplied for the declared two-output ABI",
                        "absent_cotangents": ["dsummed"]})
