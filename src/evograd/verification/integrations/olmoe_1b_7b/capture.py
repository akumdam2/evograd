"""OLMoE-1B-7B: what its real-training-payload capture records, and how.

One native training step of the pinned model (pretrained checkpoint, a real
WikiText-2 TRAIN batch, B=2 x T=2048, BF16, SDPA, ``grouped_mm`` experts,
the HF causal-LM loss plus the router's load-balancing loss, deterministic
settings) runs per data order with recorders attached; the shared loop is
:func:`evograd.verification.integrations.base.capture`.

* Level 1: ``nn.Linear`` (q/k/v/o share one configuration; ``lm_head``),
  every ``OlmoeRMSNorm`` (layer norms, the full-width q/k norms, the final
  norm), the module-global ``apply_rotary_pos_emb`` (q and k have identical
  shapes, so each call names its role), ``F.scaled_dot_product_attention``,
  ``fixed_cross_entropy``, and the experts' ``_apply_gate`` (the gate and up
  halves of every routed token, as the strided views the model passes).
  The router's ``F.linear`` and the experts' grouped GEMMs are not cases.
* Level 2: the attention block (``OlmoeAttention``'s own input, rotary
  tables, weights, output and output gradient), the routed experts
  (``OlmoeExperts``: tokens, routing indices and weights, both expert weight
  tensors), and every residual add an RMSNorm follows: 16 post-attention, 15
  cross-layer (a layer's closing add with the next layer's input norm) and the
  final one with ``model.norm``, 32 in all. ``dsummed`` is the sum's external
  gradient, the total gradient of the next residual sum it feeds, never
  including its own norm's; the final sum feeds none and gets zeros, recorded
  as such.

Arguments are ``detach()``-ed views of what the model passed; output
cotangents come from tensor hooks. Recorders change no arithmetic (checked
per order by the shared loop). Payloads cover Gate 1 and operator Gate 2
only: routed experts have no well-posed whole-model comparison yet.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

from ..base import Case, invocation, layer_index, shared_row
from ..qwen3_0_6b.capture import REAL_TEXT, TrainOnly  # the same text pins

PRETRAINED = "pretrained:allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5"
WORKLOAD_CONFIG = {
    "model": "olmoe_1b_7b", "device": "cuda", "dtype": "bfloat16", "batch_size": 2,
    "seq_len": 2048, "seed": 0, "data_seed": 0, "attn_implementation": "sdpa",
    "experts_implementation": "grouped_mm", "weights": PRETRAINED, "data": REAL_TEXT,
}


def load_workload():
    from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload

    return OlmoeWorkload.from_config(dict(WORKLOAD_CONFIG))


def train_batches(workload, orders):
    """The first batch of each data order over the packed TRAIN split, with identities."""
    import os

    from transformers import AutoTokenizer

    from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.model import pretrained_snapshot
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


def _grad(record: dict, key: str, tensor) -> None:
    if tensor is not None and getattr(tensor, "requires_grad", False):
        tensor.register_hook(lambda g, r=record, k=key: r.__setitem__(k, g.detach().clone()))


class LiveCapture:
    """Hooks and pass-through recorders on an unmodified ``OlmoeForCausalLM``."""

    def __init__(self, model):
        self.model = model
        self.linear: list[dict] = []
        self.rmsnorm: list[dict] = []
        self.layers: list[dict] = []
        self.attn: list[dict] = []
        self.experts: list[dict] = []
        self.gates: list[dict] = []
        self.rope: list[dict] = []
        self.sdpa: list[dict] = []
        self.ce: list[dict] = []
        self._handles: list = []
        self._restore: list[Callable[[], None]] = []

    def install(self) -> "LiveCapture":
        import torch
        import torch.nn.functional as F
        import transformers.loss.loss_utils as loss_utils
        import transformers.models.olmoe.modeling_olmoe as modeling
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeAttention, OlmoeDecoderLayer, OlmoeExperts, OlmoeRMSNorm)

        capture = self
        for path, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._module(path, module, self.linear)
            elif isinstance(module, OlmoeRMSNorm):
                self._module(path, module, self.rmsnorm, extra={"eps": module.variance_epsilon})
            elif isinstance(module, OlmoeDecoderLayer):
                self._module(path, module, self.layers, grads=False)
            elif isinstance(module, OlmoeAttention):
                self._attention(path, module)
            elif isinstance(module, OlmoeExperts):
                self._experts(path, module)

        original_rope = modeling.apply_rotary_pos_emb

        def rope_recorder(q, k, cos, sin, unsqueeze_dim=1):
            qe, ke = original_rope(q, k, cos, sin, unsqueeze_dim)
            record = {"q": q.detach(), "k": k.detach(), "cos": cos.detach(), "sin": sin.detach(),
                      "q_embed": qe.detach(), "k_embed": ke.detach()}
            for key, tensor in (("dq_embed", qe), ("dk_embed", ke)):
                _grad(record, key, tensor)
            capture.rope.append(record)
            return qe, ke

        modeling.apply_rotary_pos_emb = rope_recorder
        self._restore.append(lambda: setattr(modeling, "apply_rotary_pos_emb", original_rope))

        original_gate = OlmoeExperts._apply_gate

        def gate_recorder(experts, gate_up_out):
            out = original_gate(experts, gate_up_out)
            gate, up = gate_up_out.chunk(2, dim=-1)
            record = {"a": gate.detach(), "b": up.detach(), "c": out.detach()}
            _grad(record, "dc", out)
            capture.gates.append(record)
            return out

        OlmoeExperts._apply_gate = gate_recorder
        self._restore.append(lambda: setattr(OlmoeExperts, "_apply_gate", original_gate))

        original_sdpa = F.scaled_dot_product_attention

        def sdpa_recorder(query, key, value, *args, **kwargs):
            out = original_sdpa(query, key, value, *args, **kwargs)
            record = {"q": query.detach(), "k": key.detach(), "v": value.detach(), "o": out.detach(),
                      "attn_mask_is_none": kwargs.get("attn_mask", args[0] if args else None) is None}
            for name, tensor in (("dq", query), ("dk", key), ("dv", value), ("do", out)):
                _grad(record, name, tensor)
            capture.sdpa.append(record)
            return out

        F.scaled_dot_product_attention = sdpa_recorder
        self._restore.append(lambda: setattr(F, "scaled_dot_product_attention", original_sdpa))

        original_ce = loss_utils.fixed_cross_entropy

        def ce_recorder(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
            loss = original_ce(source, target, num_items_in_batch, ignore_index, **kwargs)
            # A copy: OlmoeForCausalLM adds the router's load-balancing loss to
            # this tensor in place (``loss += coef * aux_loss``) after the call.
            record = {"source": source.detach(), "target": target.detach(),
                      "num_items_in_batch": num_items_in_batch, "ignore_index": ignore_index,
                      "loss": loss.detach().clone()}
            _grad(record, "dloss", loss)
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
                _grad(record, "dx", x)

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            y = output[0] if isinstance(output, tuple) else output
            record["y"] = y.detach()
            if grads:
                _grad(record, "dy", y)
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
            _grad(record, "dy", output[0])
            self.attn.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _experts(self, path, module):
        state: dict = {}

        def pre(mod, args, kwargs):
            names = ("hidden_states", "top_k_index", "top_k_weights")
            values = dict(zip(names, args)) | {k: kwargs[k] for k in names if k in kwargs}
            state["record"] = {"path": path, **{k: v.detach() for k, v in values.items()}}

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            record["out"] = output.detach()
            _grad(record, "dout", output)
            self.experts.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        for restore in reversed(self._restore):
            restore()
        self._handles.clear()
        self._restore.clear()


def invocations(capture: LiveCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 boundary of one step, in its case's task ABI."""
    import torch

    modules = dict(model.named_modules())

    def emit(*parts):
        return invocation(cases, *parts)

    for r in capture.linear:
        module = modules[r["path"]]
        x, y, dy = r["x"], r["y"], r["dy"]
        k, n = x.shape[-1], y.shape[-1]
        yield emit("linear_no_bias", 1,
                   {"path": r["path"], "layer": layer_index(r["path"]), "role": r["path"].rsplit(".", 1)[-1],
                    "kind": "module:nn.Linear"},
                   {"x": x.reshape(-1, k), "weight": module.weight},
                   {"dy": dy.reshape(-1, n)}, {"y": y.reshape(-1, n)})

    for r in capture.rmsnorm:
        module = modules[r["path"]]
        x, y, dy, h = r["x"], r["y"], r["dy"], r["x"].shape[-1]
        yield emit("rmsnorm", 1,
                   {"path": r["path"], "layer": layer_index(r["path"]),
                    "role": "final_norm" if r["path"] == "model.norm" else r["path"].rsplit(".", 1)[-1],
                    "kind": "module:OlmoeRMSNorm"},
                   {"x": x.reshape(-1, h), "weight": module.weight, "eps": r["eps"]},
                   {"dy": dy.reshape(-1, h)}, {"y": y.reshape(-1, h)})

    for layer, r in enumerate(capture.rope):
        cos, sin = shared_row(r["cos"])[0], shared_row(r["sin"])[0]
        for name, component, x, out, dout in (("q", "rope", r["q"], r["q_embed"], r["dq_embed"]),
                                              ("k", "rope_kv", r["k"], r["k_embed"], r["dk_embed"])):
            yield emit("rope", 1,
                       {"path": f"model.layers.{layer}.self_attn#rope_{name}", "layer": layer,
                        "role": f"rope_{name}", "component": component,
                        "kind": "function:apply_rotary_pos_emb"},
                       {"x": x, "cos": cos, "sin": sin}, {"dy": dout}, {"y": out},
                       {"cos_sin": "first row of the model's [B, T, D] table (rows verified identical)"})

    for layer, r in enumerate(capture.gates):
        yield emit("swiglu", 1,
                   {"path": f"model.layers.{layer}.mlp.experts#apply_gate", "layer": layer,
                    "role": "moe_activation", "kind": "method:OlmoeExperts._apply_gate"},
                   {"a": r["a"], "b": r["b"]}, {"dc": r["dc"]}, {"c": r["c"]},
                   {"layout": "gate and up halves of the grouped gate_up output (strided views)"})

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
                   {"target": "labels as ForCausalLMLoss passes them (already shifted once)"})

    for r in capture.attn:
        attn, layer = modules[r["path"]], layer_index(r["path"])
        # The block is declared mask-free (is_causal) with one eps for both norms.
        if not (capture.sdpa[layer]["attn_mask_is_none"]
                and attn.q_norm.variance_epsilon == attn.k_norm.variance_epsilon):
            raise ValueError(f"layer {layer}: the attention call is outside the declared case")
        yield emit("olmoe_attention_block", 2,
                   {"path": f"model.layers.{layer}.self_attn#attention_block", "layer": layer,
                    "role": "attention_block", "kind": "boundary:OlmoeAttention"},
                   {"x": r["x"], "q_weight": attn.q_proj.weight, "k_weight": attn.k_proj.weight,
                    "v_weight": attn.v_proj.weight, "q_norm_weight": attn.q_norm.weight,
                    "k_norm_weight": attn.k_norm.weight, "o_weight": attn.o_proj.weight,
                    "cos": shared_row(r["cos"]), "sin": shared_row(r["sin"]),
                    "eps": attn.q_norm.variance_epsilon},
                   {"dout": r["dy"]}, {"out": r["y"]},
                   {"cos_sin": "first row of the model's [B, T, D] table (rows verified identical)"})

    for r in capture.experts:
        module, layer = modules[r["path"]], layer_index(r["path"])
        yield emit("olmoe_moe_experts", 2,
                   {"path": f"model.layers.{layer}.mlp.experts#experts", "layer": layer,
                    "role": "moe_experts", "kind": "boundary:OlmoeExperts"},
                   {"x": r["hidden_states"], "gate_up_proj": module.gate_up_proj,
                    "down_proj": module.down_proj, "top_k_index": r["top_k_index"],
                    "top_k_weights": r["top_k_weights"]},
                   {"dout": r["dout"]}, {"out": r["out"]},
                   {"routing": "the router's own top-k indices and weights of this step"})

    norms = {r["path"]: r for r in capture.rmsnorm}
    by_path = {r["path"]: r for r in (*capture.layers, *capture.attn, *capture.experts)}
    n_layers = len(capture.layers)
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        post = norms[f"{prefix}.post_attention_layernorm"]
        nxt = (norms[f"model.layers.{layer + 1}.input_layernorm"] if layer + 1 < n_layers
               else norms["model.norm"])
        # The MoE block's output: the experts' [T, H] output reshaped to [B, T, H], as
        # OlmoeSparseMoeBlock returns it.
        moe = by_path[f"{prefix}.mlp.experts"]["out"].reshape(post["x"].shape)
        sites = [(f"{prefix}#residual_post_attention", "within_layer",
                  by_path[f"{prefix}.self_attn"]["y"], by_path[prefix]["x"], post, nxt["dx"])]
        if layer + 1 < n_layers:
            sites.append((f"{prefix}#residual_mlp_to_next", "cross_layer", moe,
                          post["x"], nxt,
                          norms[f"model.layers.{layer + 1}.post_attention_layernorm"]["dx"]))
        else:
            sites.append((f"{prefix}#residual_mlp_to_final", "final_model_norm", moe,
                          post["x"], nxt, None))
        for call, role, branch, residual, norm, external in sites:
            yield emit("olmoe_residual_rmsnorm", 2,
                       {"path": call, "layer": layer, "role": role, "norm": norm["path"],
                        "kind": "boundary:residual add + OlmoeRMSNorm"},
                       {"branch": branch, "residual": residual,
                        "weight": modules[norm["path"]].weight, "eps": norm["eps"]},
                       {"dnormed": norm["dy"],
                        "dsummed": external if external is not None else
                        torch.zeros_like(norm["x"])},
                       {"normed": norm["y"], "summed": norm["x"]},
                       {"dsummed_origin": "total gradient of the next residual sum"}
                       if external is not None else
                       {"dsummed_origin": "absent in the model (the final sum has no other "
                                          "consumer); zeros supplied for the declared "
                                          "two-output ABI",
                        "absent_cotangents": ["dsummed"]})
