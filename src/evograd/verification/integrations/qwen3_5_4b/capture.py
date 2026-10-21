"""Qwen3.5-4B: what its real-training-payload capture records, and how.

One native training step of the pinned text model (``Qwen3_5ForCausalLM`` from
the published checkpoint, a real WikiText-2 TRAIN batch, B=1 x T=2048, BF16,
SDPA, the torch Gated DeltaNet path, train mode, the HF causal-LM loss,
deterministic settings) runs per data order with recorders attached; the
shared loop is :func:`evograd.verification.integrations.base.capture`.

* Module hooks: every ``nn.Linear``, every ``Qwen3_5RMSNorm`` (Gemma ``1 + w``),
  the gated norm, the decoder layers, both token mixers and the MLP.
* Function recorders, each calling the original with the original arguments:
  ``causal_conv1d_fn``, ``l2norm`` (inside the delta rule, q then k),
  ``torch_chunk_gated_delta_rule``, ``apply_rotary_pos_emb``,
  ``F.scaled_dot_product_attention`` and ``fixed_cross_entropy``. A function
  call is attributed to the token mixer running at the time (``where``).
* Values the model only forms inline are taken where they enter or leave a
  recorded call: ``swiglu`` from gate/up outputs and down_proj's input;
  ``gdn_gating``'s ``a``/``b`` from ``in_proj_a``/``in_proj_b`` and its ``g``/``beta``
  from the delta rule's arguments (their only consumer); the attention
  ``gate`` and its cotangent from the gate half of ``q_proj``'s output (pure
  data movement, the same values as the model's reshape); the residual add
  from the layer input, the mixer output and ``post_attention_layernorm``.
* Output cotangents come from tensor hooks (total gradient of the tensor);
  no cotangent is absent in this model.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

from ..base import Case, invocation, layer_index, shared_row

PRETRAINED = "pretrained:Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REAL_TEXT = "wikitext-2-raw:b08601e04326c79dfdd32d625aee71d232d685c3"
WORKLOAD_CONFIG = {
    "model": "qwen3_5_4b", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
    "seq_len": 2048, "seed": 0, "data_seed": 0, "attn_implementation": "sdpa",
    "weights": PRETRAINED, "data": REAL_TEXT,
}


def load_workload():
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

    return Qwen3_5Workload.from_config(dict(WORKLOAD_CONFIG))


def train_batches(workload, orders) -> tuple[dict[int, tuple[Any, Any]], dict[str, Any]]:
    """The first TRAIN batch of each data order, with its identity."""
    from evograd.verification.common.tensors import tensor_digest

    out, per_order = {}, {}
    for order in orders:
        batches = workload.train_batches(order)
        ids, labels = batches.batch(0)
        out[order] = (ids, labels)
        per_order[str(order)] = {"order": order, "input_ids_sha256": tensor_digest(ids),
                                 "labels_sha256": tensor_digest(labels), "shape": list(ids.shape),
                                 "rule": "workload.train_batches(order).batch(0)",
                                 "order_sha256": batches.describe()["order_sha256"]}
    return out, {**workload.data_identity(), "batches": per_order}


class LiveCapture:
    """Hooks and pass-through recorders on an unmodified ``Qwen3_5ForCausalLM``."""

    def __init__(self, model):
        self.model = model
        self.linear: list[dict] = []
        self.norm: list[dict] = []
        self.gated_norm: list[dict] = []
        self.layers: list[dict] = []
        self.gdn: list[dict] = []
        self.attn: list[dict] = []
        self.mlp: list[dict] = []
        self.conv: list[dict] = []
        self.l2norm: list[dict] = []
        self.delta: list[dict] = []
        self.rope: list[dict] = []
        self.sdpa: list[dict] = []
        self.ce: list[dict] = []
        #: The token mixer running now; function calls are attributed to it.
        self.where: str | None = None
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

        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import modeling

        M = modeling()
        for path, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._module(path, module, self.linear, dx=path.endswith("mlp.down_proj"))
            elif isinstance(module, M.Qwen3_5RMSNorm):
                self._module(path, module, self.norm, extra={"eps": module.eps})
            elif isinstance(module, M.Qwen3_5RMSNormGated):
                self._gated_norm(path, module)
            elif isinstance(module, M.Qwen3_5DecoderLayer):
                self._module(path, module, self.layers, dx=False, dy=False)
            elif isinstance(module, M.Qwen3_5GatedDeltaNet):
                self._mixer(path, module, self.gdn)
            elif isinstance(module, M.Qwen3_5Attention):
                self._mixer(path, module, self.attn)
            elif isinstance(module, M.Qwen3_5MLP):
                self._module(path, module, self.mlp, dx=False)
        self._patch(M, "causal_conv1d_fn", self._conv)
        self._patch(M, "l2norm", self._l2norm)
        self._patch(M, "torch_chunk_gated_delta_rule", self._delta)
        self._patch(M, "apply_rotary_pos_emb", self._rope)
        self._patch(F, "scaled_dot_product_attention", self._sdpa)
        self._patch(loss_utils, "fixed_cross_entropy", self._ce)
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        for restore in reversed(self._restore):
            restore()
        self._handles.clear()
        self._restore.clear()

    # ── modules ──────────────────────────────────────────────────────────

    def _module(self, path, module, sink, *, dx=True, dy=True, extra=None):
        state: dict = {}

        def pre(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            state["record"] = record = {"path": path, "x": x.detach(), **(extra or {})}
            if dx:
                self._grad(record, "dx", x)

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            y = output[0] if isinstance(output, tuple) else output
            record["y"] = y.detach()
            if dy:
                self._grad(record, "dy", y)
            sink.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _gated_norm(self, path, module):
        state: dict = {}

        def pre(mod, args, kwargs):
            x, z = args[0], args[1]
            state["record"] = record = {"path": path, "x": x.detach(), "z": z.detach(),
                                        "eps": module.variance_epsilon}
            self._grad(record, "dz", z)

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            record["y"] = output.detach()
            self._grad(record, "dy", output)
            self.gated_norm.append(record)

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _mixer(self, path, module, sink):
        state: dict = {}

        def pre(mod, args, kwargs):
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            record = {"path": path, "x": x.detach()}
            if "position_embeddings" in kwargs:
                record["cos"], record["sin"] = (t.detach() for t in kwargs["position_embeddings"])
            state["record"] = record
            self.where = path

        def post(mod, args, kwargs, output):
            record = state.pop("record")
            y = output[0] if isinstance(output, tuple) else output
            record["y"] = y.detach()
            self._grad(record, "dy", y)
            sink.append(record)
            self.where = None

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    # ── functions ────────────────────────────────────────────────────────

    def _patch(self, owner, name: str, make: Callable) -> None:
        original = getattr(owner, name)
        setattr(owner, name, make(original))
        self._restore.append(lambda: setattr(owner, name, original))

    def _conv(self, original):
        def conv(x, weight, bias=None, activation=None, **kwargs):
            y = original(x, weight, bias, activation=activation, **kwargs)
            record = {"path": self.where, "x": x.detach(), "weight": weight.detach(),
                      "bias_is_none": bias is None, "activation": activation, "y": y.detach()}
            self._grad(record, "dy", y)
            self.conv.append(record)
            return y
        return conv

    def _l2norm(self, original):
        def l2norm(x, dim=-1, eps=1e-6):
            y = original(x, dim=dim, eps=eps)
            record = {"path": self.where, "x": x.detach(), "dim": dim, "eps": eps, "y": y.detach()}
            self._grad(record, "dy", y)
            self.l2norm.append(record)
            return y
        return l2norm

    def _delta(self, original):
        def delta(q, k, v, g=None, beta=None, **kwargs):
            out = original(q, k, v, g=g, beta=beta, **kwargs)
            record = {"path": self.where, "q": q.detach(), "k": k.detach(), "v": v.detach(),
                      "g": g.detach(), "beta": beta.detach(), "o": out[0].detach(),
                      "plain": (kwargs.get("initial_state") is None
                                and not kwargs.get("output_final_state")
                                and kwargs.get("use_qk_l2norm_in_kernel") is True)}
            for name, tensor in (("dq", q), ("dk", k), ("dv", v), ("dg", g), ("dbeta", beta),
                                 ("do", out[0])):
                self._grad(record, name, tensor)
            self.delta.append(record)
            return out
        return delta

    def _rope(self, original):
        def rope(q, k, cos, sin, *args, **kwargs):
            qe, ke = original(q, k, cos, sin, *args, **kwargs)
            record = {"path": self.where, "q": q.detach(), "k": k.detach(), "cos": cos.detach(),
                      "sin": sin.detach(), "q_embed": qe.detach(), "k_embed": ke.detach()}
            self._grad(record, "dq_embed", qe)
            self._grad(record, "dk_embed", ke)
            self.rope.append(record)
            return qe, ke
        return rope

    def _sdpa(self, original):
        def sdpa(query, key, value, *args, **kwargs):
            out = original(query, key, value, *args, **kwargs)
            record = {"path": self.where, "q": query.detach(), "k": key.detach(),
                      "v": value.detach(), "o": out.detach(),
                      "plain": (kwargs.get("attn_mask", args[0] if args else None) is None
                                and kwargs.get("is_causal") is True
                                and not kwargs.get("dropout_p"))}
            for name, tensor in (("dq", query), ("dk", key), ("dv", value), ("do", out)):
                self._grad(record, name, tensor)
            self.sdpa.append(record)
            return out
        return sdpa

    def _ce(self, original):
        def ce(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
            loss = original(source, target, num_items_in_batch, ignore_index, **kwargs)
            record = {"source": source.detach(), "target": target.detach(),
                      "num_items_in_batch": num_items_in_batch, "ignore_index": ignore_index,
                      "loss": loss.detach()}
            self._grad(record, "dloss", loss)
            self.ce.append(record)
            return loss
        return ce


# ── calls -> payloads in each task's ABI ─────────────────────────────────────


def invocations(capture: LiveCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 boundary of one step, in its case's task ABI."""
    modules = dict(model.named_modules())
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    linear = {r["path"]: r for r in capture.linear}
    norms = {r["path"]: r for r in capture.norm}
    gated = {r["path"]: r for r in capture.gated_norm}
    gdn = {r["path"]: r for r in capture.gdn}
    attn = {r["path"]: r for r in capture.attn}
    delta = {r["path"]: r for r in capture.delta}
    conv = {r["path"]: r for r in capture.conv}
    rope = {r["path"]: r for r in capture.rope}
    sdpa = {r["path"]: r for r in capture.sdpa}
    if not all(r["plain"] for r in capture.delta + capture.sdpa):
        raise ValueError("a delta-rule or SDPA call is outside the declared case")

    def emit(*parts):
        return invocation(cases, *parts)

    def call(path, role, kind):
        return {"path": path, "layer": layer_index(path), "role": role, "kind": kind}

    def gate_half(attn_path, tensor):
        """The attention gate: the second half of each head of q_proj's output."""
        module = modules[attn_path]
        b, t, _ = tensor.shape
        heads, d = module.config.num_attention_heads, module.head_dim
        return tensor.view(b, t, heads, 2 * d)[..., d:].reshape(b, t, heads * d)

    # Level 1 ---------------------------------------------------------------

    for r in capture.linear:
        x, y, k, n = r["x"], r["y"], r["x"].shape[-1], r["y"].shape[-1]
        notes = {"tied_weight": "model.embed_tokens.weight"} if r["path"] == "lm_head" and tied else {}
        yield emit("linear_no_bias", 1,
                   call(r["path"], r["path"].rsplit(".", 1)[-1], "module:nn.Linear"),
                   {"x": x.reshape(-1, k), "weight": modules[r["path"]].weight},
                   {"dy": r["dy"].reshape(-1, n)}, {"y": y.reshape(-1, n)}, notes)

    for r in capture.norm:
        h = r["x"].shape[-1]
        role = ("first_input_norm" if r["path"] == "model.layers.0.input_layernorm"
                else "final_norm" if r["path"] == "model.norm" else r["path"].rsplit(".", 1)[-1])
        yield emit("gemma_rmsnorm", 1, call(r["path"], role, "module:Qwen3_5RMSNorm"),
                   {"x": r["x"].reshape(-1, h), "weight": modules[r["path"]].weight, "eps": r["eps"]},
                   {"dy": r["dy"].reshape(-1, h)}, {"y": r["y"].reshape(-1, h)})

    for r in capture.gated_norm:
        yield emit("gated_rmsnorm", 1, call(r["path"], "norm", "module:Qwen3_5RMSNormGated"),
                   {"x": r["x"], "z": r["z"], "weight": modules[r["path"]].weight, "eps": r["eps"]},
                   {"dy": r["dy"]}, {"y": r["y"]})

    for path, r in conv.items():
        if not r["bias_is_none"] or r["activation"] != "silu":
            raise ValueError(f"{path}: the conv call is outside the declared case")
        yield emit("causal_conv1d", 1, call(f"{path}#causal_conv1d", "causal_conv1d_fn",
                                            "function:causal_conv1d_fn"),
                   {"x": r["x"], "weight": r["weight"], "silu": 1}, {"dy": r["dy"]}, {"y": r["y"]})

    seen: dict[str, int] = {}
    for r in capture.l2norm:
        which = "qk"[seen.get(r["path"], 0)]
        seen[r["path"]] = seen.get(r["path"], 0) + 1
        d = r["x"].shape[-1]
        yield emit("l2norm", 1, call(f"{r['path']}#l2norm_{which}", f"l2norm_{which}",
                                     "function:l2norm"),
                   {"x": r["x"].reshape(-1, d), "eps": r["eps"]},
                   {"dy": r["dy"].reshape(-1, d)}, {"y": r["y"].reshape(-1, d)})

    for path, r in delta.items():
        heads = r["g"].shape[-1]
        module = modules[path]
        yield emit("gdn_gating", 1, call(f"{path}#gdn_gating", "gdn_gating", "inline:g,beta"),
                   {"a": linear[f"{path}.in_proj_a"]["y"].reshape(-1, heads),
                    "b": linear[f"{path}.in_proj_b"]["y"].reshape(-1, heads),
                    "A_log": module.A_log, "dt_bias": module.dt_bias},
                   {"dg": r["dg"].reshape(-1, heads), "dbeta": r["dbeta"].reshape(-1, heads)},
                   {"g": r["g"].reshape(-1, heads), "beta": r["beta"].reshape(-1, heads)},
                   {"g_beta": "the delta rule's g and beta (their only consumer)"})

    for path, r in rope.items():
        cos, sin = shared_row(r["cos"]), shared_row(r["sin"])
        for name in ("q", "k"):
            yield emit("partial_rope", 1, call(f"{path}#rope_{name}", f"rope_{name}",
                                               "function:apply_rotary_pos_emb"),
                       {"x": r[name], "cos": cos, "sin": sin}, {"dy": r[f"d{name}_embed"]},
                       {"y": r[f"{name}_embed"]})

    for path, r in sdpa.items():
        yield emit("causal_gqa_attention", 1, call(f"{path}#sdpa", "sdpa",
                                                   "function:F.scaled_dot_product_attention"),
                   {"q": r["q"], "k": r["k"], "v": r["v"]}, {"do": r["do"]}, {"o": r["o"]})

    for r in capture.mlp:
        gate_r, up_r, down_r = (linear[f"{r['path']}.{n}"] for n in ("gate_proj", "up_proj", "down_proj"))
        cols = gate_r["y"].shape[-1]
        yield emit("swiglu", 1, call(f"{r['path']}#act_mul", "act_mul", "inline:act_fn(gate)*up"),
                   {"a": gate_r["y"].reshape(-1, cols), "b": up_r["y"].reshape(-1, cols)},
                   {"dc": down_r["dx"].reshape(-1, cols)}, {"c": down_r["x"].reshape(-1, cols)})

    for r in capture.ce:
        if r["num_items_in_batch"] is not None or r["ignore_index"] != -100:
            raise ValueError("the CE call is outside the declared case")
        yield emit("cross_entropy", 1, call("loss_function#fixed_cross_entropy", "loss",
                                            "function:fixed_cross_entropy"),
                   {"logits": r["source"], "target": r["target"]}, {"dloss": r["dloss"]},
                   {"loss": r["loss"]}, {"target": "labels as ForCausalLMLoss passes them"})

    # Level 2 ---------------------------------------------------------------

    for path, r in delta.items():
        module, mixer, norm = modules[path], gdn[path], gated[f"{path}.norm"]
        b, t = mixer["x"].shape[:2]
        z = norm["z"].reshape(b, t, module.num_v_heads, module.head_v_dim)
        dz = norm["dz"].reshape(b, t, module.num_v_heads, module.head_v_dim)
        yield emit("qwen3_5_gdn_in_proj", 2, call(f"{path}#gdn_in_proj", "gdn_in_proj",
                                                  "boundary:GatedDeltaNet[:delta rule]"),
                   {"x": mixer["x"], "w_qkv": module.in_proj_qkv.weight, "w_z": module.in_proj_z.weight,
                    "w_b": module.in_proj_b.weight, "w_a": module.in_proj_a.weight,
                    "conv_w": conv[path]["weight"], "A_log": module.A_log, "dt_bias": module.dt_bias},
                   {"dq": r["dq"], "dk": r["dk"], "dv": r["dv"], "dz": dz, "dbeta": r["dbeta"],
                    "dg": r["dg"]},
                   {"q": r["q"], "k": r["k"], "v": r["v"], "z": z, "beta": r["beta"], "g": r["g"]})
        yield emit("qwen3_5_gated_delta_rule", 2, call(f"{path}#gated_delta_rule", "gated_delta_rule",
                                                       "function:torch_chunk_gated_delta_rule"),
                   {"q": r["q"], "k": r["k"], "v": r["v"], "g": r["g"], "beta": r["beta"]},
                   {"do": r["do"]}, {"o": r["o"]})
        yield emit("qwen3_5_gdn_norm_out", 2, call(f"{path}#gdn_norm_out", "gdn_norm_out",
                                                   "boundary:gated norm + out_proj"),
                   {"o": r["o"], "z": z, "w_norm": module.norm.weight, "w_out": module.out_proj.weight,
                    "eps": norm["eps"]},
                   {"dy": mixer["dy"]}, {"y": mixer["y"]})

    for path, mixer in attn.items():
        module, s, ro = modules[path], sdpa[path], rope[path]
        q_proj = linear[f"{path}.q_proj"]
        gate, dgate = gate_half(path, q_proj["y"]), gate_half(path, q_proj["dy"])
        yield emit("qwen3_5_qkv_norm_partial_rope", 2,
                   call(f"{path}#qkv_norm_partial_rope", "qkv_norm_partial_rope",
                        "boundary:Qwen3_5Attention[:rope]"),
                   {"x": mixer["x"], "q_weight": module.q_proj.weight, "k_weight": module.k_proj.weight,
                    "v_weight": module.v_proj.weight, "q_norm_weight": module.q_norm.weight,
                    "k_norm_weight": module.k_norm.weight, "cos": shared_row(mixer["cos"]),
                    "sin": shared_row(mixer["sin"]), "eps": module.q_norm.eps},
                   {"dq": ro["dq_embed"], "dk": ro["dk_embed"], "dv": s["dv"], "dgate": dgate},
                   {"q": ro["q_embed"], "k": ro["k_embed"], "v": s["v"], "gate": gate},
                   {"gate": "gate half of q_proj's output and cotangent (data movement only)"})
        yield emit("qwen3_5_gated_attention", 2, call(f"{path}#gated_attention", "gated_attention",
                                                      "boundary:sdpa * sigmoid(gate) + o_proj"),
                   {"q": s["q"], "k": s["k"], "v": s["v"], "gate": gate,
                    "o_weight": module.o_proj.weight},
                   {"dy": mixer["dy"]}, {"y": mixer["y"]})

    for layer in capture.layers:
        prefix = layer["path"]
        mixer = gdn.get(f"{prefix}.linear_attn") or attn[f"{prefix}.self_attn"]
        post = norms[f"{prefix}.post_attention_layernorm"]
        yield emit("qwen3_5_residual_rmsnorm", 2, call(f"{prefix}#residual_post_attention",
                                                       "residual_post_attention",
                                                       "boundary:residual add + Qwen3_5RMSNorm"),
                   {"branch": mixer["y"], "residual": layer["x"],
                    "weight": modules[post["path"]].weight, "eps": post["eps"]},
                   {"dnormed": post["dy"], "dsummed": post["dx"]},
                   {"normed": post["y"], "summed": post["x"]},
                   {"dsummed": "total gradient of the residual sum (post norm and next residual add)"})

    for r in capture.mlp:
        module = modules[r["path"]]
        yield emit("qwen3_5_swiglu_mlp", 2, call(f"{r['path']}#swiglu_mlp", "swiglu_mlp",
                                                 "boundary:Qwen3_5MLP"),
                   {"x": r["x"], "gate_weight": module.gate_proj.weight, "up_weight": module.up_proj.weight,
                    "down_weight": module.down_proj.weight},
                   {"dout": r["dy"]}, {"out": r["y"]})
