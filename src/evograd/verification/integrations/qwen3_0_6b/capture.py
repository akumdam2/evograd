"""Real training payloads for Qwen3-0.6B: capture and replay.

One native training step of the pinned model (pretrained checkpoint, a real
WikiText-2 TRAIN batch, B=2 x T=2048, BF16, SDPA, train mode, the HF causal-LM
loss, deterministic settings) runs per data order with recorders attached.
Every Level-1 call and Level-2 boundary of the step becomes one payload of the
benchmark case it belongs to: the call's arguments in the task's ABI, its
scalar arguments and its output cotangents.

The cases are the model's observed benchmark cases (suite
``qwen3_0_6b_observed``). A call belongs to the case of its task whose declared
shapes and dtypes its arguments bind; calls per step come from the harvest
snapshot. Nothing here edits a task or the snapshot.

What is recorded
================

* Arguments are ``detach()``-ed views of what the model passed (exact storage,
  strides and offset). 2-D task ABIs receive ``reshape(-1, k)`` of them.
* Output cotangents come from tensor hooks on the call's outputs. A hook on a
  non-leaf output fires once with the total gradient of that output, which is
  the call's output cotangent. Gradients are cloned in the hook (strides kept).
* Input gradients recorded by hooks are model-level totals, so they are not
  stored; the gates recompute each payload's local VJP from the reference.
* Residual fusion sites (``fused_add_rms_norm``): 28 within-layer, 27
  cross-layer and the final norm's sum, 56 in all. The final sum has no
  consumer; its cotangent is supplied as zeros and recorded as such.
* CE: ``fixed_cross_entropy``'s own ``(source, target)``, labels already
  shifted by ``ForCausalLMLoss``. ``lm_head.weight`` is the tied embedding.
* Recorders change no arithmetic. Each data order runs the same initial model
  uninstrumented and instrumented on the same batch; loss, logits and every
  parameter gradient must agree bit for bit.

The corpus samples the pinned initial state on these batches, not later
optimizer states.

    python -m evograd.verification.integrations.qwen3_0_6b capture --out DIR
    python -m evograd.verification.integrations.qwen3_0_6b fidelity --corpus DIR/manifest.json --out FILE

``fidelity`` checks every payload's layout and that the model reference
(:mod:`.reference`) reproduces the recorded live output bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

#: The same pins as the real-text Tier-3 workload.
PRETRAINED = "pretrained:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
REAL_TEXT = "wikitext-2-raw:b08601e04326c79dfdd32d625aee71d232d685c3"
WORKLOAD_CONFIG = {
    "model": "qwen3_0_6b", "device": "cuda", "dtype": "bfloat16", "batch_size": 2,
    "seq_len": 2048, "seed": 0, "data_seed": 0, "attn_implementation": "sdpa",
    "weights": PRETRAINED, "data": REAL_TEXT,
}
MODEL = "qwen3_0_6b"
SUITE = "qwen3_0_6b_observed"
ORDERS = (101, 202, 303, 404)
CALIBRATION_ORDERS = (101, 202)
ACCEPTANCE_ORDERS = (303, 404)
DETERMINISM = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "use_deterministic_algorithms": True,
               "cudnn.deterministic": True, "cudnn.benchmark": False,
               "cuda.matmul.allow_tf32": False, "float32_matmul_precision": "highest",
               "torch_num_threads": 8}
LOSS_DEFINITION = ("HF Qwen3ForCausalLM(input_ids, labels=labels).loss: ForCausalLMLoss -> "
                   "fixed_cross_entropy over float32 logits, labels shifted once with -100 padding, "
                   "mean over valid tokens; backward of that scalar")


def configure() -> dict[str, Any]:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISM["CUBLAS_WORKSPACE_CONFIG"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before the process starts")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(DETERMINISM["torch_num_threads"])
    return determinism_record()


def determinism_record() -> dict[str, Any]:
    import torch

    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
            "sdp_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdp_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "sdp_math": torch.backends.cuda.math_sdp_enabled(),
            "torch_num_threads": torch.get_num_threads()}


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── cases ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One observed benchmark case and how often a training step calls it."""

    case_id: str
    task: str
    level: int
    workload: Any
    frequency: int


def observed_cases() -> tuple[Case, ...]:
    """Every observed L1 and L2 case of the model, calls per step from the snapshot."""
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown import load_snapshot

    snapshot = load_snapshot(MODEL)
    cases = []
    for task, entry in snapshot["level1"].items():
        workloads = get_task(task).benchmark_suites[SUITE]
        configs = entry["configurations"]
        if len(workloads) != len(configs) or any(dict(w.dims) != c["dims"]
                                                 for w, c in zip(workloads, configs)):
            raise ValueError(f"{task}: suite {SUITE} does not follow the snapshot")
        cases += [Case(f"{task}/{w.provenance.component}", task, 1, w, int(c["frequency"]))
                  for w, c in zip(workloads, configs)]
    for task, entry in snapshot["tasks"].items():
        (workload,) = get_task(task).benchmark_suites[SUITE]
        frequency = (entry.get("fusion_sites") or {}).get("total", entry["frequency"])
        cases.append(Case(f"{task}/{workload.provenance.component}", task, 2, workload,
                          int(frequency)))
    return tuple(cases)


def case_for(cases: Sequence[Case], task: str, args: dict[str, Any]) -> Case:
    """The one case of ``task`` whose declared shapes and dtypes ``args`` bind."""
    import torch

    from evograd.benchmark import get_task
    from evograd.opdecl.activity import bind_shape
    from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

    op = get_task(task)
    shaped = [a for a in op.args if getattr(a, "shape", None) not in (None, "unspecified")
              and torch.is_tensor(args.get(a.name))]

    def binds(case: Case) -> bool:
        for arg in shaped:
            value = args[arg.name]
            try:
                shape = tuple(bind_shape(arg.shape, case.workload.dims))
            except (KeyError, ValueError):
                return False
            if shape != tuple(value.shape) or \
                    value.dtype != resolve_dtype(arg_dtype_name(arg, case.workload)):
                return False
        return True

    hits = [c for c in cases if c.task == task and binds(c)]
    if len(hits) != 1:
        shapes = {a.name: tuple(args[a.name].shape) for a in shaped}
        raise KeyError(f"{task}: {len(hits)} observed cases match {shapes}")
    return hits[0]


# ── workload, data, model ────────────────────────────────────────────────────


class TrainOnly:
    """Refuses every dataset split but ``train`` while installed (records requests)."""

    def __init__(self):
        self.requested: list[str] = []
        self._original = None

    def __enter__(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata

        original = textdata.dataset_file

        def dataset_file(split, *args, **kwargs):
            self.requested.append(str(split))
            if split != "train":
                raise RuntimeError(f"real-payload capture reads the train split only, not {split!r}")
            return original(split, *args, **kwargs)

        self._original, textdata.dataset_file = original, dataset_file
        return self

    def __exit__(self, *exc):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata

        textdata.dataset_file = self._original


def load_workload():
    from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

    return Qwen3Workload.from_config(dict(WORKLOAD_CONFIG))


def train_batches(workload, orders=ORDERS) -> tuple[dict[int, tuple[Any, Any]], dict[str, Any]]:
    """The first batch of each data order over the packed TRAIN split, with identities."""
    from transformers import AutoTokenizer

    from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import pretrained_snapshot
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


def build_model(workload):
    """A fresh native model at the pinned initial state (no site replaced)."""
    from evograd.evaluation.tier3.patch import KernelSet

    model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
    return model.train()


def parameter_state(model) -> dict[str, Any]:
    from evograd.verification.common.tensors import tensor_digest

    digests = {name: tensor_digest(p) for name, p in model.named_parameters()}
    return {"parameters": len(digests),
            "tied_lm_head": model.lm_head.weight is model.model.embed_tokens.weight,
            "state_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}


def run_step(model, ids, labels):
    output = model(input_ids=ids, labels=labels, use_cache=False)
    output.loss.backward()
    return output


def step_record(model, output) -> dict[str, Any]:
    """Loss, logits and every parameter gradient, by content hash."""
    from evograd.verification.common.tensors import tensor_digest

    grads = {name: (tensor_digest(p.grad) if p.grad is not None else None)
             for name, p in model.named_parameters()}
    return {"loss": float(output.loss.detach()), "loss_digest": tensor_digest(output.loss.detach()),
            "logits_digest": tensor_digest(output.logits.detach()), "grads": grads}


# ── live capture ─────────────────────────────────────────────────────────────


class LiveCapture:
    """Hooks and pass-through recorders on an unmodified ``Qwen3ForCausalLM``.

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
        import transformers.models.qwen3.modeling_qwen3 as modeling
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm)

        for path, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._module(path, module, self.linear)
            elif isinstance(module, Qwen3RMSNorm):
                self._module(path, module, self.rmsnorm, extra={"eps": module.variance_epsilon})
            elif isinstance(module, Qwen3DecoderLayer):
                self._module(path, module, self.layers, grads=False)
            elif isinstance(module, Qwen3Attention):
                self._attention(path, module)
            elif isinstance(module, Qwen3MLP):
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


def _layer(path: str) -> int | None:
    parts = path.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return None


def _shared_row(table):
    """The model's cos/sin are [B, T, D] with identical rows (no padding): its first row."""
    import torch

    first = table[:1]
    if table.shape[0] > 1 and not torch.equal(table, first.expand_as(table)):
        raise ValueError("cos/sin rows differ across the batch; one shared table does not apply")
    return first


def invocations(capture: LiveCapture, model, cases: Sequence[Case]) -> Iterator[dict[str, Any]]:
    """Every recorded L1 call and L2 boundary of one step, in its case's task ABI."""
    import torch

    modules = dict(model.named_modules())
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    linear_by_path = {r["path"]: r for r in capture.linear}

    def emit(task, level, call, args, cotangents, live_outputs, notes=None):
        return {"case_id": case_for(cases, task, args).case_id, "task": task, "level": level,
                "call": call, "args": args, "cotangents": cotangents,
                "live_outputs": live_outputs, "notes": notes or {}}

    for r in capture.linear:
        module = modules[r["path"]]
        x, y, dy = r["x"], r["y"], r["dy"]
        k, n = x.shape[-1], y.shape[-1]
        notes = {"tied_weight": "model.embed_tokens.weight"} if r["path"] == "lm_head" and tied else {}
        yield emit("linear_no_bias", 1,
                   {"path": r["path"], "layer": _layer(r["path"]), "role": r["path"].rsplit(".", 1)[-1],
                    "kind": "module:nn.Linear"},
                   {"x": x.reshape(-1, k), "weight": module.weight},
                   {"dy": dy.reshape(-1, n)}, {"y": y.reshape(-1, n)}, notes)

    for r in capture.rmsnorm:
        module = modules[r["path"]]
        x, y, dy, h = r["x"], r["y"], r["dy"], r["x"].shape[-1]
        role = ("first_input_norm" if r["path"] == "model.layers.0.input_layernorm"
                else "final_norm" if r["path"] == "model.norm" else r["path"].rsplit(".", 1)[-1])
        notes = {"layout": "per-head norm flattened to [rows, head_dim]"} if x.dim() == 4 else {}
        yield emit("rmsnorm", 1,
                   {"path": r["path"], "layer": _layer(r["path"]), "role": role,
                    "kind": "module:Qwen3RMSNorm"},
                   {"x": x.reshape(-1, h), "weight": module.weight, "eps": r["eps"]},
                   {"dy": dy.reshape(-1, h)}, {"y": y.reshape(-1, h)}, notes)

    for layer, r in enumerate(capture.rope):
        cos, sin = _shared_row(r["cos"])[0], _shared_row(r["sin"])[0]
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
        yield emit("qwen3_qkv_norm_rope", 2,
                   {"path": f"model.layers.{layer}.self_attn#qkv_norm_rope", "layer": layer,
                    "role": "qkv_norm_rope", "kind": "boundary:Qwen3Attention[:5]"},
                   {"x": r["x"], "q_weight": attn.q_proj.weight, "k_weight": attn.k_proj.weight,
                    "v_weight": attn.v_proj.weight, "q_norm_weight": attn.q_norm.weight,
                    "k_norm_weight": attn.k_norm.weight, "cos": _shared_row(r["cos"]),
                    "sin": _shared_row(r["sin"]), "eps": attn.q_norm.variance_epsilon},
                   {"dq": sdpa["dq"], "dk": sdpa["dk"], "dv": sdpa["dv"]},
                   {"q": sdpa["q"], "k": sdpa["k"], "v": sdpa["v"]},
                   {"cotangents": "SDPA input gradients (head-major views, as the model delivers them)"})
        yield emit("qwen3_attention", 2,
                   {"path": f"model.layers.{layer}.self_attn#attention", "layer": layer,
                    "role": "attention", "kind": "boundary:sdpa+o_proj"},
                   {"q": sdpa["q"], "k": sdpa["k"], "v": sdpa["v"], "o_weight": attn.o_proj.weight},
                   {"dout": r["dy"]}, {"out": r["y"]})

    for r in capture.mlp:
        mlp, layer = modules[r["path"]], _layer(r["path"])
        yield emit("qwen3_swiglu_mlp", 2,
                   {"path": f"model.layers.{layer}.mlp#swiglu_mlp", "layer": layer,
                    "role": "swiglu_mlp", "kind": "boundary:Qwen3MLP"},
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
            yield emit("fused_add_rms_norm", 2,
                       {"path": call, "layer": layer, "role": role, "norm": norm["path"],
                        "kind": "boundary:residual add + Qwen3RMSNorm"},
                       {"x": x.reshape(-1, cols), "r": r_in.reshape(-1, cols), "weight": module.weight,
                        "eps": norm["eps"]},
                       {"dout": norm["dy"].reshape(-1, cols), "dsummed": dsummed},
                       {"out": norm["y"].reshape(-1, cols), "summed": norm["x"].reshape(-1, cols)},
                       {"dsummed_origin": "total gradient of the " + origin if origin else
                        "absent in the model (the final sum has no consumer); zeros supplied "
                        "for the declared two-output ABI"})


def record_step(writer, recorder: LiveCapture, model, cases: Sequence[Case],
                batch: dict[str, Any]) -> int:
    """Write every invocation of one recorded step as payloads; returns how many."""
    import torch

    from evograd.verification.common.tensors import tensor_digest
    from evograd.verification.payloads.corpus import cotangent_stats

    writer.new_step()
    count = 0
    for inv in invocations(recorder, model, cases):
        pid = f"o{batch['order']}/{inv['case_id']}/{inv['call']['path']}"
        args = {name: writer.put(value) if torch.is_tensor(value) else {"scalar": value}
                for name, value in inv["args"].items()}
        cots = {name: writer.put(value) for name, value in inv["cotangents"].items()}
        stats = {name: cotangent_stats(value) for name, value in inv["cotangents"].items()}
        if inv["call"]["role"] == "final_model_norm":
            stats["dsummed"]["origin"] = "absent_in_model_zero_supplied"
        storages: dict[str, list[str]] = {}
        for name, rec in {**args, **cots}.items():
            if "storage" in rec:
                storages.setdefault(rec["storage"], []).append(name)
        writer.add_payload({
            "payload_id": pid, "case_id": inv["case_id"], "task": inv["task"], "level": inv["level"],
            "batch": dict(batch), "call": {**inv["call"], "index_in_step": count},
            "args": args, "cotangents": cots, "cotangent_stats": stats,
            "aliases": [sorted(names) for names in storages.values() if len(names) > 1],
            "live_outputs": {name: {"digest": tensor_digest(v), "shape": list(v.shape),
                                    "stride": list(v.stride()), "dtype": str(v.dtype)}
                             for name, v in inv["live_outputs"].items()},
            "notes": inv["notes"]})
        count += 1
    return count


def splits(payloads: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Calibration (orders 101/202), acceptance (303/404) and a development subset.

    Development subset: per case, role and calibration order, the first,
    middle and last recorded layer; frozen with the corpus.
    """
    calibration = [p["payload_id"] for p in payloads if p["batch"]["order"] in CALIBRATION_ORDERS]
    acceptance = [p["payload_id"] for p in payloads if p["batch"]["order"] in ACCEPTANCE_ORDERS]
    groups: dict[tuple, list[dict]] = {}
    for p in payloads:
        if p["batch"]["order"] in CALIBRATION_ORDERS:
            groups.setdefault((p["case_id"], p["call"]["role"], p["batch"]["order"]), []).append(p)
    development = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda p: (p["call"]["layer"] if p["call"]["layer"] is not None
                                                     else -1, p["call"]["path"]))
        picks = {members[0]["payload_id"], members[len(members) // 2]["payload_id"],
                 members[-1]["payload_id"]}
        development += [m["payload_id"] for m in members if m["payload_id"] in picks]
    return {"calibration": calibration, "acceptance": acceptance, "development": development}


def coverage(cases: Sequence[Case], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Every case received exactly its calls per step, in every recorded order."""
    expected = {c.case_id: c.frequency for c in cases}
    seen: dict[str, dict[str, int]] = {}
    for p in payloads:
        per_case = seen.setdefault(p["case_id"], {})
        order = str(p["batch"]["order"])
        per_case[order] = per_case.get(order, 0) + 1
    orders = sorted({str(p["batch"]["order"]) for p in payloads})
    problems = [f"{case}: order {o}: {seen.get(case, {}).get(o, 0)} of {n} invocations"
                for case, n in expected.items() for o in orders if seen.get(case, {}).get(o, 0) != n]
    unmapped = sorted(set(seen) - set(expected))
    return {"expected_per_order": expected, "observed": seen, "orders": orders,
            "problems": problems, "unmapped_cases": unmapped,
            "complete": not problems and not unmapped}


# ── capture driver ───────────────────────────────────────────────────────────


def _capture_identity(cases, workload, batch_identity, state, settings) -> dict[str, Any]:
    import torch
    import transformers
    from transformers.models.qwen3 import modeling_qwen3

    from evograd.benchmark.topdown import load_snapshot
    from evograd.opdecl.oracle import reference_identity

    from .reference import reference_op

    return {"name": f"{MODEL}-real-training-payloads", "suite": SUITE,
            "snapshot_hash": load_snapshot(MODEL).get("snapshot_hash"),
            "cases": {c.case_id: {"task": c.task, "level": c.level, "dims": dict(c.workload.dims),
                                  "dtype": c.workload.dtype, "frequency": c.frequency} for c in cases},
            "references": {t: reference_identity(reference_op(t)) for t in sorted({c.task for c in cases})},
            "modeling_qwen3_sha256": _sha_file(modeling_qwen3.__file__),
            "transformers": transformers.__version__, "torch": str(torch.__version__),
            "workload": dict(WORKLOAD_CONFIG), "checkpoint": PRETRAINED, "parameter_state": state,
            "parameter_state_note": "every capture rebuilds the model at this initial pretrained state; "
                                    "the corpus samples that state, not later optimizer states",
            "data": batch_identity, "loss": LOSS_DEFINITION, "execution": settings,
            "capture_implementation": {"module": __name__, "sha256": _sha_file(__file__)}}


def capture(out: str | Path, *, orders=ORDERS, log=print) -> Path:
    """Capture the corpus: one fresh initial model per data order, fidelity per order."""
    import gc

    import torch

    from evograd.verification.payloads.corpus import CorpusWriter, sha256_text

    settings = configure()
    cases = observed_cases()
    workload = load_workload()
    batches, batch_identity = train_batches(workload, orders)
    writer = CorpusWriter(out)
    fidelity: dict[str, Any] = {}
    state = None
    for order in orders:
        t0 = time.time()
        ids, labels = (t.to("cuda") for t in batches[order])
        model = build_model(workload)
        state_now = parameter_state(model)
        state = state or state_now
        if state_now != state:
            raise RuntimeError("the initial parameter state differs between captures")
        plain = step_record(model, run_step(model, ids, labels))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = build_model(workload)
        recorder = LiveCapture(model).install()
        try:
            output = run_step(model, ids, labels)
        finally:
            recorder.remove()
        instrumented = step_record(model, output)
        differing = sorted(n for n in plain["grads"] if plain["grads"][n] != instrumented["grads"][n])
        fidelity[str(order)] = {
            "loss_uninstrumented": plain["loss"], "loss_instrumented": instrumented["loss"],
            "loss_bitwise": plain["loss_digest"] == instrumented["loss_digest"],
            "logits_bitwise": plain["logits_digest"] == instrumented["logits_digest"],
            "grads_compared": len(plain["grads"]), "grads_differing": differing,
            "identical": (plain["loss_digest"] == instrumented["loss_digest"]
                          and plain["logits_digest"] == instrumented["logits_digest"] and not differing)}
        if not fidelity[str(order)]["identical"]:
            raise RuntimeError(f"order {order}: instrumentation changed the step: {fidelity[str(order)]}")
        del output
        batch = {"order": order, "sha256": batch_identity["batches"][str(order)]["input_ids_sha256"]}
        count = record_step(writer, recorder, model, cases, batch)
        log(f"order {order}: {count} payloads, {time.time() - t0:.1f}s, "
            f"{writer.bytes_written / 2**30:.1f} GiB written")
        del recorder, model
        gc.collect()
        torch.cuda.empty_cache()
    payloads = writer.payloads
    covered = coverage(cases, payloads)
    if not covered["complete"]:
        raise RuntimeError(f"incomplete capture: {covered['problems'][:5]} {covered['unmapped_cases']}")
    frozen = splits(payloads)
    identity = _capture_identity(cases, workload, batch_identity, state, settings)
    return writer.finish(identity=identity, splits=frozen, coverage=covered,
                         extra={"capture_fidelity": fidelity,
                                "split_rules": {
                                    "calibration": f"every payload of orders {list(CALIBRATION_ORDERS)}",
                                    "acceptance": f"every payload of orders {list(ACCEPTANCE_ORDERS)}",
                                    "development": "per case, role and calibration order: first, middle "
                                                   "and last recorded layer (frozen with the corpus)"},
                                "splits_sha256": {k: sha256_text(json.dumps(sorted(v)))
                                                  for k, v in frozen.items()}})


# ── replay fidelity ──────────────────────────────────────────────────────────


def replay_fidelity(corpus_path: str | Path, *, device: str = "cuda", log=print) -> dict[str, Any]:
    """Every payload loaded back: layout as recorded, and whether the model
    reference reproduces the live output bit for bit (counted per task)."""
    import torch

    from evograd.opdecl.inputs import as_output_tuple
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.verification.common.tensors import tensor_digest
    from evograd.verification.payloads.corpus import Corpus

    from .reference import reference_op

    configure()
    corpus = Corpus.open(corpus_path)
    rows, by_task = [], {}
    started = time.time()
    for n, record in enumerate(corpus.payloads()):
        op = reference_op(record["task"])
        values = corpus.load(record, device=device)
        layout_ok = all(list(values[name].stride()) == entry["stride"] and
                        list(values[name].shape) == entry["shape"]
                        for name, entry in {**record["args"], **record["cotangents"]}.items()
                        if "storage" in entry)
        with torch.no_grad():
            out = resolve_runtime_forward(op)(*[values.get(a.name, getattr(a, "default", None))
                                                for a in op.args])
        outs = dict(zip(op.output_names, as_output_tuple(op, out)))
        equal = {name: tensor_digest(outs[name]) == live["digest"]
                 for name, live in record["live_outputs"].items()}
        rows.append({"payload_id": record["payload_id"], "layout_ok": layout_ok, "outputs_bitwise": equal})
        task = by_task.setdefault(record["task"], {"payloads": 0, "layout_failures": 0,
                                                   "output_mismatches": 0})
        task["payloads"] += 1
        task["layout_failures"] += not layout_ok
        task["output_mismatches"] += not all(equal.values())
        if n % 200 == 0:
            log(f"replay {n}: {time.time() - started:.0f}s")
        del values, out, outs
    return {"corpus_id": corpus.corpus_id, "payloads": len(rows), "by_task": by_task,
            "layouts_ok": all(r["layout_ok"] for r in rows),
            "outputs_bitwise": all(all(r["outputs_bitwise"].values()) for r in rows),
            "seconds": time.time() - started, "rows": rows}
