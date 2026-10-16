"""Small CPU declarations, candidates, blocks and models for the verification tests.

Everything here is synthetic and deterministic. Declarations point their
``forward`` at functions in this module, so ``tests`` must be importable
(run from the repository root with ``PYTHONPATH=src:.``).
"""

from __future__ import annotations

import types
from typing import Any, Mapping

import torch
from torch import nn

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.verification.common.execution import Invocation, Sample

HERE = "tests.verification._fixtures"


# ── declarations ─────────────────────────────────────────────────────────────


def scale_shift_ref(x, w, b):
    return x * w + b


def scale_shift_runtime(x, w, b):
    return torch.addcmul(b, x, w)


SCALE_SHIFT = declare_op(
    name="vt_scale_shift", forward=f"{HERE}:scale_shift_ref",
    runtime_forward=f"{HERE}:scale_shift_runtime", dims=("R", "C"),
    args=(Active("x", "[R, C]"), Active("w", "[C]"), Active("b", "[C]")),
    output=Active("y", "[R, C]"), forward_semantics="y = x * w + b",
    backward_semantics="return dx, dw, db",
    correctness=(Workload({"R": 4, "C": 8}, "float32"), Workload({"R": 3, "C": 5}, "float32")),
    tolerances={"float32": (1e-5, 1e-5), "bfloat16": (1e-2, 1e-2)},
    parameter_args=("w", "b"),
)


def split_ref(x, w, alpha=2.0):
    scaled = x * w
    return scaled * alpha, scaled.sum(-1)


def _split_hook(workload, result_name, atol, rtol):
    # Grows atol with the reduction length, for `lo` only: a shape hook.
    if result_name == "lo":
        return atol * workload.dims["C"], rtol
    return atol, rtol


SPLIT = declare_op(
    name="vt_split", forward=f"{HERE}:split_ref", dims=("R", "C"),
    args=(Active("x", "[R, C]"), Active("w", "[C]"), Inactive("alpha", None, default=2.0)),
    output=(Active("hi", "[R, C]"), Active("lo", "[R]")),
    forward_semantics="hi = x*w*alpha; lo = sum(x*w, -1)", backward_semantics="dx, dw",
    correctness=(Workload({"R": 4, "C": 8}, "float32"),),
    tolerances={"float32": (1e-5, 1e-5)},
    tolerance_multipliers={"dw": (3.0, 1.0)}, tolerance_hook=_split_hook,
    parameter_args=("w",),
)


def sumsq_ref(x):
    return (x * x).sum(-1)


def _sumsq(reference_dtype):
    return declare_op(
        name="vt_sumsq", forward=f"{HERE}:sumsq_ref", dims=("R", "C"),
        args=(Active("x", "[R, C]"),), output=Active("y", "[R]"),
        forward_semantics="y = sum(x^2, -1)", backward_semantics="dx = 2 x dy",
        correctness=(Workload({"R": 4, "C": 2048}, "bfloat16"),),
        tolerances={"bfloat16": (1e-6, 1e-6)}, reference_dtype=reference_dtype,
        parameter_args=(),
    )


SUMSQ_F32_REF = _sumsq("float32")
SUMSQ_SAME_DTYPE = _sumsq(None)


def cancel_ref(x):
    return (x + 1000.0) - 1000.0


def _cancel(reference_dtype):
    return declare_op(
        name="vt_cancel", forward=f"{HERE}:cancel_ref", dims=("R", "C"),
        args=(Active("x", "[R, C]"),), output=Active("y", "[R, C]"),
        forward_semantics="y = (x + 1000) - 1000", backward_semantics="dx = dy",
        correctness=(Workload({"R": 4, "C": 16}, "bfloat16"),),
        tolerances={"bfloat16": (1e-6, 1e-6)}, reference_dtype=reference_dtype,
        parameter_args=(),
    )


#: The same mathematics; one oracle runs at float32 and casts back, the other
#: rounds every intermediate to bfloat16 (where x + 1000 swallows x).
CANCEL_F32_REF = _cancel("float32")
CANCEL_SAME_DTYPE = _cancel(None)


def cancel_f32_pair():
    def forward_with_saved(x):
        return ((x.float() + 1000.0) - 1000.0).to(x.dtype), ()

    def backward_from_saved(dy, saved):
        return dy.clone()

    return types.SimpleNamespace(forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved, __name__="vt_cancel_pair")


def mul_ref(a, b):
    return a * b


MUL = declare_op(
    name="vt_mul", forward=f"{HERE}:mul_ref", dims=("R", "C"),
    args=(Active("a", "[R, C]"), Active("b", "[R, C]")), output=Active("y", "[R, C]"),
    forward_semantics="y = a * b", backward_semantics="da = dy*b, db = dy*a",
    correctness=(Workload({"R": 4, "C": 6}, "float32"),),
    tolerances={"float32": (1e-5, 1e-5)}, parameter_args=("b",),
)


# ── candidate pairs ──────────────────────────────────────────────────────────


SCALE_SHIFT_OVERWRITE = declare_op(
    name="vt_scale_shift_ow", forward=f"{HERE}:scale_shift_ref", dims=("R", "C"),
    args=(Active("x", "[R, C]"), Active("w", "[C]"), Active("b", "[C]")),
    output=Active("y", "[R, C]"), forward_semantics="y = x * w + b",
    backward_semantics="return dx, dw, db; may overwrite x with dx",
    correctness=(Workload({"R": 4, "C": 8}, "float32"),),
    tolerances={"float32": (1e-5, 1e-5)}, parameter_args=("w", "b"),
    backward_may_overwrite=("x",),
)


def scale_shift_pair(*, fwd_bias=0.0, dw_scale=1.0, drop_grad=False, extra_grad=False,
                     none_grad=False, nan_output=False, wrong_dtype=False, wrong_shape=False,
                     mutate_upstream=False, mutate_input=False, inf_output=False,
                     overwrite_x=False, clobber_output=False):
    def forward_with_saved(x, w, b):
        y = x * w + b + fwd_bias
        if mutate_input:
            x.mul_(1.0001)
        if nan_output or inf_output:
            y = y.clone()
            y.view(-1)[0] = float("nan") if nan_output else float("inf")
        return y, ((x, w, y) if clobber_output else (x, w))

    def backward_from_saved(dy, saved):
        x, w = saved[:2]
        grads = [dy * w, (dy * x).sum(0) * dw_scale, dy.sum(0)]
        if overwrite_x:                 # Liger-style: the activation is dead now
            x.copy_(grads[0])
            grads[0] = x
        if clobber_output:              # reuse the dead output buffer as workspace
            saved[2].zero_()
        if mutate_upstream:
            dy.mul_(2.0)
        if wrong_dtype:
            grads[0] = grads[0].double()
        if wrong_shape:
            grads[1] = grads[1][:-1]
        if none_grad:
            grads[2] = None
        if drop_grad:
            grads = grads[:2]
        if extra_grad:
            grads.append(grads[2])
        return tuple(grads)

    return types.SimpleNamespace(forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved,
                                 __name__="vt_scale_shift_pair")


def split_pair(*, lo_bias=0.0):
    def forward_with_saved(x, w, alpha=2.0):
        scaled = x * w
        return (scaled * alpha, scaled.sum(-1) + lo_bias), (x, w)

    def backward_from_saved(output_grads, saved, alpha=2.0):
        dhi, dlo = output_grads
        x, w = saved
        dx = dhi * w * alpha + dlo.unsqueeze(-1) * w
        dw = (dhi * x * alpha).sum(0) + (dlo.unsqueeze(-1) * x).sum(0)
        return dx, dw

    return types.SimpleNamespace(forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved, __name__="vt_split_pair")


def sumsq_f32_pair():
    def forward_with_saved(x):
        return (x.float() * x.float()).sum(-1).to(x.dtype), (x,)

    def backward_from_saved(dy, saved):
        (x,) = saved
        return (2.0 * x.float() * dy.float().unsqueeze(-1)).to(x.dtype)

    return types.SimpleNamespace(forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved, __name__="vt_sumsq_pair")


class CountingProvider:
    """A PairProvider look-alike whose ``prepare`` is observable."""

    adapter_kind = "test_provider"
    source_hash = "test"
    uses_autograd_engine = False
    name = "counting_provider"

    def __init__(self, pair):
        self.pair = pair
        self.prepared: list[dict] = []

    def prepare(self, values):
        self.prepared.append(values)
        values["x"].add_(1.0)            # scribbling on its own copy is harmless

    def forward(self, values):
        return self.pair.forward_with_saved(values["x"], values["w"], values["b"])

    def backward(self, dout, saved, values):
        return self.pair.backward_from_saved(dout, saved)


def mul_artifact(*, swap=False):
    """A direct-deployment artifact for ``vt_mul``. With ``swap`` its autograd
    Function routes ``da`` to ``b`` and ``db`` to ``a``; the pair is correct."""
    module = types.ModuleType("vt_mul_artifact" + ("_swapped" if swap else ""))

    def vt_mul_forward_with_saved(a, b):
        return a * b, (a, b)

    def vt_mul_backward_from_saved(dy, saved_tensors):
        a, b = saved_tensors
        return dy * b, dy * a

    class VtMulFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            y, saved = vt_mul_forward_with_saved(a, b)
            ctx.save_for_backward(*saved)
            return y

        @staticmethod
        def backward(ctx, dy):
            da, db = vt_mul_backward_from_saved(dy, ctx.saved_tensors)
            return (db, da) if swap else (da, db)

    class VtMulModule(nn.Module):
        def forward(self, a, b):
            return VtMulFunction.apply(a, b)

    def vt_mul_deployment(a, b):
        return VtMulFunction.apply(a, b)

    for name, value in dict(vt_mul_forward_with_saved=vt_mul_forward_with_saved,
                            vt_mul_backward_from_saved=vt_mul_backward_from_saved,
                            VtMulFunction=VtMulFunction, VtMulModule=VtMulModule,
                            vt_mul_deployment=vt_mul_deployment,
                            DEPLOYMENT_ENTRY="vt_mul_deployment").items():
        setattr(module, name, value)
    return module


# ── a small network with named sites (a model-scope fixture) ────────────────


def norm_site(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def mix_site(h, weight, bias):
    return torch.nn.functional.linear(h, weight, bias)


NATIVE_SITES = {"norm": norm_site, "mix": mix_site}


def _init(generator, *shape, scale=0.3):
    return nn.Parameter(torch.randn(*shape, generator=generator) * scale)


class ToyBlock(nn.Module):
    """Two residual layers; returns ``(y, aux, tag)`` -- ``tag`` is exact metadata."""

    def __init__(self, sites: Mapping[str, Any], width: int = 16, layers: int = 2,
                 skip_last_norm: bool = False):
        super().__init__()
        g = torch.Generator().manual_seed(1234)
        self.sites = dict(sites)
        self.skip_last_norm = skip_last_norm
        self.norm_w = nn.ParameterList([nn.Parameter(1 + 0.1 * torch.randn(width, generator=g))
                                        for _ in range(layers)])
        self.mix_w = nn.ParameterList([_init(g, width, width) for _ in range(layers)])
        self.mix_b = nn.ParameterList([_init(g, width) for _ in range(layers)])
        self.unused = _init(g, 3)            # takes no part: its gradient is absent

    def forward(self, x, *, tag: int = 7):
        for i in range(len(self.norm_w)):
            norm = (norm_site if self.skip_last_norm and i == len(self.norm_w) - 1
                    else self.sites["norm"])
            h = norm(x, self.norm_w[i])
            x = x + self.sites["mix"](h, self.mix_w[i], self.mix_b[i])
        return x, x.pow(2).mean(-1), torch.tensor(tag)


def block_invocation(sample: Sample, device: str) -> Invocation:
    g = torch.Generator().manual_seed(int(sample.seed))
    x = torch.randn(3, 5, 16, generator=g).to(device)
    return Invocation(args=(x,), kwargs={"tag": 7}, differentiable_inputs={"x": "args[0]"})


def block_outputs(result, invocation, sample):
    return {"y": result[0], "aux": result[1], "tag": result[2]}


def toy_module_scope(*, device="cpu", factory=None, calls=None):
    """The toy network at model scope (Gate 2 has no block scope)."""
    from evograd.verification.gate2.adapters.module import ModuleScope

    return ModuleScope(
        scope="model", name="toy_block",
        factory=factory or (lambda sites: ToyBlock(sites)), native_sites=dict(NATIVE_SITES),
        make_invocation=block_invocation, select_outputs=block_outputs,
        output_names=("y", "aux", "tag"), metadata_outputs=("tag",),
        calls_per_execution=calls or {"norm": 2, "mix": 2},
        config={"width": 16, "layers": 2, "batch": [3, 5]}, device=device)


# ── a tiny language model with tied embeddings ──────────────────────────────


class TinyLM(nn.Module):
    def __init__(self, sites: Mapping[str, Any], vocab: int = 32, width: int = 16,
                 layers: int = 2, tie: bool = True, dropout: float = 0.0):
        super().__init__()
        g = torch.Generator().manual_seed(4321)
        self.sites = dict(sites)
        self.embed = nn.Embedding(vocab, width)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, width, generator=g) * 0.5)
        self.norm_w = nn.ParameterList([nn.Parameter(1 + 0.1 * torch.randn(width, generator=g))
                                        for _ in range(layers + 1)])
        self.mix_w = nn.ParameterList([_init(g, width, width) for _ in range(layers)])
        self.mix_b = nn.ParameterList([_init(g, width) for _ in range(layers)])
        self.head = nn.Linear(width, vocab, bias=False)
        if tie:
            self.head.weight = self.embed.weight
        else:
            with torch.no_grad():
                self.head.weight.copy_(torch.randn(vocab, width, generator=g) * 0.5)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for i in range(len(self.mix_w)):
            n = self.sites["norm"](h, self.norm_w[i])
            h = h + self.drop(self.sites["mix"](n, self.mix_w[i], self.mix_b[i]))
        return {"logits": self.head(self.sites["norm"](h, self.norm_w[-1]))}


def lm_batch(seed: int, *, batch: int = 2, seq: int = 9, vocab: int = 32, masked: int = 0):
    g = torch.Generator().manual_seed(int(seed))
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    labels = ids.clone()
    if masked:
        labels[:, -masked:] = -100
    return ids, labels


def tiny_lm_scope(*, dropout=0.0, device="cpu", tie=True):
    from evograd.verification.gate2.adapters.model import model_scope, next_token_mean_nll

    def invocation(sample, device):
        ids, labels = lm_batch(sample.seed)
        return Invocation(args=(ids.to(device),), context={"labels": labels.to(device)})

    return model_scope(
        name="tiny_lm", factory=lambda sites: TinyLM(sites, dropout=dropout, tie=tie),
        native_sites=dict(NATIVE_SITES), make_invocation=invocation,
        forward_outputs=lambda result, inv, sample: {"logits": result["logits"]},
        loss=lambda outputs, inv, sample: next_token_mean_nll(outputs["logits"],
                                                               inv.context["labels"]),
        output_names=("logits",), loss_definition="token-mean next-token NLL",
        calls_per_execution={"norm": 3, "mix": 2},
        config={"vocab": 32, "width": 16, "layers": 2, "tie": tie, "dropout": dropout},
        device=device)


# ── site implementations used as candidates ─────────────────────────────────


def norm_scaled(factor):
    def norm(x, w):
        return norm_site(x, w) * factor
    return norm


def norm_detached(x, w):
    """Correct values; cuts the gradient to its inputs."""
    return norm_site(x.detach(), w.detach())


def norm_spike(x, w):
    """Correct everywhere except one element, bumped by a large relative amount."""
    out = norm_site(x, w)
    bump = torch.zeros_like(out)
    bump.view(-1)[0] = 0.5 * out.detach().abs().max()
    return out + bump


def norm_writes_parameter(x, w):
    with torch.no_grad():
        w.add_(1e-3)
    return norm_site(x, w)


def norm_nan(x, w):
    out = norm_site(x, w)
    return out + torch.where(torch.arange(out.numel()).reshape(out.shape) == 0,
                             torch.tensor(float("nan")), torch.tensor(0.0))


class StatefulNorm:
    """Correct on its first call per input shape; drifts on later calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, w):
        self.calls += 1
        return norm_site(x, w) * (1.0 + 0.05 * (self.calls - 1))


class MisroutedNorm(torch.autograd.Function):
    """Correct forward; the backward returns d/dw scaled -- a gradient-routing fault."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return norm_site(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        with torch.enable_grad():
            xl, wl = x.detach().requires_grad_(True), w.detach().requires_grad_(True)
            out = norm_site(xl, wl)
        dx, dw = torch.autograd.grad(out, (xl, wl), dy)
        return dx, dw * 1.5


def norm_misrouted(x, w):
    return MisroutedNorm.apply(x, w)
