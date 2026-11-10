"""MegaFold's site boundaries: what one kernel receives at each place in the tree.

A :class:`Boundary` states one site once, and three things read it:

* the deployment adapter (:class:`CompositeAdapter`, or a class of its own in
  :mod:`.workload` for the older sites), which calls the kernel;
* the recorder (``verification.integrations.megafold.capture``), which watches
  the eager module pass through and records the same arguments;
* ``expected_site_calls``, which counts the module's calls.

So a recorded call is exactly what a deployed kernel receives: the arguments
after the flattening, views and autocast casts of ``docs/AF3.md`` section 2. Every
tensor is built with the module's own parameters and left in the graph, so a
gradient reaches the float32 parameter as autocast's would. Nothing here touches
MegaFold until a model is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import nn

#: LayerNorm epsilon. MegaFold's fused kernel hardcodes it and its own torch
#: fallback takes ``F.layer_norm``'s default, so both spellings agree on 1e-5.
LAYER_NORM_EPS = 1e-5


def _autocast_dtype(t) -> torch.dtype | None:
    """The dtype an enclosing autocast runs reduced-precision ops in, or None."""
    kind = t.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None


def _no_autocast(t):
    """A kernel sees exactly the dtypes its contract declares, never autocast's rewrites."""
    return torch.autocast(device_type=t.device.type, enabled=False)


def _sp_size() -> int:
    from megafold.distributed.parallel_info import get_parallel_info

    return get_parallel_info()["sp_size"]


def _require_one_rank() -> None:
    """Every collective inside a boundary is a no-op at group size one, and the references omit them."""
    if _sp_size() != 1:
        raise ValueError(f"MegaFold's site contracts omit its collectives, which needs sp_size 1, "
                         f"not {_sp_size()}")


def gemm_dtype(t) -> torch.dtype:
    """The case dtype: autocast's where it is on, else the tensor's own."""
    return _autocast_dtype(t) or t.dtype


def norm_input(t):
    """``t`` as a LayerNorm receives it: autocast runs ``layer_norm`` in float32."""
    return t.float() if _autocast_dtype(t) is not None else t


def _rows(t):
    return t.reshape(-1, t.shape[-1])


def _same(t, _inputs, _kwargs):
    return t


def structured_keep(dropout, like, dtype):
    """The scaled mask ``Dropout.forward`` draws for row or column dropout of ``like``.

    The same call on the same shape and dtype, so the step consumes RNG as it would.
    """
    b, rows, cols, d = like.shape
    if dropout.dropout_type == "row":
        shape = (b, 1, cols, d)
    elif dropout.dropout_type == "col":
        shape = (b, rows, 1, d)
    else:
        raise ValueError(f"a keep mask is row or column dropout, not {dropout.dropout_type!r}")
    return dropout.dropout(like.new_ones(shape, dtype=dtype))


@dataclass(frozen=True)
class Boundary:
    """One site: which module it is and what its kernel is called with."""

    site: str
    task: str
    #: ``(module, inputs, kwargs, tapped) -> {argument: value}``, in the task's order.
    arguments: Callable[..., Mapping[str, Any]]
    #: The module whose forward is the boundary, given what the site's matcher took.
    module: Callable[[nn.Module], nn.Module] = lambda m: m
    #: The submodule whose output the boundary needs besides its inputs (a keep mask).
    tap: Callable[[nn.Module], nn.Module] | None = None
    #: ``(module, inputs, kwargs) -> value``: what a deployed adapter draws in the tap's place.
    draw: Callable[..., Any] | None = None
    #: ``(result, inputs, kwargs) -> output``: the kernel's result as the module returns it.
    restore: Callable[..., Any] = _same
    #: ``(output or its gradient, inputs, kwargs) -> value`` in the kernel's layout.
    output: Callable[..., Any] = _same
    #: False where the primitive recorder already sees the call.
    recorded: bool = True


class CompositeAdapter(nn.Module):
    """A module's whole boundary through one kernel.

    The original's submodules, parameters and plain attributes stay under their
    names, so a gradient compared by parameter name finds the same parameter in
    either tree. Their forwards no longer run: the kernel computes the boundary.
    """

    def __init__(self, original: nn.Module, kernel: Callable, boundary: Boundary):
        super().__init__()
        _require_one_rank()
        for name, child in original.named_children():
            self.add_module(name, child)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, value in vars(original).items():
            if not name.startswith("_") and name != "training":
                setattr(self, name, value)
        self.kernel = kernel
        self.boundary = boundary

    def forward(self, *inputs, **kwargs):
        boundary = self.boundary
        drawn = None if boundary.draw is None else boundary.draw(self, inputs, kwargs)
        values = boundary.arguments(self, inputs, kwargs, drawn)
        anchor = next(v for v in values.values() if torch.is_tensor(v))
        with _no_autocast(anchor):
            result = self.kernel(*values.values())
        return boundary.restore(result, inputs, kwargs)


def composite(boundary: Boundary) -> Callable[[nn.Module, Callable], nn.Module]:
    """``ModulePatch.build`` for a boundary replaced whole."""
    return lambda original, kernel: CompositeAdapter(original, kernel, boundary)


# -- the sites MegaFold fuses ------------------------------------------------


def layernorm_linear_arguments(module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    """Leading axes flattened; float32 rows and the weight in the GEMM's dtype under autocast."""
    x = inputs[0]
    rows, weight, dtype = _rows(x), module.linear_weight, _autocast_dtype(x)
    if dtype is not None:
        rows, weight = rows.float(), weight.to(dtype)
    return {"x": rows, "weight": module.WEIGHT, "bias": module.BIAS, "linear_weight": weight,
            "eps": LAYER_NORM_EPS}


def _attend_layout(q, batch_size) -> tuple[int, int]:
    """``(B, S)`` of Attend's folded first axis.

    MegaFold folds the leading axes into one batch before ``Attend``, ``[b, h, n,
    d]`` with ``b = batch_size * n1``, and its own Triton path unfolds them again
    right before the kernel: ``(b n1) h n2 d -> b n1 n2 h d``. The declaration's
    ``S`` is that ``n1`` axis, so ``S == 1`` over the single representation and
    ``S == n1`` for triangle attention. Folding ``n1`` into ``B`` instead would run
    the kernel at a shape no tier measured and hand it ``S`` copies of the bias.
    """
    batch = q.shape[0]
    b = batch_size if batch_size and batch % batch_size == 0 else batch
    return b, batch // b


def attend_arguments(module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    """``Attend``'s tensors as ``megafold_evoattention`` takes them, views of the same memory."""
    q, k, v = inputs[:3]
    if kwargs.get("windowed_mask") is not None or kwargs.get("memory_kv") is not None:
        raise ValueError("evoattention site received windowed/memory-kv attention inputs")
    mask, attn_bias = kwargs.get("mask"), kwargs.get("attn_bias")
    batch, heads, n, _d = q.shape
    b, s = _attend_layout(q, kwargs.get("batch_size", 1))
    if attn_bias is not None and attn_bias.dtype != q.dtype:
        raise ValueError(f"attention bias in {attn_bias.dtype} with q in {q.dtype}: "
                         "MegaFold adds them in one dtype")

    def declared(t):  # [(B S), h, n, d] view -> [B, S, N, H, D] view of the same memory
        return t.permute(0, 2, 1, 3).unflatten(0, (b, s))

    if attn_bias is None:
        pair_bias = q.new_zeros((b, 1, heads, n, n))
    else:
        # One [b, h, n, n] bias per batch element, broadcast over S by the task,
        # which reduces d_pair_bias over S itself. Triangle attention arrives with
        # that bias repeated over its rows, so the site passes the first row (a
        # view): autograd's gradient into it is the S-sum the repeat's backward forms.
        pair_bias = attn_bias.expand(batch, heads, n, n).reshape(b, s, heads, n, n)[:, :1]
    # MegaFold broadcasts one key mask over the rows (stride 0); the kernel gets it
    # materialized, a few hundred KB, so no argument overlaps itself.
    key_mask = q.new_ones((b, s, n), dtype=torch.bool) if mask is None else mask.view(b, s, n).contiguous()
    return {"q": declared(q), "k": declared(k), "v": declared(v), "key_mask": key_mask,
            "pair_bias": pair_bias}


def _attend_output(t, inputs, kwargs):
    b, s = _attend_layout(inputs[0], kwargs.get("batch_size", 1))
    return t.permute(0, 2, 1, 3).unflatten(0, (b, s))


def swiglu_arguments(_module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    """MegaFold chunks ``[x, gates]`` and computes ``silu(gates) * x``; the task's ``a`` is ``gates``."""
    first, gates = inputs[0].chunk(2, dim=-1)
    return {"a": _rows(gates), "b": _rows(first)}


def _layer_norm_arguments(module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    rows = norm_input(_rows(inputs[0]))
    return {"x": rows, "weight": module.weight, "bias": module.bias, "eps": module.eps}


def _rows_of(t, _inputs, _kwargs):
    return _rows(t)


# -- the Level-2 contracts ---------------------------------------------------


def _pair_mask_input(mask, like, rows):
    """A key mask as a kernel takes it: all True where MegaFold passes none."""
    return like.new_ones((rows, like.shape[-2]), dtype=torch.bool) if mask is None else mask


def _keyword(inputs, kwargs, position, name):
    return inputs[position] if len(inputs) > position else kwargs.get(name)


def triangle_multiplication_arguments(module, inputs, kwargs, keep) -> dict[str, Any]:
    x, mask = inputs[0], _keyword(inputs, kwargs, 1, "mask")
    combined, projection, dtype = module.combined, module.to_out[0], gemm_dtype(x)
    return {"x": norm_input(x), "ln_weight": combined.WEIGHT, "ln_bias": combined.BIAS,
            "linear_weight": combined.linear_weight.to(dtype), "linear_bias": combined.linear_bias.to(dtype),
            "out_ln_weight": projection.WEIGHT, "out_ln_bias": projection.BIAS,
            "out_linear_weight": projection.linear_weight.to(dtype),
            "mask": _pair_mask_input(mask, x, x.shape[0]), "keep": keep, "eps": LAYER_NORM_EPS}


def _triangle_multiplication(mix: str) -> Boundary:
    return Boundary(f"triangle_multiplication_{mix}", f"megafold_triangle_multiplication_{mix}",
                    triangle_multiplication_arguments, tap=lambda m: m.to_out[1].dropout,
                    draw=lambda m, inputs, _kwargs: structured_keep(m.to_out[1], inputs[0],
                                                                    gemm_dtype(inputs[0])))


def _attention_weights(attention, dtype) -> dict[str, Any]:
    """``Attention``'s projections in the GEMM's dtype."""
    return {"q_weight": attention.to_q.weight.to(dtype), "q_bias": attention.to_q.bias.to(dtype),
            "kv_weight": attention.to_kv.weight.to(dtype),
            "gate_weight": attention.to_gates.weight.to(dtype),
            "gate_bias": attention.to_gates.bias.to(dtype),
            "out_weight": attention.to_out.weight.to(dtype)}


def triangle_attention_arguments(module, inputs, kwargs, keep) -> dict[str, Any]:
    x, mask = inputs[0], _keyword(inputs, kwargs, 1, "mask")
    dtype = gemm_dtype(x)
    return {"x": norm_input(x), "ln_weight": module.norm.weight, "ln_bias": module.norm.bias,
            "bias_weight": module.fn.to_attn_bias[0].weight.to(dtype),
            **_attention_weights(module.fn.attn, dtype),
            "mask": _pair_mask_input(mask, x, x.shape[0]), "keep": keep, "eps": module.norm.eps}


def _triangle_attention(node: str) -> Boundary:
    return Boundary(f"triangle_attention_{node}", f"megafold_triangle_attention_{node}",
                    triangle_attention_arguments, tap=lambda m: m.fn.dropout.dropout,
                    draw=lambda m, inputs, _kwargs: structured_keep(m.fn.dropout, inputs[0],
                                                                    gemm_dtype(inputs[0])))


def transition_arguments(module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    x, projection, dtype = inputs[0], module.ff[0], gemm_dtype(inputs[0])
    return {"x": norm_input(_rows(x)), "weight": projection.WEIGHT, "bias": projection.BIAS,
            "linear_weight": projection.linear_weight.to(dtype),
            "out_weight": module.ff[2].weight.to(dtype), "eps": LAYER_NORM_EPS}


def _restore_rows(result, inputs, _kwargs):
    return result.reshape(*inputs[0].shape[:-1], result.shape[-1])


def outer_product_mean_arguments(module, inputs, kwargs, _tapped=None) -> dict[str, Any]:
    """The step always passes an MSA mask; the branch without one divides by something else."""
    x = _keyword(inputs, kwargs, 0, "msa")
    if kwargs.get("msa_mask") is None:
        raise ValueError("outer_product_mean models the MSA mask branch, which the step always takes")
    msa_mask, msa_count = kwargs["msa_mask"]
    hidden, projection, dtype = module.to_hidden, module.to_pairwise_repr, gemm_dtype(x)
    return {"x": norm_input(x), "weight": hidden.WEIGHT, "bias": hidden.BIAS,
            "linear_weight": hidden.linear_weight.to(dtype), "pair_weight": projection.weight.to(dtype),
            "pair_bias": projection.bias.to(dtype),
            "mask": _pair_mask_input(kwargs.get("mask"), x, x.shape[0]), "msa_mask": msa_mask.to(dtype),
            "msa_count": msa_count, "eps": LAYER_NORM_EPS, "count_eps": module.eps}


def msa_pair_weighted_averaging_arguments(module, inputs, kwargs, keep) -> dict[str, Any]:
    x, pair = kwargs["msa"], kwargs["pairwise_repr"]
    hidden, attention, dtype = module.msa_to_values_and_gates[0], module.pairwise_repr_to_attn[0], gemm_dtype(x)
    return {"x": norm_input(x), "ln_weight": hidden.WEIGHT, "ln_bias": hidden.BIAS,
            "linear_weight": hidden.linear_weight.to(dtype), "pair": norm_input(pair),
            "pair_ln_weight": attention.WEIGHT, "pair_ln_bias": attention.BIAS,
            "pair_linear_weight": attention.linear_weight.to(dtype),
            "out_weight": module.to_out[1].weight.to(dtype),
            "mask": _pair_mask_input(kwargs.get("mask"), x, x.shape[0]), "keep": keep,
            "eps": LAYER_NORM_EPS}


BOUNDARIES: dict[str, Boundary] = {
    # Every LayerNorm call is seen as a primitive, so this one is only counted.
    "layer_norm": Boundary("layer_norm", "layernorm", _layer_norm_arguments, module=lambda m: m.norm,
                           recorded=False),
    "layernorm_linear": Boundary("layernorm_linear", "megafold_layernorm_linear",
                                 layernorm_linear_arguments, output=_rows_of),
    "transition": Boundary("transition", "swiglu", swiglu_arguments, module=lambda m: m.ff[1],
                           output=_rows_of),
    "pair_bias_attention": Boundary("pair_bias_attention", "megafold_evoattention", attend_arguments,
                                    module=lambda m: m.attn.attend, output=_attend_output),
    "triangle_attention": Boundary("triangle_attention", "megafold_evoattention", attend_arguments,
                                   module=lambda m: m.attn.attend, output=_attend_output),
    "triangle_multiplication_outgoing": _triangle_multiplication("outgoing"),
    "triangle_multiplication_incoming": _triangle_multiplication("incoming"),
    "triangle_attention_starting": _triangle_attention("starting"),
    "triangle_attention_ending": _triangle_attention("ending"),
    "transition_block": Boundary("transition_block", "megafold_transition", transition_arguments,
                                 restore=_restore_rows, output=_rows_of),
    "outer_product_mean": Boundary("outer_product_mean", "megafold_outer_product_mean",
                                   outer_product_mean_arguments),
    "msa_pair_weighted_averaging": Boundary(
        "msa_pair_weighted_averaging", "megafold_msa_pair_weighted_averaging",
        msa_pair_weighted_averaging_arguments, tap=lambda m: m.to_out[2].dropout,
        draw=lambda m, _inputs, kwargs: structured_keep(m.to_out[2], kwargs["msa"],
                                                        gemm_dtype(kwargs["msa"]))),
}


def composite_patches(model_module, inside, transitions=()) -> tuple:
    """One ``ModulePatch`` per site replaced whole, on MegaFold's real classes.

    ``inside`` keeps a matcher out of ``workload.NOT_SITES``. Each matcher takes
    both spellings of the tree: ``transitions`` is ``FusedTransition`` and its
    eager replacement, and the other classes are the same class in both.
    """
    from evograd.evaluation.tier3.patch import ModulePatch

    PreLayerNorm = model_module.PreLayerNorm
    TriangleAttention = model_module.TriangleAttention
    TriangleMultiplication = model_module.TriangleMultiplication

    def triangle_multiplication(mix):
        site = f"triangle_multiplication_{mix}"
        return ModulePatch(site, inside(lambda m: type(m) is TriangleMultiplication and m.mix == mix),
                           composite(BOUNDARIES[site]))

    def triangle_attention(node):
        site = f"triangle_attention_{node}"
        ending = node == "ending"
        return ModulePatch(site, inside(lambda m: type(m) is PreLayerNorm
                                        and type(m.fn) is TriangleAttention
                                        and m.fn.need_transpose == ending),
                           composite(BOUNDARIES[site]))

    def has_layernorm(module):
        # ``FusedTransition`` records no flag: its first entry is a LayernormLinear
        # with a LayerNorm, or a bare projection with ``include_ln=False``, which
        # stays stock (``docs/AF3.md`` section 10).
        return hasattr(module.ff[0], "WEIGHT")

    return (triangle_multiplication("outgoing"), triangle_multiplication("incoming"),
            triangle_attention("starting"), triangle_attention("ending"),
            ModulePatch("transition_block",
                        inside(lambda m: type(m) in transitions and has_layernorm(m)),
                        composite(BOUNDARIES["transition_block"])),
            ModulePatch("outer_product_mean", inside(lambda m: type(m) is model_module.OuterProductMean),
                        composite(BOUNDARIES["outer_product_mean"])),
            ModulePatch("msa_pair_weighted_averaging",
                        inside(lambda m: type(m) is model_module.MSAPairWeightedAveraging),
                        composite(BOUNDARIES["msa_pair_weighted_averaging"])))
