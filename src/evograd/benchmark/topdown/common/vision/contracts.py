"""Model boundaries as pair contracts, with execution-derived coverage.

The observed benchmark/coverage suites come from the tracked snapshot (the
harvest of the verification run); a model without one, or a snapshot that
predates a task, has none.
"""

from dataclasses import dataclass

from evograd.opdecl import Active, Inactive, Workload, Provenance, declare_op
from evograd.benchmark.topdown import has_snapshot, load_snapshot

PREFIX = "evograd.benchmark.topdown.common.vision.operators:"
BACKWARD = ("Return gradients for every active argument in declared order, including all "
            "parameters. Accumulate reductions in float32.")
RESIDUAL_BACKWARD = (" The backward receives (dnormed, dsummed): dsummed is the gradient that "
                     "reaches the sum from the rest of the stream (zero after the last block), "
                     "so dx = dr = dsummed + the normalization's input gradient.")


@dataclass(frozen=True)
class Boundary:
    """One Level-2 boundary: its arguments and outputs in order, which arguments
    are module parameters, and what it computes."""

    family: str
    args: tuple
    outputs: tuple
    parameters: tuple
    semantics: str
    backward: str = BACKWARD

    @property
    def dims(self) -> tuple:
        """The shape symbols, in order of first appearance."""
        seen = []
        for arg in (*self.args, *self.outputs):
            for token in _tokens(getattr(arg, "shape", None)):
                if not token.isdigit() and token not in seen:
                    seen.append(token)
        return tuple(seen)


def _tokens(shape):
    return [t.strip() for t in shape.strip()[1:-1].split(",") if t.strip()] if shape else []


def _boundary(family, args, outputs, semantics, **kwargs):
    """Every Active argument but the first is a parameter unless ``parameters`` says otherwise."""
    parameters = kwargs.pop("parameters", tuple(a.name for a in args[1:] if isinstance(a, Active)))
    return Boundary(family, tuple(args), tuple(outputs), tuple(parameters), semantics, **kwargs)


BOUNDARIES = {
    "convnext_mlp": _boundary(
        "mlp",
        (Active("x", "[B, Y, X, H]"), Active("up_weight", "[I, H]"), Active("up_bias", "[I]"),
         Active("grn_weight", "[1, 1, 1, I]"), Active("grn_bias", "[1, 1, 1, I]"),
         Active("down_weight", "[H, I]"), Active("down_bias", "[H]")),
        (Active("out", "[B, Y, X, H]"),),
        "NHWC linear, exact GELU, spatial GRN (eps 1e-6), linear; biases included."),
    "dwconv_layernorm": _boundary(
        "convolution",
        (Active("x", "[B, C, Y, X]"), Active("dw_weight", "[C, 1, K, K]"), Active("dw_bias", "[C]"),
         Active("ln_weight", "[C]"), Active("ln_bias", "[C]"), Inactive("eps", default=1e-6)),
        (Active("out", "[B, Y, X, C]"),),
        "Depthwise KxK convolution with bias on the NCHW input (stride 1, padding K // 2, "
        "groups C), permuted to NHWC, then LayerNorm over C with weight, bias and eps."),
    "text_mlp": _boundary(
        "mlp",
        (Active("x", "[B, T, H]"), Active("gate_weight", "[I, H]"), Active("up_weight", "[I, H]"),
         Active("down_weight", "[H, I]")),
        (Active("out", "[B, T, H]"),),
        "Biasless gate/up projections, SiLU(gate)*up, down projection."),
    "vision_mlp": _boundary(
        "mlp",
        (Active("x", "[T, H]"), Active("up_weight", "[I, H]"), Active("up_bias", "[I]"),
         Active("down_weight", "[H, I]"), Active("down_bias", "[H]")),
        (Active("out", "[T, H]"),),
        "Biased linear, QuickGELU x*sigmoid(1.702*x), biased linear."),
    "vision_attention": _boundary(
        "attention",
        (Active("x", "[T, H]"), Active("qkv_weight", "[QKV, H]"), Active("qkv_bias", "[QKV]"),
         Inactive("cos", "[T, D]", dtype="float32"), Inactive("sin", "[T, D]", dtype="float32"),
         Active("proj_weight", "[H, H]"), Active("proj_bias", "[H]")),
        (Active("out", "[T, H]"),),
        "Biased qkv projection to H // D heads of size D (qkv viewed [T, 3, heads, D]); "
        "rotate-half RoPE on q and k computed in float32 with the float32 cos/sin tables and "
        "cast back; non-causal SDPA over all T tokens of the one image, scale D**-0.5; heads "
        "merged to [T, H]; biased output projection."),
    "vision_residual_layernorm": _boundary(
        "norm",
        (Active("x", "[T, H]"), Active("r", "[T, H]"), Active("weight", "[H]"),
         Active("bias", "[H]"), Inactive("eps", default=1e-6)),
        (Active("normed", "[T, H]"), Active("summed", "[T, H]")),
        "summed = r + x (the residual stream plus the attention or MLP branch); normed = "
        "LayerNorm(summed) with weight, bias and eps over H. Returns (normed, summed).",
        parameters=("weight", "bias"), backward=BACKWARD + RESIDUAL_BACKWARD),
    "text_attention_block": _boundary(
        "attention",
        (Active("x", "[B, T, H]"), Active("q_weight", "[H, H]"), Active("q_bias", "[H]"),
         Active("k_weight", "[KV, H]"), Active("k_bias", "[KV]"), Active("v_weight", "[KV, H]"),
         Active("v_bias", "[KV]"), Inactive("cos", "[B, T, D]"), Inactive("sin", "[B, T, D]"),
         Inactive("attention_mask", "[B, 1, T, T]", dtype="bool"), Active("o_weight", "[H, H]")),
        (Active("out", "[B, T, H]"),),
        "Biased q/k/v projections to H // D query and KV // D key/value heads of size D; "
        "multimodal RoPE: rotate-half with cos/sin (the M-RoPE tables after their section "
        "interleave, in the model dtype) on q and k; GQA SDPA with the boolean attention_mask "
        "(causal and right-padding, True = attend), key/value heads repeated, scale D**-0.5; "
        "heads merged to [B, T, H]; biasless output projection."),
    "text_residual_rmsnorm": _boundary(
        "norm",
        (Active("x", "[B, T, H]"), Active("r", "[B, T, H]"), Active("weight", "[H]"),
         Inactive("eps", default=1e-6)),
        (Active("normed", "[B, T, H]"), Active("summed", "[B, T, H]")),
        "summed = r + x (the residual stream plus the attention or MLP branch); normed = "
        "weight * (float32 summed * rsqrt(mean(summed**2, H) + eps)) cast to the input dtype "
        "before the weight product. Returns (normed, summed).",
        parameters=("weight",), backward=BACKWARD + RESIDUAL_BACKWARD),
}


def dims_of(args, values):
    """The dims ``values`` bind ``args``' shapes to (a shape mismatch raises)."""
    dims: dict = {}
    for arg, value in zip(args, values):
        tokens = _tokens(getattr(arg, "shape", None))
        if not tokens:
            continue
        if len(tokens) != value.dim():
            raise ValueError(f"{arg.name}: rank {value.dim()} for shape {arg.shape}")
        for token, size in zip(tokens, value.shape):
            bound = int(token) if token.isdigit() else dims.setdefault(token, int(size))
            if bound != size:
                raise ValueError(f"{arg.name}: {token} is {size}, bound to {bound}")
    return dims


def observed(model, task):
    if not has_snapshot(model):
        return ()
    entry = load_snapshot(model)["tasks"].get(task)
    if entry is None:
        return ()
    return tuple(
        Workload(
            dims=c["dims"],
            dtype=c["dtype"],
            provenance=Provenance(
                model=model,
                component=(c.get("provenance") or {}).get("component", task),
                free=dict((c.get("provenance") or {}).get("free") or {}),
                source="execution",
                note=f"Execution observed: {c['config_id']}; see hashed harvest snapshot",
                strides=tuple((k, tuple(v)) for k, v in c["strides"].items()),
                layout="strided",
            ),
        )
        for c in entry["configurations"]
    )


def make_inputs(torch, op, workload, device="cuda"):
    from evograd.opdecl.activity import bind_shape as resolve_shape
    from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

    g = torch.Generator(device="cpu").manual_seed(17)
    free = dict(getattr(workload.provenance, "free", None) or {})
    values, angles = {}, None
    for arg in op.args:
        if getattr(arg, "shape", None) is None:
            values[arg.name] = free.get(arg.name, arg.default)
            continue
        shape = resolve_shape(arg.shape, workload.dims)
        dtype = resolve_dtype(arg_dtype_name(arg, workload))
        if arg.name in ("cos", "sin"):
            angles = torch.randn(shape, generator=g) * 3 if angles is None else angles
            value = angles.cos() if arg.name == "cos" else angles.sin()
        elif arg.name == "attention_mask":
            # Causal, with the last quarter of the keys right padding.
            t = shape[-1]
            keys = torch.arange(t)
            value = (keys[None, :] <= keys[:, None]) & (keys < t - t // 4)
            value = value.expand(shape).clone()
        else:
            value = torch.randn(shape, generator=g) * (
                shape[-1] ** -0.5 if "weight" in arg.name else 0.5
            )
        values[arg.name] = value.to(device=device, dtype=dtype)
    for out in op.outputs:
        values[out.grad_name] = (
            torch.randn(resolve_shape(out.shape, workload.dims), generator=g) * 0.5
        ).to(device=device, dtype=getattr(torch, workload.dtype))
    return values


def declaration(model, site):
    name = f"{model}_{site}"
    boundary = BOUNDARIES[site]
    bench = observed(model, name)
    return declare_op(
        name=name,
        level=2,
        family=boundary.family,
        forward=PREFIX + site,
        dims=boundary.dims,
        args=boundary.args,
        output=boundary.outputs[0] if len(boundary.outputs) == 1 else boundary.outputs,
        parameter_args=boundary.parameters,
        forward_semantics=boundary.semantics,
        backward_semantics=boundary.backward,
        coverage=bench,
        benchmark=bench,
        benchmark_suites={model + "_observed": bench},
        make_inputs=make_inputs,
    )
