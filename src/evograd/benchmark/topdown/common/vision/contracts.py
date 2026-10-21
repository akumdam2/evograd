"""Model boundaries as pair contracts, with execution-derived coverage.

Until a canonical harvest is recorded, benchmark/coverage suites are empty.
The small correctness grid is deliberately labelled handpicked.
"""

from evograd.opdecl import Active, Workload, Provenance, declare_op
from evograd.benchmark.topdown import has_snapshot, load_snapshot

PREFIX = "evograd.benchmark.topdown.common.vision.operators:"
SHAPES = {
    "convnext_mlp": [
        ("x", "[B, Y, X, H]"),
        ("up_weight", "[I, H]"),
        ("up_bias", "[I]"),
        ("grn_weight", "[1, 1, 1, I]"),
        ("grn_bias", "[1, 1, 1, I]"),
        ("down_weight", "[H, I]"),
        ("down_bias", "[H]"),
    ],
    "text_mlp": [
        ("x", "[B, T, H]"),
        ("gate_weight", "[I, H]"),
        ("up_weight", "[I, H]"),
        ("down_weight", "[H, I]"),
    ],
    "vision_mlp": [
        ("x", "[T, H]"),
        ("up_weight", "[I, H]"),
        ("up_bias", "[I]"),
        ("down_weight", "[H, I]"),
        ("down_bias", "[H]"),
    ],
}
DIMS = {
    "convnext_mlp": {"B": 1, "Y": 3, "X": 5, "H": 8, "I": 32},
    "text_mlp": {"B": 1, "T": 7, "H": 16, "I": 32},
    "vision_mlp": {"T": 7, "H": 16, "I": 32},
}


def observed(model, task):
    if not has_snapshot(model):
        return ()
    entry = load_snapshot(model)["tasks"][task]
    return tuple(
        Workload(
            dims=c["dims"],
            dtype=c["dtype"],
            provenance=Provenance(
                model=model,
                component=(c.get("provenance") or {}).get("component", task),
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

    g = torch.Generator(device="cpu").manual_seed(17)
    values = {}
    for arg in op.args:
        shape = resolve_shape(arg.shape, workload.dims)
        value = torch.randn(shape, generator=g) * (
            shape[-1] ** -0.5 if "weight" in arg.name else 0.5
        )
        values[arg.name] = value.to(device=device, dtype=getattr(torch, workload.dtype))
    values["dout"] = (
        torch.randn(resolve_shape(op.outputs[0].shape, workload.dims), generator=g) * 0.5
    ).to(device=device, dtype=getattr(torch, workload.dtype))
    return values


def declaration(model, site):
    name = f"{model}_{site}"
    provenance = Provenance(
        model=model,
        component=site,
        source="handpicked",
        scaled=True,
        note="Reduced correctness fixture; not an observed model configuration",
    )
    cases = tuple(
        Workload(dims=DIMS[site], dtype=dtype, provenance=provenance)
        for dtype in ("float32", "bfloat16")
    )
    bench = observed(model, name)
    return declare_op(
        name=name,
        level=2,
        family="mlp",
        forward=PREFIX + site,
        dims=tuple(DIMS[site]),
        args=tuple(Active(n, s) for n, s in SHAPES[site]),
        output=Active("out", SHAPES[site][0][1]),
        parameter_args=tuple(n for n, _ in SHAPES[site][1:]),
        forward_semantics={
            "convnext_mlp": "NHWC linear, exact GELU, spatial GRN (eps 1e-6), linear; biases included.",
            "text_mlp": "Biasless gate/up projections, SiLU(gate)*up, down projection.",
            "vision_mlp": "Biased linear, QuickGELU x*sigmoid(1.702*x), biased linear.",
        }[site],
        backward_semantics="Return gradients for every active argument in declared order, including all parameters. Accumulate reductions in float32.",
        correctness=cases,
        coverage=bench,
        benchmark=bench,
        benchmark_suites={model + "_observed": bench},
        tolerances={"float32": (2e-5, 2e-5), "bfloat16": (0.02, 0.02)},
        make_inputs=make_inputs,
    )
