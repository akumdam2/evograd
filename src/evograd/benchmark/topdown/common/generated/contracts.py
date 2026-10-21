"""Executable Level-2 contracts; measured grids require a harvested snapshot."""

import torch
from evograd.opdecl import Active, Workload, Provenance, declare_op
from evograd.benchmark.topdown import has_snapshot, load_snapshot

PREFIX = "evograd.benchmark.topdown.common.generated.model:"
SHAPES = {
    "dit_modulation": (("c", "[B, H]"), ("weight", "[J, H]"), ("bias", "[J]")),
    "dit_mlp": (("x", "[B, T, H]"), ("w1", "[I, H]"), ("b1", "[I]"),
                ("w2", "[H, I]"), ("b2", "[H]")),
    "dlrm_interaction": (("dense", "[B, D]"), ("sparse", "[B, S, D]")),
}
DIMS = {"dit_modulation": dict(B=2, H=12, J=72),
        "dit_mlp": dict(B=2, T=7, H=12, I=48),
        "dlrm_interaction": dict(B=3, S=4, D=8, O=18)}


def _inputs(torch_module, op, workload, device="cuda"):
    from evograd.opdecl.activity import bind_shape
    gen = torch.Generator(device="cpu").manual_seed(17)
    result = {}
    for arg in op.args:
        shape = bind_shape(arg.shape, workload.dims)
        result[arg.name] = torch.randn(shape, generator=gen).to(device=device, dtype=getattr(torch, workload.dtype))
    result["dout"] = torch.randn(bind_shape(op.outputs[0].shape, workload.dims), generator=gen).to(
        device=device, dtype=getattr(torch, workload.dtype))
    return result


def _observed(model, name):
    if not has_snapshot(model):
        return ()
    return tuple(Workload(dims=c["dims"], dtype=c["dtype"],
                          provenance=Provenance(model=model,
                                                component=(c.get("provenance") or {}).get("component",
                                                                                          name),
                                                source="execution",
                                                note=f"Observed {c['config_id']}"))
                 for c in load_snapshot(model)["tasks"].get(name, {}).get("configurations", ()))


def declaration(model, site):
    name = model + "_" + site
    observed = _observed(model, name)
    example = tuple(Workload(dims=DIMS[site], dtype=dtype,
                             provenance=Provenance(model=model, component=site,
                                                   source="handpicked", scaled=True,
                                                   note="Reduced correctness fixture, not a timing case"))
                    for dtype in ("float32", "bfloat16"))
    output = {"dit_modulation": "[B, J]", "dit_mlp": "[B, T, H]",
              "dlrm_interaction": "[B, O]"}[site]
    return declare_op(name=name, level=2, family={"dit_modulation": "normalization",
                                                  "dit_mlp": "mlp",
                                                  "dlrm_interaction": "interaction"}[site],
                      forward=PREFIX + site, dims=tuple(DIMS[site]),
                      args=tuple(Active(n, s) for n, s in SHAPES[site]),
                      output=Active("out", output),
                      parameter_args=tuple(n for n, _ in SHAPES[site][1:]) if site.startswith("dit_") else (),
                      forward_semantics={"dit_modulation": "SiLU conditioning projection to six adaLN-Zero shift, scale, and gate vectors",
                                         "dit_mlp": "Biased linear, tanh-approximate GELU, biased linear",
                                         "dlrm_interaction": "Dense vector and strict triangular pairwise dot products"}[site],
                      backward_semantics="Autograd gradients for every active input and parameter",
                      correctness=example, coverage=observed, benchmark=observed,
                      benchmark_suites={model + "_observed": observed},
                      tolerances={"float32": (2e-5, 2e-5), "bfloat16": (0.02, 0.02)},
                      make_inputs=_inputs)
