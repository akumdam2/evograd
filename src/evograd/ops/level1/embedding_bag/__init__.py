"""Summed embedding bags with jagged offsets and dense weight gradients."""

from evograd.opdecl import Active, Inactive, Workload, declare_op


def make_inputs(torch, op, workload, device="cuda"):
    d = workload.dims
    generator = torch.Generator(device="cpu").manual_seed(37)
    weight = (torch.randn(d["N"], d["D"], generator=generator) * 0.02).to(
        device=device, dtype=getattr(torch, workload.dtype))
    indices = torch.randint(d["N"], (d["L"],), generator=generator).to(device)
    offsets = torch.arange(0, d["L"], d["M"], dtype=torch.long, device=device)
    dout = torch.randn(d["B"], d["D"], generator=generator).to(
        device=device, dtype=getattr(torch, workload.dtype))
    return dict(weight=weight, indices=indices, offsets=offsets, dout=dout)


op = declare_op(
    name="embedding_bag", level=1, family="embedding",
    forward="evograd.ops.level1.embedding_bag.reference:forward",
    dims=("N", "D", "B", "M", "L"),
    args=(Active("weight", "[N, D]"), Inactive("indices", "[L]", dtype="int64"),
          Inactive("offsets", "[B]", dtype="int64")),
    output=Active("out", "[B, D]"), parameter_args=("weight",),
    forward_semantics="Sum embedding rows between consecutive offsets; include the final bag through the end of indices. Fixed M entries per bag, L=B*M.",
    backward_semantics="Accumulate a dense weight gradient for repeated indices; integer indices and offsets are inactive.",
    correctness=tuple(Workload(dims=dict(N=n, D=d, B=b, M=m, L=b*m), dtype=dtype)
                      for n, d, b, m, dtype in ((17, 8, 3, 1, "float32"),
                                               (11, 7, 4, 3, "float32"),
                                               (17, 8, 3, 2, "bfloat16"))),
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (0.02, 0.02)},
    make_inputs=make_inputs,
)
