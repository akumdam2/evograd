from evograd.opdecl import Active, Inactive, declare_op


def make_inputs(torch, op, workload, device="cuda"):
    from evograd.opdecl.activity import bind_shape

    d = workload.dims
    g = torch.Generator(device="cpu").manual_seed(31)
    result = {
        a.name: (torch.randn(bind_shape(a.shape, d), generator=g) * 0.5).to(
            device=device, dtype=getattr(torch, workload.dtype)
        )
        for a in (*op.active_args(), *op.outputs)
    }
    result["dout"] = result.pop("out")
    result.update(stride=d["S"], padding=d["P"], groups=d["G"])
    return result


op = declare_op(
    name="spatial_conv2d",
    level=1,
    family="convolution",
    forward="evograd.ops.level1.spatial_conv2d.reference:forward",
    dims=("B", "C", "Y", "X", "O", "CG", "K", "OY", "OX", "S", "P", "G"),
    args=(
        Active("x", "[B, C, Y, X]"),
        Active("weight", "[O, CG, K, K]"),
        Active("bias", "[O]"),
        Inactive("stride", default=1),
        Inactive("padding", default=0),
        Inactive("groups", default=1),
    ),
    output=Active("out", "[B, O, OY, OX]"),
    parameter_args=("weight", "bias"),
    forward_semantics="Biased NCHW square Conv2D, declared stride/padding/groups, dilation 1. C=CG*G. O and C divisible by G.",
    backward_semantics="Return dx, dweight, dbias in that order; respect groups and boundary padding. Accumulate reductions in float32.",
    make_inputs=make_inputs,
)
