from evograd.opdecl import Active, Workload, declare_op

op = declare_op(
    name="grn",
    level=1,
    family="normalization",
    forward="evograd.ops.level1.grn.reference:forward",
    dims=("B", "Y", "X", "H"),
    args=(
        Active("x", "[B, Y, X, H]"),
        Active("weight", "[1, 1, 1, H]"),
        Active("bias", "[1, 1, 1, H]"),
    ),
    output=Active("out", "[B, Y, X, H]"),
    parameter_args=("weight", "bias"),
    forward_semantics="NHWC global response normalization: g=L2norm(x, spatial axes); n=g/(mean_channels(g)+1e-6); out=weight*x*n+bias+x.",
    backward_semantics="Return dx, dweight, dbias. Include the spatial norm and channel-mean derivatives; at a zero norm use the PyTorch zero subgradient.",
    correctness=(Workload(dims=dict(B=2, Y=3, X=5, H=8), dtype="float32"),),
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (0.02, 0.02)},
)
