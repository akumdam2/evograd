"""Operator declaration: dense NCHW conv2d."""

from evograd.opdecl import Active, Provenance, Workload, declare_op

# ResNet-style stage resolutions and channel widths (56x56x64, 28x28x256,
# 14x14x512), but NOT literal ResNet layers: the declared contract is
# stride 1 / padding 0 / dilation 1 / groups 1, whereas ResNet's 3x3
# convolutions use padding 1. Recorded as handpicked rather than derived,
# because inventing a model config to justify these numbers would be exactly
# the pretence provenance exists to prevent.
_PROVENANCE = Provenance(
    model="resnet_style",
    component="conv_stage",
    source="handpicked",
    note=(
        "ResNet-50 stage resolutions and channel widths; the stride-1/padding-0 "
        "contract means these are not literal ResNet layers"
    ),
)


_DIMS = ("B", "C", "H", "W", "O", "KH", "KW", "OH", "OW")
_BENCHMARK_CASES = (
    # B, C, H, W, O, KH, KW, OH, OW (stride=1, padding=0)
    (32, 64, 56, 56, 64, 3, 3, 54, 54),
    (32, 64, 56, 56, 128, 3, 3, 54, 54),
    (16, 256, 28, 28, 256, 3, 3, 26, 26),
    (16, 256, 28, 28, 512, 1, 1, 28, 28),
    (8, 512, 14, 14, 512, 3, 3, 12, 12),
    (8, 512, 14, 14, 1024, 1, 1, 14, 14),
)


def _workload(values, dtype):
    return Workload(
        dims=dict(zip(_DIMS, values)), dtype=dtype, provenance=_PROVENANCE
    )


def _benchmark_workloads():
    return tuple(_workload(values, "bfloat16") for values in _BENCHMARK_CASES)


def make_conv2d_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    expected_h = dims["H"] - dims["KH"] + 1
    expected_w = dims["W"] - dims["KW"] + 1
    if (dims["OH"], dims["OW"]) != (expected_h, expected_w):
        raise ValueError("declared OH/OW do not match conv2d output formula")
    dtype = getattr(torch, workload.dtype)
    seed = sum((index + 1) * dims[name] for index, name in enumerate(_DIMS))
    torch.manual_seed(seed)
    x = torch.randn(
        (dims["B"], dims["C"], dims["H"], dims["W"]),
        device=device,
        dtype=dtype,
    )
    weight = (
        torch.randn(
            (dims["O"], dims["C"], dims["KH"], dims["KW"]),
            device=device,
            dtype=dtype,
        )
        * ((dims["C"] * dims["KH"] * dims["KW"]) ** -0.5)
    ).to(dtype)
    bias = torch.randn((dims["O"],), device=device, dtype=dtype)
    dy = torch.randn(
        (dims["B"], dims["O"], dims["OH"], dims["OW"]),
        device=device,
        dtype=dtype,
    )
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "dy": dy,
    }


op = declare_op(
    name="conv2d",
    level=1,
    family="conv",
    forward="evograd.ops.level1.conv2d.forward_ref:conv2d_forward_ref",
    dims=_DIMS,
    args=(
        Active("x", "[B, C, H, W]"),
        Active("weight", "[O, C, KH, KW]"),
        Active("bias", "[O]"),
    ),
    output=Active("y", "[B, O, OH, OW]"),
    forward_semantics=(
        "Compute a dense groups=1, dilation=1 NCHW convolution with OIHW "
        "weights, fixed stride=1 and padding=0, followed by a "
        "per-output-channel bias. Generated math must implement convolution "
        "with Triton (for example implicit GEMM) and must not call "
        "torch.nn.functional.conv2d or PyTorch autograd."
    ),
    backward_semantics=(
        "Return dx, dweight, and dbias with the corresponding input dtypes. "
        "Use stride=1 and padding=0. Accumulate channel/spatial "
        "reductions in float32 before casting outputs."
    ),
    extra_constraints=(
        "The default performance baseline is PyTorch autograd backed by "
        "cuDNN; Liger does not provide a generic convolution kernel. Pipeline "
        "B lowers this fixed contract to handwritten implicit-GEMM Triton "
        "forward, dX, dWeight, and dBias primitives."
    ),
    correctness=(
        _workload((2, 3, 8, 8, 4, 3, 3, 6, 6), "float32"),
        _workload((1, 4, 9, 7, 6, 3, 3, 7, 5), "float32"),
        _workload((2, 8, 16, 16, 8, 3, 3, 14, 14), "float16"),
        _workload((2, 8, 15, 15, 16, 3, 3, 13, 13), "float16"),
        _workload((2, 8, 16, 16, 8, 3, 3, 14, 14), "bfloat16"),
        _workload((2, 8, 15, 15, 16, 3, 3, 13, 13), "bfloat16"),
    ),
    coverage=_benchmark_workloads(),
    benchmark=_benchmark_workloads(),
    benchmark_suites={"cnn_bf16": _benchmark_workloads()},
    tolerances={
        "float32": (5e-4, 5e-4),
        "float16": (1e-1, 5e-2),
        "bfloat16": (2e-1, 8e-2),
    },
    tolerance_multipliers={
        "dweight": (2.0, 1.0),
        "dbias": (2.0, 1.0),
    },
    make_inputs=make_conv2d_inputs,
)
