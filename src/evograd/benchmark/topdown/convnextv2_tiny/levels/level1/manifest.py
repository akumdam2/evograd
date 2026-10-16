from evograd.benchmark.cases import ObservedBinding

OBSERVED_BINDINGS = tuple(
    ObservedBinding("convnextv2_tiny", op, "convnextv2_tiny_observed")
    for op in ("linear", "layernorm", "spatial_conv2d", "grn", "gelu")
)
