from evograd.benchmark.cases import ObservedBinding

OBSERVED_BINDINGS = tuple(
    ObservedBinding("dit_s_2", op, "dit_s_2_observed")
    for op in ("linear", "spatial_conv2d")
)
