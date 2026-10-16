from evograd.benchmark.cases import ObservedBinding

OBSERVED_BINDINGS = tuple(
    ObservedBinding("qwen2_vl_2b", op, "qwen2_vl_2b_observed")
    for op in ("linear", "linear_no_bias", "layernorm", "rmsnorm", "quick_gelu", "gelu")
)
