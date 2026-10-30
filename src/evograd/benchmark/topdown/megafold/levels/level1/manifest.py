from evograd.benchmark.cases import ObservedBinding

#: The two MegaFold sites whose precision-uniform generic contract is exactly
#: what MegaFold computes there: the LayerNorm inside PreLayerNorm (float32 in
#: and out, autocast or not) and the SwiGLU inside FusedTransition.
OBSERVED_BINDINGS = tuple(
    ObservedBinding("megafold", op, "megafold_observed")
    for op in ("layernorm", "swiglu")
)
