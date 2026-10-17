from evograd.benchmark.cases import ObservedBinding

OBSERVED_BINDINGS = tuple(
    ObservedBinding("torchrec_dlrm", op, "torchrec_dlrm_observed")
    for op in ("linear", "embedding_bag")
)
