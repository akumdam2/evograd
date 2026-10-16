"""Where the OLMoE shapes come from: one observed whole-model training step.

The canonical Level-3 step (B2 x T2048, grouped-GEMM experts) runs once with
the observer attached; the manifest is frozen into ``snapshot.json``
(provenance ``observed``). The Level-1 observed suite ``olmoe_1b_7b_observed``
is built from it, :func:`.snapshot.check_declared_dims` holds every Level-2
site's declared dims to it, and the whole-model call counts (16 of every site)
are read from it rather than derived.
"""
