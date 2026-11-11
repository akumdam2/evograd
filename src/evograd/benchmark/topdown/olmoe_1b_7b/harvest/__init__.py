"""Where the OLMoE shapes come from: one observed whole-model training step.

The step the real-payload capture runs first (the pinned checkpoint, B2 x
T2048, grouped-GEMM experts, on the first WikiText-2 TRAIN batch of data order
101, built by ``evograd.verification.integrations.base.real_run``) runs once
with the observer attached; the manifest is frozen into ``snapshot.json``
(provenance ``observed``) with that run's identity (``run``). The Level-1 observed suite ``olmoe_1b_7b_observed``
is built from it, :func:`.snapshot.check_declared_dims` holds every Level-2
site's declared dims to it, and the whole-model call counts (16 of every site)
are read from it rather than derived.
"""
