"""One OLMoE decoder layer -- a data source for the Level-2 sites, not a level.

Layer 8, with the boundary ``hidden_out = layer(hidden_states, position_embeddings)``
and cotangent ``g``. Two sources: **captured** from the whole-model step
(:mod:`.capture`, :mod:`.artifact`) or **config-derived** (:mod:`.prepare`, the
model's own initialization rules applied explicitly).
"""
