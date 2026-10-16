"""The task hierarchy for OLMoE-1B-7B.

* level 3 -- the whole model's training step at B2 x T2048;
* level 2 -- the four fused sites its decoder layer decomposes into;
* level 1 -- the primitives the sites are built from.

Layer 8, captured from the level-3 step or config-derived, is where the
level-2 sites take their real invocations from (:mod:`..layer`); it is not a
level.
"""
