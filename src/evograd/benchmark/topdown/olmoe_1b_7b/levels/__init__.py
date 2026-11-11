"""The task hierarchy for OLMoE-1B-7B.

* level 3 -- the whole model's training step at B2 x T2048;
* level 2 -- the three fused sites its decoder layer decomposes into;
* level 1 -- the primitives the sites are built from.
"""
