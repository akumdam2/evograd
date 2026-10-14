"""The task hierarchy for Qwen3.5-4B.

* level 3 -- the whole text model's training step at B1 x T2048;
* level 2 -- the seven fused sites its decoder layers decompose into;
* level 1 -- the primitives the sites are built from.

Layers 14 (Gated DeltaNet) and 15 (gated full attention), captured from the
level-3 step or config-derived, are where the level-2 sites take their real
invocations from (:mod:`..layer`); they are not a level.
"""
