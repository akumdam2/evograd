"""The task hierarchy for Qwen3.5-4B.

* level 3 -- the whole text model's training step at B1 x T2048;
* level 2 -- the six fused sites its decoder layers decompose into;
* level 1 -- the primitives the sites are built from.
"""
