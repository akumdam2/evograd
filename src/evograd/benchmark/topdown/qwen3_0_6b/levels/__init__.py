"""The task hierarchy: how much of the workload one measurement covers.

Level is *what is being asked of a kernel*, and it is independent of the
evaluation tier, which is *how carefully the answer is checked*. A level-2
operator can be measured by a tier-1 pair benchmark or by a tier-3 model run;
neither choice changes what the operator is.

* level 3 -- the whole canonical training step;
* level 2 -- the four fused Qwen operators the step decomposes into (three per
  decoder layer, and the LM head with the loss);
* level 1 -- the primitive operations those four are built from.
"""
