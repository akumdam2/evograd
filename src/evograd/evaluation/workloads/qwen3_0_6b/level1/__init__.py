"""Level-1 verification and calibration for the primitives Qwen3-0.6B runs.

The primitive contracts are :mod:`evograd.ops.level1`'s and the model's
selection of them is
:mod:`evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest`'s. What is
here decides whether an implementation of one of them agrees with what the
model actually computed.
"""
