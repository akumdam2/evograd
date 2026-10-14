"""Qwen3.5-4B tests, all on CPU with tiny configurations.

The real workload is the whole text model at B1 x T2048 (Level 3), its layer 14
(Gated DeltaNet) and layer 15 (gated full attention) as Level-3 blocks, and the
seven Level-2 sites those blocks decompose into. Nothing here builds a
real-width model; that is the GPU sessions' job.
"""
