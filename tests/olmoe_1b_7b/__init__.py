"""OLMoE-1B-7B tests, all on CPU with tiny configurations.

The real workload is the whole model at B2 x T2048 (Level 3) with grouped-GEMM
experts, its layer 8 as the data source, and the four Level-2 sites a layer
decomposes into. Nothing here builds a real-width model; that is the GPU
sessions' job.
"""
