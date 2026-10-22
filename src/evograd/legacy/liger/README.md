# Liger-Kernel adapters (retired)

Liger-Kernel is not a baseline of the benchmark any more. These are the former
reviewed Liger autograd pairs, moved here unchanged on 2026-10-21:

| file | was |
| --- | --- |
| `ops/<op>.py` | `ops/level1/<op>/liger.py` (13 primitives) |
| `tasks/fused_linear_cross_entropy.py` | `benchmark/operator_suite/tasks/level2/fused_linear_cross_entropy/liger.py` |
| `tasks/fused_moe_swiglu.py` | `benchmark/operator_suite/tasks/level2/fused_moe_swiglu/liger.py` |
| `tasks/qwen3_residual_rmsnorm.py` | `benchmark/topdown/qwen3_0_6b/levels/level2/residual_rmsnorm/liger.py` |
| `library_pairs.py` | the Liger half of `ops/library_pairs.py` |
| `qwen3_5_library.py` | the Liger half of `benchmark/topdown/qwen3_5_4b/levels/level2/library.py` |
| `scripts/run_layernorm_map_vs_explicit.py` | `scripts/run_layernorm_map_vs_explicit.py` (timed against Liger) |

The declarations' `performance_baselines={"liger": ...}` entries, the short
`_liger_factory` wrappers beside them and layernorm's inline adapter were
deleted, not moved; they are in commit e176a05.

No declaration registers these files, no CLI offers them, and they are not
tested. The candidate source check still forbids calling `liger_kernel`.
