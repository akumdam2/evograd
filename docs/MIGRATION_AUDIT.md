# Final OpenEvolve-fork migration audit

This audit compares the last pre-migration fork baseline (`aa75aed`) with the
fork's final `origin/main` (`e7e1e3a`) and records where each durable behavior
lives in evograd. Generated experiment outputs are intentionally distinguished
from reusable source.

## Commit-by-commit disposition

| Fork commit | Change | Evograd disposition |
|---|---|---|
| `35fdb1d` | NCU-guided evolution | Owned integration under `evograd.ncu`: two-process NCU profiling, deterministic roofline triage, diagnosis/generation, strict re-evaluation, and durable pass records. It runs at the supported OpenEvolve API boundary instead of patching OpenEvolve workers. |
| `4a77112` | SwiGLU, GeGLU, Softmax, Cross-Entropy, and KL-Divergence Liger cases | Declaration packages, forward references, exact workloads/contracts, input recipes, tolerances, and reviewed Liger adapters under `evograd.ops`. |
| `93731df` | Merge of the above | Covered by the two rows above. |
| `0d1985d` | One-forward autodiff product pipeline | `evograd.api.evograd`, `evograd run`, callable snapshots, file references, declaration synthesis for unknown ops, three evolution groups, measured shape dispatch, and Markdown/JSON results. |
| `e76143b`, `ad1e565`, `93d52b8` | Interface and benchmark documentation | Consolidated in the root README and this audit. |
| `a54e6e7` | Python callable as forward | Named, self-contained callables are snapshotted to `user_forward.py`; lambdas, closures, and unsupported globals fail at the boundary with actionable errors. |
| `f842029` | Killable evaluation, baseline cache, benchmark smoke pass, scaled tolerances | Generic evaluator subprocess isolation, fd restoration, full benchmark-shape smoke, persistent baseline timing cache, and declaration-level workload/output tolerance hooks. |
| `cfc5048` | Seven additional Liger-suite ops | Dyt, ReLU-squared, Sparsemax, TVD, JSD, PolyNorm, and fused-add RMSNorm declarations and Liger adapters. |
| `e62cd39` | Rename interface to evograd and add registry | `evograd` package/CLI and automatic declaration-package discovery. External generated declarations use an explicit `path.py:op` reference. |
| `e7e1e3a` | NCU agent improvements | Accepted-only NCU rewrite, roofline skip, NCU/evaluator metric separation, configurable timeouts/model/threshold, original/proposed code, `.ncu-rep`, and JSON/Markdown records. |

## Behavioral comparison

| Area | Final fork | Evograd |
|---|---|---|
| Operator contract | `task_spec.py`, forward, per-bench evaluators/configs, and wrappers | One `declare_op` object plus forward; oracle, binding, prompts, wrappers, evaluator, dispatch, and examples derive from it |
| Registered operators | 18 built task specs | The same 18 declarations |
| Liger performance baseline | Per-bench wrapper files | 13 declaration-local reviewed adapters selected by `--baseline auto`; explicit `liger` never silently falls back |
| Correctness oracle | Mechanical templates in task specs | `torch.autograd.grad` over exactly the `Active` inputs |
| Evaluators | Many generated evaluator files | One generic evaluator and five data-driven scoring policies |
| Fault isolation | Killable evaluator child | Same; the 850-second child timeout is below OpenEvolve's 900-second outer timeout |
| Shape safety | Correctness then benchmark smoke | Same for every selected benchmark workload |
| Baseline timing | Persistent JSON cache | Same, keyed by op, baseline, GPU, timing parameters, dtype, and dimensions |
| Generalist/specialists | Full, small, and large runs | Same when a declaration has regime metadata; single-regime declarations collapse to the full winner |
| Deployment | Measured crossing threshold | Same, with a non-tensor route tag in saved state |
| One-call interface | Forward file/callable to deployed pair | Same; known ops reuse their reviewed declaration, unknown ops synthesize and validate a declaration first |
| OpenEvolve integration | Modified in-tree fork | Upstream `openevolve>=0.3.2,<0.4` through `run_evolution` |
| NCU integration | Patched worker internals every N iterations | Supported-boundary final-candidate refinement. This preserves the profiling/diagnosis/acceptance behavior without depending on private worker internals |

The NCU scheduling point is deliberately different. OpenEvolve 0.3.2 has no
public per-candidate/post-evaluation hook, so reproducing the fork's exact
every-N worker mutation would require keeping the fork. Evograd instead refines
each requested evolution group's returned best program and accepts it only
after the same declaration-native evaluator proves correctness and a strictly
higher score. This is the stable transition boundary.

## Source intentionally not copied

- `benchmark/Untitled/checkpoints/**`, evolved programs, reports, and other
  checked-in run products are historical experimental outputs, not framework
  behavior.
- Per-operator evaluator/config/scaffold files are superseded by declarations
  and the generic evaluator.
- The fork's OpenEvolve controller, process-pool, config, and evaluator patches
  are not vendored. Isolation is outside OpenEvolve, and NCU is an evograd
  refinement stage.
- The fork described 14 Liger benchmark ops, but its final RMSNorm directory
  contains no reviewed autograd-pair Liger adapter. Evograd exposes the 13
  baselines actually present rather than labeling PyTorch RMSNorm as Liger.

## Validation boundary

CPU-independent validation covers source compilation, declaration fidelity,
workload partitioning, wrapper generation, scoring, config rendering, dispatch
source generation, and the upstream OpenEvolve 0.3.2 config/API. CUDA execution
must still be run on a GPU using the checklist in the README.
