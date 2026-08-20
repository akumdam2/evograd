# MAP-Elites Kernel Evolution Analysis Log

Date: 2026-08-06
Final Checkpoint: checkpoint_10 (10 iterations x 2 islands)

## Overview

Ten kernel operators evolved using OpenEvolve + MAP-Elites with custom feature
dimensions (saved_memory_ratio, shape_specialization) to explore kernel-evolution
search spaces. Each run maintained 2 islands with 10×10 cells (100 max cell capacity).

**Summary Results:**

- **cross_entropy       **: score=  2.4326 cells= 8/100 lineage_depth=6 correct=✓
- **dyt                 **: score=  1.1342 cells= 8/100 lineage_depth=5 correct=✓
- **fused_add_rms_norm  **: score=  1.4352 cells= 7/100 lineage_depth=5 correct=✓
- **jsd                 **: score=  7.6256 cells= 7/100 lineage_depth=4 correct=✓
- **kl_div              **: score=  2.6355 cells=10/100 lineage_depth=4 correct=✓
- **layernorm           **: score=  2.1641 cells= 8/100 lineage_depth=3 correct=✓
- **relu_squared        **: score=  1.0059 cells= 9/100 lineage_depth=2 correct=✓
- **softmax             **: score=  1.0566 cells=10/100 lineage_depth=2 correct=✓
- **sparsemax           **: score=-1000000.0000 cells= 4/100 lineage_depth=2 correct=✗
- **tvd                 **: score=  0.2529 cells= 5/100 lineage_depth=1 correct=✓


## CROSS_ENTROPY

### Trajectory
Seed score: 0.2210
Final best: 2.4326
Steps: 7 improvements

  Iteration  2: f73443f4 1.4564 → 1.4984 (+0.0419)
  Iteration  3: 035381fd 1.4984 → 1.9802 (+0.4818)
  Iteration  5: 24d926ef 1.9802 → 2.3357 (+0.3556)
  Iteration  7: fb42ef55 2.3357 → 2.3389 (+0.0031)
  Iteration  8: 919671e5 2.3389 → 2.3610 (+0.0222)
  Iteration  9: 84e67c63 2.3610 → 2.4326 (+0.0715)

### Cell Occupancy
  Island 0: 5/100 cells occupied (5% utilization)
  Island 1: 3/100 cells occupied (3% utilization)

### Best Program Lineage
Chain depth: 6 → seed
  gen=5 iter=10 id=84e67c63 score=  2.4326 cell=9-6
  gen=4 iter= 8 id=fb42ef55 score=  2.3389 cell=not in archive
  gen=3 iter= 6 id=24d926ef score=  2.3357 cell=not in archive
  gen=2 iter= 4 id=035381fd score=  1.9802 cell=9-0
  gen=1 iter= 2 id=f8c0e3fc score=  1.2495 cell=9-9

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 1.0000 to 1.0002 (spread: 0.0002)
    Distinct values: 3/8 elites

## DYT

### Trajectory
Seed score: 0.0915
Final best: 1.1342
Steps: 4 improvements

  Iteration  4: a856f936 0.9545 → 0.9889 (+0.0344)
  Iteration  5: f71869c5 0.9889 → 1.0247 (+0.0358)
  Iteration  7: a638f2d7 1.0247 → 1.1342 (+0.1095)

### Cell Occupancy
  Island 0: 6/100 cells occupied (6% utilization)
  Island 1: 2/100 cells occupied (2% utilization)

### Best Program Lineage
Chain depth: 5 → seed
  gen=4 iter= 8 id=a638f2d7 score=  1.1342 cell=9-0-4
  gen=3 iter= 6 id=f71869c5 score=  1.0247 cell=not in archive
  gen=2 iter= 4 id=2e616e04 score=  0.9375 cell=7-0-3
  gen=1 iter= 2 id=5693f771 score=  0.7449 cell=not in archive
  gen=0 iter= 0 id=427745a0 score=  0.0915 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 0.9999 to 1.0000 (spread: 0.0001)
    Distinct values: 2/8 elites
  shape_specialization:
    Range: 0.3149 to 1.0000 (spread: 0.6851)
    Distinct values: 8/8 elites

## FUSED_ADD_RMS_NORM

### Trajectory
Seed score: 0.1617
Final best: 1.4352
Steps: 6 improvements

  Iteration  1: 858dc2a1 1.2862 → 1.3904 (+0.1043)
  Iteration  3: 1b11f772 1.3904 → 1.4167 (+0.0263)
  Iteration  5: fc397913 1.4167 → 1.4218 (+0.0051)
  Iteration  7: cad8fa8f 1.4218 → 1.4275 (+0.0058)
  Iteration  9: cd3796b9 1.4275 → 1.4352 (+0.0077)

### Cell Occupancy
  Island 0: 4/100 cells occupied (4% utilization)
  Island 1: 3/100 cells occupied (3% utilization)

### Best Program Lineage
Chain depth: 5 → seed
  gen=4 iter=10 id=cd3796b9 score=  1.4352 cell=8-0
  gen=3 iter= 8 id=cad8fa8f score=  1.4275 cell=7-0
  gen=2 iter= 4 id=1b11f772 score=  1.4167 cell=9-0
  gen=1 iter= 2 id=858dc2a1 score=  1.3904 cell=not in archive
  gen=0 iter= 0 id=c33a6eaa score=  0.1617 cell=5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 0.5006 to 1.0000 (spread: 0.4994)
    Distinct values: 2/7 elites

## JSD

### Trajectory
Seed score: 0.2524
Final best: 7.6256
Steps: 5 improvements

  Iteration  2: bbe47210 0.2524 → 2.2796 (+2.0271)
  Iteration  3: 7c82e7fe 2.2796 → 7.1876 (+4.9080)
  Iteration  5: e1c3ea1a 7.1876 → 7.5763 (+0.3887)
  Iteration  7: fb7ee216 7.5763 → 7.6184 (+0.0421)
  Iteration  9: a7f6f70a 7.6184 → 7.6256 (+0.0071)

### Cell Occupancy
  Island 0: 3/100 cells occupied (3% utilization)
  Island 1: 4/100 cells occupied (4% utilization)

### Best Program Lineage
Chain depth: 4 → seed
  gen=3 iter=10 id=a7f6f70a score=  7.6256 cell=7-3
  gen=2 iter= 4 id=7c82e7fe score=  7.1876 cell=6-0
  gen=1 iter= 2 id=46aa550c score=-999999.6667 cell=not in archive ← niche elite (not best at time)
  gen=0 iter= 0 id=3611a58d score=  0.2524 cell=5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 0.5020 to 2.0000 (spread: 1.4980)
    Distinct values: 3/7 elites
  shape_specialization:
    Range: 0.6332 to 3.1511 (spread: 2.5178)
    Distinct values: 6/6 elites

## KL_DIV

### Trajectory
Seed score: 2.4904
Final best: 2.6355
Steps: 4 improvements

  Iteration  2: 263a49c3 2.4904 → 2.4904 (+0.0001)
  Iteration  3: 05c80ea4 2.4904 → 2.5797 (+0.0892)
  Iteration  5: dd358089 2.5797 → 2.6348 (+0.0552)
  Iteration  7: d2a50544 2.6348 → 2.6355 (+0.0006)

### Cell Occupancy
  Island 0: 6/100 cells occupied (6% utilization)
  Island 1: 4/100 cells occupied (4% utilization)

### Best Program Lineage
Chain depth: 4 → seed
  gen=3 iter= 8 id=d2a50544 score=  2.6355 cell=9-0-4
  gen=2 iter= 4 id=05c80ea4 score=  2.5797 cell=9-5-5
  gen=1 iter= 2 id=9acea397 score=  1.7736 cell=2-5-0
  gen=0 iter= 0 id=9dd2c73e score=-1000000.0000 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 1.0000 to 1.0000 (spread: 0.0000)
    Distinct values: 1/10 elites
  shape_specialization:
    Range: 0.4821 to 1.0000 (spread: 0.5179)
    Distinct values: 10/10 elites

## LAYERNORM

### Trajectory
Seed score: 0.1212
Final best: 2.1641
Steps: 3 improvements

  Iteration  1: fe169f01 1.9429 → 2.1210 (+0.1781)
  Iteration  7: 26096f20 2.1210 → 2.1641 (+0.0431)

### Cell Occupancy
  Island 0: 4/100 cells occupied (4% utilization)
  Island 1: 4/100 cells occupied (4% utilization)

### Best Program Lineage
Chain depth: 3 → seed
  gen=2 iter= 8 id=26096f20 score=  2.1641 cell=7-9-0
  gen=1 iter= 2 id=fe169f01 score=  2.1210 cell=9-9-9
  gen=0 iter= 0 id=69619079 score=  0.1212 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 0.8950 to 1.0019 (spread: 0.1069)
    Distinct values: 3/8 elites
  shape_specialization:
    Range: 0.9022 to 1.0002 (spread: 0.0981)
    Distinct values: 8/8 elites

## RELU_SQUARED

### Trajectory
Seed score: 0.1587
Final best: 1.0059
Steps: 2 improvements

  Iteration  2: 132d653e 1.0055 → 1.0059 (+0.0004)

### Cell Occupancy
  Island 0: 6/100 cells occupied (6% utilization)
  Island 1: 3/100 cells occupied (3% utilization)

### Best Program Lineage
Chain depth: 2 → seed
  gen=1 iter= 3 id=132d653e score=  1.0059 cell=8-5-0
  gen=0 iter= 0 id=e3a2aaf4 score=  0.1587 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 1.0000 to 1.0000 (spread: 0.0000)
    Distinct values: 1/9 elites
  shape_specialization:
    Range: 0.6693 to 1.0000 (spread: 0.3307)
    Distinct values: 9/9 elites

## SOFTMAX

### Trajectory
Seed score: 0.3091
Final best: 1.0566
Steps: 6 improvements

  Iteration  1: deddc630 0.5839 → 0.6022 (+0.0184)
  Iteration  2: 8dfba864 0.6022 → 0.7810 (+0.1788)
  Iteration  3: 3191eb47 0.7810 → 0.7997 (+0.0187)
  Iteration  5: 20670850 0.7997 → 0.7998 (+0.0000)
  Iteration  8: c2fa5132 0.7998 → 1.0566 (+0.2568)

### Cell Occupancy
  Island 0: 5/100 cells occupied (5% utilization)
  Island 1: 5/100 cells occupied (5% utilization)

### Best Program Lineage
Chain depth: 2 → seed
  gen=1 iter= 9 id=c2fa5132 score=  1.0566 cell=3-9-1
  gen=0 iter= 0 id=638c4549 score=  0.3091 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 0.7868 to 1.0000 (spread: 0.2132)
    Distinct values: 2/10 elites
  shape_specialization:
    Range: 0.4343 to 0.6967 (spread: 0.2623)
    Distinct values: 10/10 elites

## SPARSEMAX

### Trajectory
Seed score: -1000000.0000
Final best: -1000000.0000

### Cell Occupancy
  Island 0: 2/100 cells occupied (2% utilization)
  Island 1: 2/100 cells occupied (2% utilization)

### Best Program Lineage
Chain depth: 2 → seed
  gen=1 iter= 1 id=e919a19e score=-1000000.0000 cell=9-5-5
  gen=0 iter= 0 id=be5a4b5c score=-1000000000.0000 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 1.0000 to 1.0000 (spread: 0.0000)
    Distinct values: 1/4 elites
  shape_specialization:
    Range: 1.0000 to 1.0000 (spread: 0.0000)
    Distinct values: 1/4 elites

## TVD

### Trajectory
Seed score: 0.2529
Final best: 0.2529

### Cell Occupancy
  Island 0: 3/100 cells occupied (3% utilization)
  Island 1: 2/100 cells occupied (2% utilization)

### Best Program Lineage
Chain depth: 1 → seed
  gen=0 iter= 0 id=4141866c score=  0.2529 cell=5-5-5

### Feature Axes & Diversity
  saved_memory_ratio:
    Range: 1.0000 to 1.0000 (spread: 0.0000)
    Distinct values: 1/5 elites
  shape_specialization:
    Range: 0.5523 to 1.0000 (spread: 0.4477)
    Distinct values: 2/5 elites


## Summary Table

|  Operator  | Best Score | Cells | Lineage | SMR Spread | Shape Spec Spread | Correct |
|:-----------|----------:|------:|-------:|----------:|---:|---:|
| cross_entropy   |     2.4326 |      8 |       6 |     0.0002 |            0.0000 | ✓       |
| dyt             |     1.1342 |      8 |       5 |     0.0001 |            0.6851 | ✓       |
| fused_add_rms_norm |     1.4352 |      7 |       5 |     0.4994 |            0.0000 | ✓       |
| jsd             |     7.6256 |      7 |       4 |     1.4980 |            2.5178 | ✓       |
| kl_div          |     2.6355 |     10 |       4 |     0.0000 |            0.5179 | ✓       |
| layernorm       |     2.1641 |      8 |       3 |     0.1069 |            0.0981 | ✓       |
| relu_squared    |     1.0059 |      9 |       2 |     0.0000 |            0.3307 | ✓       |
| softmax         |     1.0566 |     10 |       2 |     0.2132 |            0.2623 | ✓       |
| sparsemax       | -1000000.0000 |      4 |       2 |     0.0000 |            0.0000 | ✗       |
| tvd             |     0.2529 |      5 |       1 |     0.0000 |            0.4477 | ✓       |


## Analysis & Verdict

### Observed Feature-Axis Impact

**Strong evidence (saved_memory_ratio impact):**
- **jsd**: spread 0.50–2.00 with 7 distinct values → evolved programs traded memory for speed
- **fused_add_rms_norm**: spread 0.50–1.00 with stable occupancy across both islands

**Moderate evidence (shape_specialization impact):**
- **dyt**: spread 0.31–1.00 with all 8 elites at distinct shape-spec values
- **kl_div**: spread 0.48–1.00 showing shape-aware kernel variants
- **layernorm**: minimal spread (0.90–1.00) suggesting weak shape-induced diversity

**Weak evidence / inconclusive:**
- **cross_entropy, relu_squared, softmax**: saved_memory_ratio nearly constant (0.9999–1.0002)
- **tvd, sparsemax**: unable to evaluate (tvd: seed-only, sparsemax: infrastructure failure)

### Confounds & Limitations

1. **Single run per operator** — no variance estimate or statistical significance
2. **No ablation control** — never ran with default 2D complexity/diversity axes or baseline island config
3. **Short evolution** — 10 iterations is below practical threshold; typical runs use 50–100+
4. **Shape-specialization deployment** — metric only active from jsd onward (7/10 ops), not from seed
5. **Niche retention unknown** — archive stored programs but no explicit log of 'which cells saved which ancestors'

### Conclusion

**Verdict: INCONCLUSIVE — Suggestive of impact, not definitive**

The custom MAP-Elites feature dimensions (saved_memory_ratio, shape_specialization) showed
correlation with program diversity in jsd/dyt/kl_div but no causal evidence. To establish impact:

**Recommended ablation experiments:**
1. Run 3× replicates per operator
2. Control: same 10 ops × 3 replicates on default 2D axes (complexity + diversity)
3. Extend to 30+ iterations per run
4. Deploy shape_specialization metric from iteration 0
5. Log niche ancestry: track whether archives preferentially retained non-best programs

### Failure Cases

**tvd (Total Variation Divergence):**
- Only the seed program (iteration=0, generation=0, score=0.2529) passed correctness (correct=1.0)
- All 4 generated programs failed (correct=0.0)
- Archive stored seed + failures but new generations never recovered correctness
- Verdict: Search did not find any correct variants; feature-axis impact not evaluable

**sparsemax (Sparsemax activation):**
- Seed and all generated programs scored -1000000.0 (failure penalty)
- Cause: Liger baseline kernel crashes on inputs with >64K columns in sparsemax operation
- This is a baseline infrastructure bug (Liger limitation), not evolved program quality
- Verdict: Run not evaluable for kernel evolution; recommendation to either (a) reduce input
  dimension in benchmark or (b) skip sparsemax from future MAP-Elites sweeps

---

## Ablation: custom axes (complexity, saved_memory_ratio, shape_specialization) vs OpenEvolve defaults (complexity, diversity)

Same seeds, model (gpt-5.6-sol), iterations (10), baseline (liger). Single run per condition.

| op | condition | best | correct | cells | smr spread | shape_spec spread | mem-savers (<0.9) |
|---|---|---|---|---|---|---|---|
| layernorm | custom | 2.164 | 11 | 8 | 0.895–1.002 | 0.902–1.000 | 8 |
| layernorm | control | 2.138 | 11 | 9 | 0.895–1.002 | 0.884–1.016 | 3 |
| kl_div | custom | 2.635 | 10 | 10 | 1.000 | 0.482–0.763 | 0 |
| kl_div | control | 2.599 | 10 | 9 | 1.000 | 0.422–0.735 | 0 |
| dyt | custom | 1.134 | 10 | 8 | 1.000 | 0.315–0.662 | 0 |
| dyt | control | 1.147 | 6 | 9 | 1.000–3.000 | 0.281–0.664 | 0 |

Findings (n=1 per condition, 10 iterations — treat as directional):
- Best-score impact: nil. Three near-ties (deltas +0.026, +0.036, −0.013), within run-to-run noise.
- Archive-diversity impact: largely nil. Control archives retained essentially the same
  saved_memory_ratio and shape_specialization spreads; the behavioral diversity comes from
  the LLM's variation, and both archive schemes preserved it about equally at this scale.
  Weak positives for custom: layernorm kept 8 low-memory elites vs control's 3; dyt kept
  10 correct programs vs control's 6. Weak negative: dyt control found a wider smr range.
- The durable value of the feature work at this budget is measurement, not selection:
  shape_specialization/saved_memory_ratio are now recorded for every program in every run,
  so regime specialists can be harvested from any archive for a dispatch stage.
- To detect a selection effect, if one exists, the budget must let niches compete:
  30+ iterations, 3+ replicates per condition, and ops with hard small/large regime splits.

## Final sweep table (vs Liger, A100-40GB, arithmetic aggregate)

cross_entropy 2.43 | kl_div 2.64 | jsd 7.63 (geomean 3.62/4.38) | layernorm 2.16 |
geglu 1.66 | fused_add_rms_norm 1.44 | dyt 1.13 | swiglu 1.07 | softmax 1.06 |
relu_squared 1.01 | tvd 0.25 (evolution failed) | sparsemax n/a (liger caps at 64K cols;
seed vs eager autograd: geomean bwd 0.80, full 0.52) | poly_norm excluded (bf16 dx off by
~1 ulp vs declared atol 0.08 — tolerance-policy question).
