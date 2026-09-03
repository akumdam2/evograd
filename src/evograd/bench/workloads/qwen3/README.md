# Qwen3-0.6B benchmark workload

## Purpose

One real training workload, specified precisely enough that every number in this
repository traces back to it. The package records what the model actually runs —
which operators, at which shapes, how often — derives the operator tasks from
that record, and can swap a candidate kernel into the live model and check it
before anything is timed. It does not own operator contracts: those live in
`evograd.ops`, reusable across workloads. What is here is the Qwen-specific part
— shapes, frequencies, provenance, adapters, calibration and validation.

## Canonical workload

Qwen3-0.6B, batch 2, sequence 2048, BF16, CUDA, SDPA, `model.train()`,
`use_cache=False`, no gradient checkpointing, fixed seed, no optimizer step:

```python
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

The spec hashes to `qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad`;
every artifact downstream carries that id.

## Level hierarchy

Level says **what is being asked of a kernel** — an axis independent of the
evaluation tier below. A level-2 operator can be measured by a tier-1 pair
benchmark or inside a tier-3 model run; neither choice changes what it is.

| Level | Scope | Question it answers | Package |
|-------|-------|--------------------|---------|
| 4 | The whole training step | Does the reference execution work, and what exactly was it? | `levels/level4` |
| 3 | One captured decoder layer | Replayed offline, do the layer's own tensors reproduce? | `levels/level3` |
| 2 | Four fused Qwen operators | At the shapes the model calls them, within what tolerance? | `levels/level2` |
| 1 | The primitives those four are built from | Which primitive composes into which fused operator? | `levels/level1` |

## Evaluation tiers

Tier says **how carefully an answer is checked**, orthogonal to level.

- **Tier 1** — one operator, one pair of implementations, isolated timing.
- **Tier 2** — one operator as a module, in a small harness.
- **Tier 3** — a drop-in replacement inside the real `Qwen3ForCausalLM`, timed
  only after an untimed correctness gate passes.

Only tier 3 needs Qwen-specific machinery (`evaluation/tier3`). Its gate runs in
a fixed order and names the stage that refused: site preflight → provider purity
→ live-boundary validation → whole-model numerical envelopes → loss trajectory →
invocation counts and patch provenance.

## Package layout

```
qwen3/
├── README.md                       this file
├── __init__.py                     public API: the spec, the report, Qwen3Workload
├── __main__.py                     entry point for `python -m ...qwen3`
├── cli.py                          command router and shared spec overrides
├── harvest/                        where the shapes came from
│   ├── observe.py                  instruments one canonical run, records invocations
│   ├── harvest.py                  runs the instrumented step, writes a manifest
│   ├── manifest.py                 aggregates observations into a manifest
│   ├── snapshot.py                 freezes a manifest into tracked provenance
│   └── snapshot.json               the frozen record; tracked, hashed, versioned
├── levels/
│   ├── level4/
│   │   ├── spec.py                 the canonical WorkloadSpec and its hash
│   │   ├── model.py                model construction; imports transformers lazily
│   │   ├── smoke.py                runs the step and checks gradient coverage
│   │   └── report.py               the smoke report schema
│   ├── level3/
│   │   ├── artifact.py             the .pt layer artifact format and hashing
│   │   ├── capture.py              captures layer 14 from a live model
│   │   └── replay.py               replays it without a GPU model
│   ├── level2/
│   │   ├── qkv_norm_rope.py        derive/verify/calibrate qwen3_qkv_norm_rope
│   │   ├── attention.py            derive/verify/calibrate qwen3_attention
│   │   ├── swiglu_mlp.py           derive/verify/calibrate qwen3_swiglu_mlp
│   │   ├── residual_rmsnorm.py     derive/verify/calibrate fused_add_rms_norm
│   │   ├── calibrate.py            the shared tolerance-calibration inventory
│   │   └── negative_controls.py    shows a calibrated tolerance still rejects errors
│   └── level1/
│       └── mapping.py              primitive → fused operator composition and gates
└── evaluation/
    └── tier3/
        ├── workload.py             Qwen3Workload: build, patch, batch, correctness hook
        ├── sites.py                the four patch sites and their live-module adapters
        ├── validate.py             site preflight at the observed shapes
        ├── purity.py               is the provider a function of its arguments?
        ├── boundary.py             all 140 invocations against their declaration
        ├── numerics.py             tensor statistics, envelopes, the policy schema
        ├── calibrate.py            measures the envelope on this machine
        ├── gate.py                 the ordered untimed gate; timing depends on it
        ├── faults.py               injected defects, kernel-level and optimizer-level
        └── controls.py             runs every fault through the real gate
```

Detailed design — why each site exists, how the tolerances were derived — is in
[`docs/QWEN3_LEVEL4.md`](../../../../../docs/QWEN3_LEVEL4.md), not duplicated
here.

## Reproduction commands

All prefixed with `PYTHONPATH=src python -m evograd.bench.workloads.qwen3`:

```bash
                                    --out results/qwen3-level4/canonical.json
.harvest.harvest                    --out results/qwen3-level4/harvest.json
.harvest.snapshot                   --validate
.levels.level3.capture              --layer 14
.levels.level3.replay               --artifact results/qwen3-level4/layer14.pt
.levels.level2.attention            verify
.levels.level1.mapping              mapping
.evaluation.tier3.validate
.evaluation.tier3.calibrate         run
.evaluation.tier3.controls
```

A measured tier-3 run goes through the installed CLI, which applies the gate
before timing: `evograd tier3-bench --model qwen3_0_6b --structural-identity`.

## Artifact policy

`harvest/snapshot.json` is **tracked provenance**: versioned with the code,
carrying a semantic hash the tests assert
(`e456ae5a9eedae4ffa4ac560097e22b6778581df1ba7de1a0b859cbe7275bb12`), and the
only file declarations are allowed to read.

Everything under `results/` — the `*.json` reports and the `*.pt` layer artifacts
— is a **local run artifact** recording what one machine measured once.
Declarations never read it, and no task identity depends on it.

GPU numerical calibration is **environment-specific**: the tier-3 envelope
measures one GPU, driver and library stack, is stamped with an environment hash,
and is refused rather than reused when that hash does not match.

## Current limitations

- Evolved-kernel evaluation has not started. Every measurement so far uses an
  identity provider — the production spelling, or the bound-pair control — which
  shows the machinery is sound, not that anything is faster.
- The `wrong_update` negative control still passes at 2% on BF16/GH200: an
  update scaled by 1.02 stays inside the calibrated `parameter_update`
  envelope. The same control is rejected on the CPU reference through the
  identical code path, so this is an open measurement question, not a broken
  path. It is not fixed here.
- A CUDA out-of-memory failure in the seed-17 controls run is pending
  verification: captures were moved to host memory to address it, and that
  change has not yet been re-run on a GPU.
- Tier-3 report aggregation and the Pipeline B/D integration are not written.
