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

## The four-part correctness protocol (schema `evograd-qwen3-t3-protocol/4`)

`evaluation/tier3/protocol4.py`, `prediction.py`, `training.py`, `textdata.py`
and the driver `protocol4_cli.py`. Four questions, each with its own metric,
floor and threshold, calibrated on a **pretrained checkpoint and real text**
(`Qwen/Qwen3-0.6B@c1899de2` on WikiText-2 raw `@b08601e0`), which has its own
workload identity -- no synthetic calibration can bind to it.

| part | metric | enforced? | where |
|------|--------|-----------|-------|
| A local computation | per-result atol/rtol on every live output, input gradient, weight gradient; structure; missing gradients; finiteness; purity; per-site invocation coverage | **enforced**, unchanged | `boundary.py`, `purity.py` |
| B model prediction | mean valid-token **KL(P_eager ‖ P_provider)**, float32 log-softmax over the full vocabulary, temperature 1, float64 accumulation, causal shift, `ignore_index=-100`, chunked; raw-logit rel-L2 kept as a diagnostic | **enforced** | `prediction.mean_token_kl` |
| C model gradients | global parameter-gradient vector relative L2, matched by name, before the first optimizer update; missing/extra/shape/non-finite fail | **enforced** | `simple.global_grad_rel_l2` |
| D training behaviour | max abs Δ of 50-step token-weighted training NLL windows; max abs Δ of validation NLL at steps 0/100/500/1000 through one trusted eager evaluator on disjoint held-out text | **enforced** | `training.py` |

Parameter updates, Adam moments, per-role errors, raw-logit error and perplexity
remain **diagnostics**.

```
T_m = 2 * max( max compile-vs-eager distance over calibration seeds {0,1,2},
               max repeated-eager distance over the same seeds,
               FLOOR_m )
FLOORS: KL 1e-6 nats · grad rel-L2 1e-4 · train-window NLL 1e-4 · validation NLL 1e-4
```

Floors are independent and justified in `protocol4.py`; the KL floor is *not*
inherited from any logits floor. The policy binds to workload id and hash, patch
set, environment, data identity (dataset revision, tokenizer, packing, masks),
metric definitions and the training plan, and refuses anything else. Older
schemas are refused rather than reinterpreted. Holdout seeds {11, 17} are judged
against the frozen file -- trusted compile as well as the candidate -- and a
candidate cannot reach the derivation.

In the timing runner, `--real-text --protocol4-calibration P --protocol4-verdict V`
makes this the enforced gate: A, B, C run in-process and D is read from the
holdout verdict for the provider being timed; a provider without a verdict is
not timed.

**Provenance.** The local elementwise layer follows Liger-Kernel's and
FlashAttention's correctness-test practice; comparing trained models rather than
operators follows Cut Cross-Entropy and FlashMask. The KL anchor, the
compile-vs-eager calibration and the automated trajectory thresholds are this
project's design choices, not published standards.

**Known limits.** Real text is WikiText-2 raw, ~2.5M train tokens: a 1,000-step
run at batch 2 x 2048 cycles the split 1.6 times in a fixed order. Five-step or
thousand-step agreement is a statement about that horizon only.

## The simplified numerical policy (shadow mode, NOT promoted)

`evaluation/tier3/simple.py` implements a second, much smaller whole-model
correctness model, calibrated per `(workload, dtype, patch set)` against a
trusted replacement that patches **exactly the sites the candidate patches**:

```
threshold[m] = max( max reference noise[m],        # eager vs rebuilt eager
                    max matched trusted drift[m],  # eager vs R(P)
                    FLOORS[m] ) * 2.0
```

Hard at model level: presence of every output and gradient, finiteness, logits
relative L2, and one streamed global parameter-gradient relative L2. Everything
the detailed policy measured is still collected as a **diagnostic**, including
the five-step loss trajectory — a five-step delta is not evidence about
long-horizon training quality and is no longer treated as if it were.

**It is shadow-only by default and was not promoted.** `check_model_correctness`
takes `simple_policy` and `simple_primary`; with `simple_primary=False`, the
default, the detailed policy still decides and the simplified verdict is only
recorded beside it.

The blocker is specific. For `qkv_norm_rope`, `attention` and `swiglu_mlp` the
matched trusted replacement recomputes through the *same* `runtime_forward` the
production spelling calls, so its drift is identically zero and the threshold
collapses onto the metric floor (2.0e-05 for logits). **`torch.compile` of that
same reference function — a correct, independent implementation that passes
tier 1, tier 2 and the full live-boundary gate — measures 1.0971e-02 and is
rejected at 548x.** A trusted provider must pass, so the policy stays in shadow.
No margin and no floor was adjusted in response.

`residual_rmsnorm` is the counter-example showing the design is sound where the
trusted reference is non-degenerate: drift 3.4285e-03, threshold 6.8569e-03,
Liger passes at 2.6259e-03, the evolved candidate fails at 1.35x, and all seven
negative controls are rejected.

Measured separately, and not interchangeable: the calibrated smoke and canonical
thresholds differ by between 1.00x and 38.57x depending on patch set and metric,
so "smoke must pass before canonical" is a cost-saving prefilter rather than a
sound implication.

## Current limitations

- Evolved-kernel evaluation has not started. Every measurement so far uses an
  identity provider — the production spelling, or the bound-pair control — which
  shows the machinery is sound, not that anything is faster.
- The simplified numerical policy above is **not promoted**; it runs in shadow
  only, and its `qkv_norm_rope` threshold is known to reject `torch.compile`.
- A 2% multiplicative update fault is **not detectable at bfloat16** and is no
  longer required to be. One AdamW step at lr=1e-4 moves a projection weight by
  1.22e-4, exactly two bfloat16 ULPs, so scaling it by 1.02 changes 3.9% of
  stored values and leaves the rest bit-identical; on a norm weight, where the
  ULP is 7.8e-3, the whole update rounds away and nothing changes at all. The
  residual deviation is 5-11x below the measured hardware noise for that
  quantity, and below it even with the integration term removed. The required
  control is now a two-ULP perturbation of the stored parameter, which changes
  every element by construction; `wrong_update` is kept as a diagnostic and
  every control reports how much of its fault reached stored state, so "not
  detected" can never be read as "no fault occurred".
- A CUDA out-of-memory failure in the seed-17 controls run is pending
  verification: captures were moved to host memory to address it, and that
  change has not yet been re-run on a GPU.
- Tier-3 report aggregation and the Pipeline B/D integration are not written.
