"""The block scope of ``tier3-bench``: ``--scope block``.

Everything the model scope's CLI already does -- one killable child per
provider, seeded random order, the provider conventions (``--candidate``,
``--patch-set``, ``--compile-site``, ``--structural-identity``,
``--identity-control``, ``--baseline``) -- applies unchanged. What is added is
the case (``--block-source captured --artifact PATH`` or ``--block-source
config``), the timing loop's own knobs, and the policy: calibrated from
trusted controls in a child of its own before any provider is judged, frozen
to a file, and handed to every provider child by path.

    python -m evograd.evaluation.tier3.cli --scope block --model qwen3_0_6b \\
        --block-source captured --artifact results/qwen3-level4/layer14.pt \\
        --structural-identity --identity-control --compile-site swiglu_mlp \\
        --patch-set best3:swiglu_mlp=compile,residual_rmsnorm=compile \\
        --out results/evaluation/tier3/qwen3_0_6b/block/layer14.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from evograd.evaluation.tier3 import block as _block
from evograd.evaluation.tier3.gate import block as _gate

#: Provider name the parent uses to ask a child for the calibrated policy.
POLICY_PROVIDER = "__policy__"

#: Origins that mark a provider as a trusted control rather than a candidate.
CONTROL_ORIGINS = {"structural_identity", "identity_control", "trusted_torch_compile"}

#: Model-scope options that mean nothing for a block. Checked against the
#: parser's defaults so an explicitly given one is refused by name.
MODEL_ONLY = ("loss_steps", "learning_rate", "layers", "data_seed", "calibration", "residues",
              "simple_calibration", "real_text", "protocol4_calibration", "protocol4_verdict",
              "protocol4_diagnostic_timing", "whole_model_compile")


def add_block_arguments(parser) -> None:
    group = parser.add_argument_group("block scope (--scope block)")
    group.add_argument("--scope", choices=("model", "block"), default="model",
                       help="model: the whole training step (default). block: one "
                            "architectural block, forward + supplied-cotangent VJP, for "
                            "workloads that declare a block adapter")
    group.add_argument("--block-source", choices=("captured", "config"), default="config",
                       help="captured: replay a verified layer artifact (--artifact); config: "
                            "seeded weights/inputs/cotangents from the pinned architecture")
    group.add_argument("--artifact", type=Path, default=None,
                       help="the captured layer artifact (block scope, captured source)")
    group.add_argument("--layer-index", type=int, default=None,
                       help="which block instance; not --layers, which is a model-depth override")
    group.add_argument("--block-seed", type=int, default=0,
                       help="seed of the config-derived recipe (weights, inputs, cotangents)")
    group.add_argument("--no-canonical-check", action="store_true",
                       help="captured source: verify both artifact hashes but do not require the "
                            "artifact to be the tracked canonical capture (debug captures)")
    group.add_argument("--samples", type=int, default=10,
                       help="independent timed repetitions per block (reset before each)")
    group.add_argument("--noise-repeats", type=int, default=3,
                       help="extra native repetitions for the repeatability floor")
    group.add_argument("--block-policy", type=Path, default=None,
                       help="a frozen block policy to judge providers by; default: calibrate "
                            "one in this run from the trusted controls")
    group.add_argument("--freeze-policy", type=Path, default=None,
                       help="where to write the policy calibrated in this run")
    group.add_argument("--calibrate-with", default=",".join(_gate.DEFAULT_CONTROLS),
                       help="trusted controls the policy is derived from, comma separated: "
                            "structural_identity, bound_pair_identity, trusted_torch_compile")
    group.add_argument("--no-purity", action="store_true",
                       help="skip the provider purity gate (production-width repeated calls)")
    group.add_argument("--arch-override", action="append", default=[], metavar="KEY=INT",
                       help="config source only: override one architecture field (a reduced "
                            "block for a smoke; the case records it and is not the canonical one)")


def check_block_options(args, parser, adapter) -> None:
    if adapter.block is None:
        parser.error(f"--scope block: {args.model} declares no block adapter; workloads with one "
                     "are those whose Tier3Adapter sets `block`")
    for dest in MODEL_ONLY:
        if getattr(args, dest, None) != parser.get_default(dest):
            parser.error(f"--{dest.replace('_', '-')} is a model-scope option and means nothing "
                         "for --scope block")
    if args.block_source == "captured" and args.artifact is None:
        parser.error("--block-source captured needs --artifact PATH")
    if args.block_source == "captured" and args.arch_override:
        parser.error("--arch-override applies to the config source; a captured artifact "
                     "carries its own architecture")
    overrides = {}
    for entry in args.arch_override:
        key, sep, value = entry.partition("=")
        if not sep or not key:
            parser.error(f"--arch-override wants KEY=INT, got {entry!r}")
        try:
            overrides[key] = int(value)
        except ValueError:
            parser.error(f"--arch-override {key}: {value!r} is not an integer")
    args.arch_overrides = overrides
    if args.samples < 1 or args.blocks < 1 or args.warmup < 0:
        parser.error("--samples and --blocks must be >= 1, --warmup >= 0")


def _role(name: str, kernels) -> str:
    if name == _block.REFERENCE_PROVIDER:
        return "reference"
    origins = {getattr(s, "origin", "") for s in kernels.sources}
    if origins and all(o in CONTROL_ORIGINS for o in origins):
        return "control"
    return "candidate"


def build_block_providers(args, adapter, *, quiet: bool = False) -> dict[str, Any]:
    """The model scope's providers, on the block's registry, reference renamed."""
    from evograd.evaluation.tier3.cli import build_providers

    providers = build_providers(args, quiet=quiet, registry=adapter.registry)
    native = providers.pop("eager")
    return {_block.REFERENCE_PROVIDER: native, **providers}


def _load_policy(path: Path, adapter, environment) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = _gate.check_policy_binding(policy, adapter, environment)
    if problems:
        raise _block.BlockError(f"frozen policy {path} does not apply: " + "; ".join(problems))
    return policy


def run_worker(args, parser) -> dict[str, Any]:
    """Child mode: one provider (or the policy calibration), returned as its entry."""
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.runner import _environment

    adapter = parser_adapter(args)
    providers = build_block_providers(args, adapter, quiet=True)
    if args.provider == POLICY_PROVIDER:
        patch_sets = {_gate.patch_key(k): k for n, k in providers.items()
                      if n != _block.REFERENCE_PROVIDER and k.patched}
        return {"ok": True, "provider": POLICY_PROVIDER, "policy": _gate.calibrate_policy(
            adapter, patch_sets, ops=TASKS, device=args.device,
            controls=tuple(c for c in args.calibrate_with.split(",") if c),
            noise_repeats=args.noise_repeats, environment=_environment())}
    if args.provider not in providers:
        return {"ok": False, "provider": args.provider, "failed_at": "setup", "latency": None,
                "memory": None, "error": f"provider {args.provider!r} was not built in this process"}
    kernels = providers[args.provider]
    role = _role(args.provider, kernels)
    policy = None
    if args.block_policy is not None:
        policy = _load_policy(args.block_policy, adapter, _environment())
    reference = None
    if role != "reference":
        reference, _built, _inv = _block.native_reference(adapter, device=args.device)
        del _built, _inv
    return _block.measure_block_one(
        adapter, args.provider, kernels, role=role, device=args.device, ops=TASKS,
        policy=policy, reference=reference, verify=not args.no_verify,
        purity=not args.no_purity, noise_repeats=args.noise_repeats,
        warmup=args.warmup, samples=args.samples, blocks=args.blocks,
    )


def parser_adapter(args):
    from evograd.evaluation.tier3.cli import tier3_adapter

    return tier3_adapter(args.model).block(args)


def main_block(args, parser, argv: list[str]) -> int:
    """Parent mode: calibrate, then every provider, then the report."""
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3.cli import _run_isolated
    from evograd.evaluation.tier3.runner import _environment, provider_order

    adapter = parser_adapter(args)
    providers = build_block_providers(args, adapter)
    order = provider_order(providers, seed=args.seed if args.order_seed is None else args.order_seed)
    print(f"[tier3/block] case {adapter.case.case_id} ({adapter.case.case_hash[:12]}) "
          f"providers {list(providers)}", file=sys.stderr, flush=True)

    # The policy: frozen by the user, or calibrated here first -- in a child of
    # its own so nothing the controls did reaches a provider.
    child_argv = list(argv)
    if args.block_policy is not None:
        policy = _load_policy(args.block_policy, adapter, _environment())
        policy_path = args.block_policy
        print(f"[tier3/block] frozen policy {policy_path} ({policy['policy_hash'][:12]})",
              file=sys.stderr, flush=True)
    else:
        print("[tier3/block] calibrating the policy from trusted controls", file=sys.stderr, flush=True)
        if args.no_isolate:
            calibrated = run_worker(_with(args, provider=POLICY_PROVIDER), parser)
        else:
            calibrated = _run_isolated(child_argv, POLICY_PROVIDER, args.timeout)
        if not calibrated.get("ok"):
            print(f"[tier3/block] calibration FAILED: {calibrated.get('error')}", file=sys.stderr)
            policy, policy_path = None, None
        else:
            policy = calibrated["policy"]
            policy_path = args.freeze_policy or (
                args.out.with_suffix(".policy.json") if args.out else
                Path(tempfile.mkstemp(prefix="evograd_block_policy_", suffix=".json")[1]))
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[tier3/block] policy {policy['policy_hash'][:12]} frozen at {policy_path}",
                  file=sys.stderr, flush=True)
            child_argv += ["--block-policy", str(policy_path)]

    results: dict[str, Any] = {}
    for index, name in enumerate(order):
        print(f"[tier3/block] {index + 1}/{len(order)} {name}", file=sys.stderr, flush=True)
        if args.no_isolate:
            results[name] = run_worker(_with(args, provider=name, block_policy=policy_path), parser)
        else:
            results[name] = _run_isolated(child_argv, name, args.timeout)

    report = _block.assemble_block_report(
        adapter, results, order, policy=policy, ops=TASKS, seed=args.seed,
        isolation=("single process" if args.no_isolate
                   else f"one child process per provider, {args.timeout}s budget"),
        warmup=args.warmup, samples=args.samples, blocks=args.blocks,
        verify=not args.no_verify, purity=not args.no_purity,
    )
    if policy_path is not None:
        report["policy_file"] = str(policy_path)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    print_summary(report)
    return 0


def _with(args, **overrides):
    import copy

    clone = copy.copy(args)
    for key, value in overrides.items():
        setattr(clone, key, value)
    return clone


def print_summary(report: dict[str, Any]) -> None:
    from evograd.evaluation.tier3.report import provider_rows

    for row in provider_rows(report):
        if not row.ok:
            print(f"  {row.provider:24} FAILED [{row.failed_at}] {row.error}", file=sys.stderr)
            continue
        entry = report["providers"][row.provider]
        peak = row.execution_peak_bytes
        saved = row.saved_state_bytes
        speed = row.speedup_vs_reference
        print(
            f"  {row.provider:24} {row.latency_ms:9.3f} ms fwd+vjp  "
            f"{'  n/a ' if peak is None else f'{peak / 2**30:5.2f}'} GiB peak  "
            f"{'  n/a ' if saved is None else f'{saved / 2**20:6.1f}'} MiB saved  "
            f"{'ref' if speed is None else f'{speed:.3f}x'}  "
            f"role={entry.get('role')} patched={entry.get('patched')}",
            file=sys.stderr,
        )
