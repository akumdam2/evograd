"""Construct and verify a deterministic seed for one deployment configuration."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evograd.opdecl.activity import OpDecl, Workload
from evograd.pipelines.unified.contract import CONTRACT_FILE, build_contract, validate_source
from evograd.pipelines.shared.runner import report_passed, verify_candidate


@dataclass(frozen=True)
class UnifiedSeedConfig:
    op: OpDecl
    workload: Workload
    output_dir: Path
    python: str = sys.executable
    device: str = "cuda"
    eval_timeout: int = 180


def select_workload(op: OpDecl, *, dtype: str | None = None,
                    dims: dict[str, int] | None = None) -> Workload:
    """Select an exact configuration, without perturbing dimensions."""
    cases = op.correctness or op.benchmark
    if dtype is None:
        if not cases:
            raise ValueError("supply --dtype and --dims for a declaration without workloads")
        dtype = cases[0].dtype
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported seed dtype: {dtype}")
    matching = [case for case in cases if case.dtype == dtype]
    if dims is None:
        if not matching:
            raise ValueError(f"no declared {dtype} workload; supply --dims explicitly")
        return matching[0]
    message = f"--dims must bind every dimension {op.dims} to a positive integer"
    if set(dims) != set(op.dims) or any(type(v) is not int for v in dims.values()):
        raise ValueError(message)
    # A declared configuration is taken as declared: some dims are parameters,
    # not sizes (a convolution's padding P is 0 at every patchify stem).
    declared = next((case for case in matching if case.dims == dims), None)
    if declared is not None:
        return declared
    if any(v <= 0 for v in dims.values()):
        raise ValueError(message)
    return Workload(dims=dict(dims), dtype=dtype)


def scoped_op(op: OpDecl, workload: Workload, *, declaration: str | None = None) -> OpDecl:
    """The same operator/oracle/tolerances, restricted to the captured c.

    Each scalar argument's default becomes the value the input generator passes
    at ``workload`` (``case_scalars``), so every signature and argument list
    derived from the scoped declaration states the configuration's own eps or
    stride rather than the operator's generic one.
    """
    from evograd.opdecl.inputs import case_scalars

    scalars = case_scalars(op, workload)
    args = tuple(replace(arg, default=scalars[arg.name]) if arg.name in scalars else arg
                 for arg in op.args)
    # A user-selected deployment shape is not an official benchmark case and
    # must not invent model provenance to satisfy the benchmark taxonomy.
    return replace(op, level=None, family=None, args=args,
                   correctness=(workload,), coverage=(workload,), benchmark=(workload,),
                   benchmark_suites={}, regime_feature=None, regime_split=None, case_weight=None,
                   declaration=declaration if declaration is not None else op.declaration)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_declaration(path: Path, op: OpDecl, workload: Workload) -> None:
    # Reuse the trusted input factory, oracle and tolerances through the original
    # declaration. This file also lets the existing evolve/verify CLIs use c.
    # The workload is written whole: its provenance (recorded strides, a norm's
    # eps, the model a generator reads its tables from) decides the inputs the
    # declaration generates, so dropping it would scope a different case.
    base = (f"load_task({op.declaration!r})" if op.declaration
            else f"get_task({op.name!r})")
    source = f'''"""The fixed deployment configuration of this unified-pipeline seed."""
from dataclasses import replace
from pathlib import Path
from evograd.benchmark import get_task, load_task
from evograd.opdecl.activity import Provenance, Workload
from evograd.pipelines.unified.synthesize import scoped_op

_base = {base}
_case = {workload!r}
op = scoped_op(replace(_base, forward={op.forward!r}), _case,
               declaration=str(Path(__file__).resolve()) + ":op")
'''
    path.write_text(source, encoding="utf-8")


def synthesize_unified_seed(config: UnifiedSeedConfig) -> int:
    """Capture compiler prior, lower R, and publish only an accepted seed.

    There is no LLM, repair search or timing objective in seed synthesis.
    Unsupported lowering fails with diagnostics instead of adopting Inductor's
    Triton or silently substituting eager math.
    """
    from evograd.pipelines.unified.capture import capture_unified_pair
    from evograd.pipelines.unified.codegen import generate_unified_seed

    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    published = root / "initial_program_autograd_pair.py"
    if published.exists():
        raise FileExistsError(f"an accepted seed already exists at {published}; use a new directory")
    phase = "configuration"
    try:
        import torch
        import triton

        config.op.validate()
        select_workload(config.op, dtype=config.workload.dtype, dims=config.workload.dims)
        scoped_op(config.op, config.workload).validate()
        if not config.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("unified seed synthesis and acceptance require a CUDA device")
        if config.eval_timeout <= 0:
            raise ValueError("eval_timeout must be positive")
        phase = "capture"
        print(f"[unified] Capture {config.op.name}: {config.workload.dims} / {config.workload.dtype}")
        captured = capture_unified_pair(config.op, config.workload, device=config.device)
        graphs = root / "graphs"
        graphs.mkdir(exist_ok=True)
        for side in ("joint", "forward", "backward"):
            _write_json(graphs / f"{side}.json", getattr(captured, f"{side}_graph"))
        raw = root / "compiler_prior"
        raw.mkdir(exist_ok=True)
        for side in ("forward", "backward"):
            (raw / f"{side}.py").write_text(getattr(captured.pair, f"{side}_source"), encoding="utf-8")
        manifest = {
            "pipeline": "unified", "llm_calls": 0,
            "configuration": {"dims": dict(config.workload.dims), "dtype": config.workload.dtype,
                              "device": captured.pair.device, "hardware": torch.cuda.get_device_name(config.device),
                              "static_parameters": captured.pair.baked_scalars},
            "versions": {"torch": str(torch.__version__), "triton": triton.__version__},
            "external_regions": [asdict(region) for region in captured.external_regions],
            "evolvable_nodes": {
                side: [node["name"] for node in getattr(captured, f"{side}_graph")["nodes"]
                       if node["op"] == "call_function" and not node.get("external_kernel")]
                for side in ("forward", "backward")
            },
            "saved": [asdict(value) for value in captured.pair.saved],
            "backward_arg_spec": captured.pair.backward_arg_spec,
            "grad_indices": captured.pair.grad_indices,
        }
        _write_json(root / "partition.json", manifest)
        phase = "lowering"
        print(f"[unified] Lower ATen primitives; preserve {len(captured.external_regions)} external regions")
        source = generate_unified_seed(config.op, config.workload, captured)
        from evograd.pipelines.shared.primitives import check_source

        check_source(source)
        contract = build_contract(source)
        validate_source(source, contract)
        _write_json(root / CONTRACT_FILE, contract)
        candidate = root / "candidate.py"
        candidate.write_text(source, encoding="utf-8")
        declaration = root / "declaration.py"
        _write_declaration(declaration, config.op, config.workload)
        phase = "verification"
        print("[unified] Verify forward, every gradient and the deployment interface at the captured configuration")
        report = verify_candidate(
            python=config.python, op_name=config.op.name, program_path=candidate,
            declaration=f"{declaration}:op", forward=config.op.forward,
            log_dir=root, timeout=config.eval_timeout, device=config.device,
            seed_acceptance=True,
        )
        _write_json(root / "verification_report.json", report)
        if not report_passed(report):
            print(f"[unified] Rejected; see {root / 'verification_report.json'}")
            return 1
        candidate.replace(published)
        print(f"[unified] Verified seed: {published}")
        print(f"[unified] Use --declaration {declaration}:op when verifying or evolving this fixed-configuration seed")
        return 0
    except Exception:
        failure = {"phase": phase, "error": traceback.format_exc()}
        _write_json(root / "failure.json", failure)
        print(f"[unified] Failed during {phase}:\n{failure['error']}", file=sys.stderr)
        return 1
