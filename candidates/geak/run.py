#!/usr/bin/env python3
"""Run GEAK's author-mode kernel workflow on an ADBench task, on NVIDIA.

    bash candidates/geak/setup.sh
    python candidates/geak/run.py --task TASK --run-dir RUN --model MODEL

The default Codex backend runs GEAK's unmodified JS workflow through its Node
runtime, using the shared --model and --api-base (Responses API). The optional
--backend claude retains the native Claude Workflow path and --claude-model.
Both paths use the same task preparation, evaluator and final measurement.
GEAK's budget remains optimization directions, default max(2, budget // 5).
API requests are counted through the shared proxy without a request cap.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "geak"
HERE = Path(__file__).resolve().parent
DEFAULT_WORK = HERE / ".work" / "GEAK"


def workflow_args(task, op_task: Path, workflow_dir: Path, exp_root: Path, budget: int,
                  device: str) -> dict:
    spec = task.spec
    return {
        "kernel_path": str(op_task),
        "workflow_dir": str(workflow_dir),
        "budget": budget,
        "gpu_ids": "0",
        "mode": "author",
        "target_language": "triton",
        "op_spec": {
            "op_kind": spec.get("family") or task.op_name,
            "shapes": {e["name"]: e.get("size") for e in spec["abi"]["arguments"]},
            "dtype": spec["workload"]["dtype"],
            "math_contract": "see TASK.md and meta.json (forward and backward)",
            "regime": "training (forward+backward)",
        },
        "task": (f"The GPU is {device} (NVIDIA, CUDA): AMD-specific guidance (rocminfo, MFMA, "
                 "wave64, LDS, HIP) does not apply. Write Triton. The deliverable is "
                 "kernel_src/candidate.py as specified in TASK.md; unittest.py is the oracle and "
                 "prints the score."),
        "exp_root": str(exp_root),
        "apply_to_original": "false",
    }


def prompt(args_json: dict, workflow_dir: Path) -> str:
    return (
        "Run GEAK's kernel_workflow. Call the Workflow tool exactly once with\n"
        f"  scriptPath: {json.dumps(str(workflow_dir / 'kernel_workflow.js'))}\n"
        f"  args: {json.dumps(args_json, indent=2)}\n"
        "Pass args as a JSON object, not a string. Do not edit files or run other commands "
        "yourself. When the workflow returns, reply with its result JSON and nothing else."
    )


def find_candidate(exp_root: Path, reply: str | None) -> Path | None:
    """The canonical workspace's candidate (GEAK commits each round's winner there)."""
    eval_dir = None
    if reply:
        try:
            eval_dir = json.loads(reply[reply.find("{"): reply.rfind("}") + 1]).get("eval_dir")
        except (ValueError, AttributeError, TypeError):
            eval_dir = None
    if eval_dir and (Path(eval_dir) / "workspace/kernel_src/candidate.py").is_file():
        return Path(eval_dir) / "workspace/kernel_src/candidate.py"
    found = sorted(exp_root.glob("**/workspace/kernel_src/candidate.py"),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--geak-root", type=Path, default=DEFAULT_WORK,
                   help="NVIDIA working copy from setup.sh")
    p.add_argument("--geak-budget", type=int, default=None,
                   help="GEAK optimization directions (default max(2, budget // 5))")
    p.add_argument("--backend", choices=("codex", "claude"), default="codex")
    p.add_argument("--claude-model", default=None,
                   help="model Claude Code runs GEAK with (default: Claude Code's)")
    p.add_argument("--effort", default=None,
                   help="backend reasoning effort (default: GEAK/backend default)")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p.add_argument("--time-limit", type=int, default=6 * 3600, help="seconds")
    p.set_defaults(api_base=None)
    args = p.parse_args(argv)
    args.api_base = args.api_base or (_common.DEFAULT_API_BASE if args.backend == "codex"
                                     else _common.DEFAULT_ANTHROPIC_API_BASE)
    if args.backend == "codex" and args.claude_model:
        p.error("--claude-model requires --backend claude; use --model for Codex")
    if args.backend == "claude" and args.claude_model is None:
        args.claude_model = os.environ.get("ANTHROPIC_MODEL")
    if args.geak_budget is not None and args.geak_budget < 1:
        p.error("--geak-budget must be positive")
    geak_budget = args.geak_budget or max(2, args.budget // 5)
    workflow_dir = (args.geak_root / "kernel_workflow").resolve()
    if not (workflow_dir / "kernel_workflow.js").is_file():
        raise SystemExit(f"no GEAK working copy at {args.geak_root}; run candidates/geak/setup.sh")
    runtime = args.geak_root.resolve() / "interface/runtime/engine/run_workflow.mjs"
    if args.backend == "codex" and not runtime.is_file():
        raise SystemExit("GEAK working copy lacks the Codex runtime; rerun candidates/geak/setup.sh")
    for executable in (("node", "codex") if args.backend == "codex" else ("claude",)):
        if shutil.which(executable) is None:
            raise SystemExit(f"the {executable} CLI is not on PATH")
    task, ledger = _common.start(args, AGENT, {
        "geak_commit": (args.geak_root / ".adbench_tools_geak_commit").read_text().strip()
        if (args.geak_root / ".adbench_tools_geak_commit").is_file() else None,
        "budget": geak_budget, "budget_unit": "optimization_directions",
        "mode": "author", "target_language": "triton",
        "claude_model": args.claude_model, "effort": args.effort, "gpu": args.gpu,
        "backend": args.backend, "api": "openai" if args.backend == "codex" else "anthropic",
    })

    from prepare import prepare  # candidates/geak/prepare.py

    op_task = prepare(task, ledger.root, ledger.root / "op_task")
    exp_root = ledger.root / "exp"
    exp_root.mkdir(exist_ok=True)
    device = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name",
                             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    wf_args = workflow_args(task, op_task, workflow_dir, exp_root, geak_budget, device or "NVIDIA GPU")
    (ledger.root / "workflow_args.json").write_text(json.dumps(wf_args, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "IS_SANDBOX": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PROFILER_PRIORITY": "metrix",
        "GPU_IDENTITY_SCRIPT": str(args.geak_root.resolve() / "scripts" / "gpu_identity.py"),
        "PATH": os.pathsep.join([str(HERE / "nvidia" / "bin"), str(Path(sys.executable).parent),
                                 env.get("PATH", "")]),
        "PYTHONPATH": os.pathsep.join([str(_common.REPO / "src"), str(task.root),
                                       env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    })
    result_file = ledger.root / "workflow_result.json"
    if args.backend == "codex":
        # --model in GEAK's runtime names a registry entry, not an API model.
        # Use its raw-model environment setting and clear inherited selection.
        for name in ("GEAK_AGENT_PROFILE", "GEAK_MODEL", "GEAK_REGISTRY",
                     "GEAK_CODEX_EXTRA_ARGS", "GEAK_CODEX_BIN", "OPENAI_CUSTOM_HEADERS",
                     "GEAK_AMDKEY"):
            env.pop(name, None)
        env.update({"GEAK_AGENT_BACKEND": "codex", "GEAK_CODEX_MODEL": args.model,
                    "GEAK_CODEX_AUTOCONFIG": "1", "OPENAI_BASE_URL": args.api_base,
                    "GEAK_CODEX_EXTRA_ARGS": "--skip-git-repo-check "
                    "-c model_providers.geak_auto.requires_openai_auth=false "
                    "-c model_providers.geak_auto.supports_websockets=false"})
        if args.effort:
            env["GEAK_CODEX_EFFORT"] = args.effort
        command = ["node", str(runtime), str(workflow_dir / "kernel_workflow.js"),
                   "--agent", "codex", "--args-file", str(ledger.root / "workflow_args.json"),
                   "--result-file", str(result_file),
                   "--metrics-file", str(ledger.root / "workflow_metrics.json")]
    else:
        command = ["claude", "-p", prompt(wf_args, workflow_dir), "--dangerously-skip-permissions",
                   "--output-format", "json"]
        if args.claude_model:
            command += ["--model", args.claude_model]
        if args.effort:
            command += ["--effort", args.effort]
    status = "completed"
    try:
        completed = subprocess.run(command, cwd=ledger.root, env=env, capture_output=True,
                                   text=True, timeout=args.time_limit)
        stdout, stderr = completed.stdout, completed.stderr
        if getattr(completed, "returncode", 0) != 0:
            status = "workflow_failed"
    except subprocess.TimeoutExpired as exc:
        status = "time_limit"
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    prefix = "workflow" if args.backend == "codex" else "claude"
    stdout_file = "workflow_stdout.txt" if args.backend == "codex" else "claude_stdout.json"
    (ledger.root / stdout_file).write_text(stdout or "", encoding="utf-8")
    (ledger.root / f"{prefix}_stderr.txt").write_text(stderr or "", encoding="utf-8")
    reply = None
    if args.backend == "codex":
        if result_file.is_file():
            reply = result_file.read_text(encoding="utf-8")
    else:
        try:
            session = json.loads(stdout)
            reply = session.get("result")
            ledger.log_llm({"source": "claude-code", "total_cost_usd": session.get("total_cost_usd"),
                            "num_turns": session.get("num_turns"), "usage": session.get("usage")})
        except (ValueError, TypeError):
            pass
    candidate = find_candidate(exp_root, reply)
    best = None
    if candidate is not None:
        best = ledger.root / "best.py"
        best.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
    return _common.finish(args, task, ledger, best, status=None if status == "completed" else status,
                          geak_reply=reply)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
