#!/usr/bin/env python3
"""GEAK v4 (AMD-AGI, ``kernel_workflow``) on an adbench-tools task, on NVIDIA.

    bash candidates/geak/setup.sh                       # once: NVIDIA working copy
    python candidates/geak/run.py --task TASK --run-dir RUN --budget 30 \\
        [--claude-model MODEL] [--gpu 0]

GEAK as shipped is a Claude Code Workflow (``kernel_workflow.js``): a Director,
a TechLead and specialist engineer agents, each writing code, running the
task's immutable unittest, profiling, and having every claimed speedup
re-verified. It is run here in its **author** mode -- "there is NO existing
source to optimize ... write a fresh implementation in target_language" --
which is the from-a-specification setting every other agent is in:

* ``kernel_path`` is an op task dir built by ``prepare.py``: ``meta.json``
  (the op spec), ``TASK.md`` (the shared statement), an immutable
  ``unittest.py`` that runs the evograd evaluator on
  ``kernel_src/candidate.py`` and logs every call to this run's ledger, and a
  frozen eager-autograd ``baseline_src/``.
* ``target_language`` is ``triton``; ``task`` (GEAK's natural-language steer)
  says the device is NVIDIA, because GEAK's knowledge base is AMD-specific and
  would otherwise steer toward MFMA/wave64/HIP. Nothing else is steered.
* Claude Code is launched headless with one instruction: call the Workflow tool
  with those args. GEAK's own prompts (``roles/*.md``) are untouched.

Differences that cannot be removed, and are recorded:

* GEAK's model is whatever Claude Code runs (``--claude-model``); the other
  agents use ``--model`` through an OpenAI-compatible API. For model parity run
  all agents on the same model, or report GEAK as a separate point.
* GEAK's budget unit is an *optimization direction* (an engineer session that
  may run the unittest many times). ``--geak-budget`` defaults to
  ``max(2, budget // 5)``; the ledger counts the evaluator calls actually made,
  which is the comparable number.
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
        except (ValueError, AttributeError):
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
    p.add_argument("--claude-model", default=os.environ.get("ANTHROPIC_MODEL"),
                   help="model Claude Code runs GEAK with (default: Claude Code's)")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p.add_argument("--time-limit", type=int, default=6 * 3600, help="seconds")
    args = p.parse_args(argv)
    geak_budget = args.geak_budget or max(2, args.budget // 5)
    workflow_dir = (args.geak_root / "kernel_workflow").resolve()
    if not (workflow_dir / "kernel_workflow.js").is_file():
        raise SystemExit(f"no GEAK working copy at {args.geak_root}; run candidates/geak/setup.sh")
    if shutil.which("claude") is None:
        raise SystemExit("the claude CLI is not on PATH")
    task, ledger = _common.start(args, AGENT, {
        "geak_commit": (args.geak_root / ".adbench_tools_geak_commit").read_text().strip()
        if (args.geak_root / ".adbench_tools_geak_commit").is_file() else None,
        "mode": "author", "target_language": "triton", "geak_budget": geak_budget,
        "claude_model": args.claude_model, "gpu": args.gpu,
        "model_note": "GEAK runs on Claude Code's model, not --model",
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
    command = ["claude", "-p", prompt(wf_args, workflow_dir), "--dangerously-skip-permissions",
               "--output-format", "json"]
    if args.claude_model:
        command += ["--model", args.claude_model]
    status = "completed"
    try:
        completed = subprocess.run(command, cwd=ledger.root, env=env, capture_output=True,
                                   text=True, timeout=args.time_limit)
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        status = "time_limit"
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    (ledger.root / "claude_stdout.json").write_text(stdout or "", encoding="utf-8")
    (ledger.root / "claude_stderr.txt").write_text(stderr or "", encoding="utf-8")
    reply = None
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
