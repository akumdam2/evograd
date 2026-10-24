#!/usr/bin/env python3
"""Claude Code (Anthropic's coding agent CLI) on an adbench-tools task.

    python candidates/claude_code/run.py --task TASK --run-dir RUN --budget 30 \\
        [--claude-model MODEL] [--effort LEVEL] [--gpu 0]

Claude Code as shipped, headless (``claude -p``), in the workspace
``_coding_agent`` builds: ``TASK.md``, ``candidate.py`` (the skeleton) and a
metered ``evaluate.py`` that runs the evograd evaluator. Its prompt is
``_coding_agent.prompt``, the same one Codex gets: a short paragraph on the
workspace, then the problem statement verbatim. Its own system prompt, tools
and subagents are untouched. Settings and MCP servers of whoever launches it
are not loaded (``--setting-sources project,local --strict-mcp-config``; the
workspace has no project settings), so a run does not depend on the account.

Budget mapping: ``--budget N`` evaluations through ``evaluate.py``, enforced
exactly (see ``_coding_agent``).

Differences that cannot be removed, and are recorded:

* Claude Code runs on an Anthropic model (``--claude-model``), not ``--model``.
  For model parity run all agents on the same model, or report it as a
  separate point, as for GEAK.
* It runs with ``--dangerously-skip-permissions`` and a shell on the GPU node:
  the node is its sandbox, and nothing but the prompt keeps it inside the
  workspace.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _coding_agent  # noqa: E402
import _common  # noqa: E402

AGENT = "claude_code"


def command(args, prompt: str) -> list[str]:
    command = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--dangerously-skip-permissions", "--setting-sources", "project,local",
               "--strict-mcp-config"]
    if args.claude_model:
        command += ["--model", args.claude_model]
    if args.effort:
        command += ["--effort", args.effort]
    return command


def session_result(log: Path) -> dict | None:
    """The stream's final ``result`` event: reply, cost, turns, usage."""
    result = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    return result


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    _coding_agent.add_arguments(p)
    p.add_argument("--claude-model", default=os.environ.get("ANTHROPIC_MODEL"),
                   help="model Claude Code runs on (default: Claude Code's)")
    args = p.parse_args(argv)
    if shutil.which("claude") is None:
        raise SystemExit("the claude CLI is not on PATH")
    task, ledger = _common.start(args, AGENT, {
        "claude_code_version": _coding_agent.cli_version("claude"),
        "claude_model": args.claude_model, "effort": args.effort, "gpu": args.gpu,
        "evaluations": args.budget, "time_limit": args.time_limit,
        "model_note": "Claude Code runs on --claude-model, not --model",
    })

    workspace = _coding_agent.Workspace(task, ledger, AGENT, args)
    log = ledger.root / "claude_stream.jsonl"
    env = workspace.environment(args.gpu)
    env.update({"IS_SANDBOX": "1", "DISABLE_AUTOUPDATER": "1"})  # the version in run.json holds
    prompt = (ledger.root / _coding_agent.PROMPT).read_text(encoding="utf-8")
    status = _coding_agent.launch(command(args, prompt), workspace.root, env, log,
                                  args.time_limit)

    result = session_result(log)
    if result is not None:
        ledger.log_llm({"source": "claude-code", "total_cost_usd": result.get("total_cost_usd"),
                        "num_turns": result.get("num_turns"), "usage": result.get("usage"),
                        "model_usage": result.get("modelUsage"),
                        "subtype": result.get("subtype"), "is_error": result.get("is_error")})
    best, extra = workspace.collect()
    return _common.finish(args, task, ledger, best, status=None if status == "completed" else status,
                          reply=(result or {}).get("result"), **extra)


if __name__ == "__main__":
    raise SystemExit(main())
