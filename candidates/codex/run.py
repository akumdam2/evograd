#!/usr/bin/env python3
"""Codex (OpenAI's coding agent CLI) on an adbench-tools task.

    npm install -g @openai/codex                  # or a release binary on PATH
    python candidates/codex/run.py --task TASK --run-dir RUN --budget 30 [--effort LEVEL] [--gpu 0]

Codex as shipped, headless (``codex exec``), in the workspace ``_coding_agent``
builds: ``TASK.md``, ``candidate.py`` (the skeleton) and a metered
``evaluate.py`` that runs the evograd evaluator. Its prompt is
``_coding_agent.prompt``, the same one Claude Code gets: a short paragraph on
the workspace, then the problem statement verbatim. Its own system prompt and
tools are untouched. ``~/.codex/config.toml`` is not loaded
(``--ignore-user-config``), so a run does not depend on the account.

Codex is the coding agent that shares the other agents' model: it runs on
``--model`` at ``--api-base`` (``openai_base_url``) with ``$OPENAI_API_KEY``
(passed as ``CODEX_API_KEY`` unless that is already set).

Budget mapping: ``--budget N`` evaluations through ``evaluate.py``, enforced
exactly (see ``_coding_agent``).

Difference that cannot be removed, and is recorded: Codex runs with
``--dangerously-bypass-approvals-and-sandbox`` and a shell on the GPU node.
Its own sandbox blocks the GPU and the Triton cache, so the node is its
sandbox. Nothing it is given names the repository (see ``_coding_agent``), but
as the same user it could still search for it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _coding_agent  # noqa: E402
import _common  # noqa: E402

AGENT = "codex"
OPENAI_API_BASE = "https://api.openai.com/v1"


def command(args, workspace: Path) -> list[str]:
    """``codex exec`` on the prompt read from stdin."""
    command = ["codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
               "--skip-git-repo-check", "--ignore-user-config", "-C", str(workspace),
               "-m", args.model]
    if args.api_base.rstrip("/") != OPENAI_API_BASE:
        command += ["-c", f"openai_base_url={json.dumps(args.api_base)}"]
    if args.effort:
        command += ["-c", f"model_reasoning_effort={json.dumps(args.effort)}"]
    return [*command, "-"]


def read_events(log: Path) -> tuple[list[dict], str | None]:
    """The stream's usage events (one per turn) and the agent's last message."""
    usage, reply = [], None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("usage"), dict):
            usage.append(event)
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            reply = item.get("text")
    return usage, reply


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    _coding_agent.add_arguments(p)
    args = p.parse_args(argv)
    if shutil.which("codex") is None:
        raise SystemExit("the codex CLI is not on PATH")
    task, ledger = _common.start(args, AGENT, {
        "codex_version": _coding_agent.cli_version("codex"), "effort": args.effort,
        "gpu": args.gpu, "evaluations": args.budget, "time_limit": args.time_limit,
    })

    workspace = _coding_agent.Workspace(task, ledger, AGENT, args)
    log = ledger.root / "codex_events.jsonl"
    env = workspace.environment(args.gpu)
    if env.get("OPENAI_API_KEY") and not env.get("CODEX_API_KEY"):
        env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]
    prompt = (ledger.root / _coding_agent.PROMPT).read_text(encoding="utf-8")
    status = _coding_agent.launch(command(args, workspace.root), workspace.root, env, log,
                                  args.time_limit, input_text=prompt)

    usage, reply = read_events(log)
    for event in usage:
        ledger.log_llm({"source": "codex", "event": event.get("type"), "usage": event["usage"]})
    best, extra = workspace.collect()
    return _common.finish(args, task, ledger, best, status=None if status == "completed" else status,
                          reply=reply, **extra)


if __name__ == "__main__":
    raise SystemExit(main())
