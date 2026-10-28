"""What every candidate driver shares: flags, the run ledger, the final score.

Each ``candidates/<agent>/run.py`` is::

    args = parse(parser)                     # the common flags below + its own
    task, ledger = start(args, agent, config)
    best = <run the agent; return the path of its best module or None>
    finish(args, task, ledger, best)         # final.json via evograd.adbench_tools.score

For the OpenAI-compatible agents and Claude Code, ``--budget`` caps actual
model API requests, including initialization, reflection and retries, through
a shared local proxy (in front of OpenAI's or Anthropic's API). GEAK's requests
go through the same proxy and are counted, not capped: its budget unit is its
own optimization direction. Native iteration/evaluation limits remain stopping
controls, not API counters.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

DEFAULT_MODEL = os.environ.get("ADBENCH_MODEL", "gpt-5.6-sol")
DEFAULT_API_BASE = os.environ.get("ADBENCH_API_BASE", "https://api.openai.com/v1")
DEFAULT_ANTHROPIC_API_BASE = os.environ.get("ADBENCH_ANTHROPIC_API_BASE",
                                            "https://api.anthropic.com")
DEFAULT_BUDGET = int(os.environ.get("ADBENCH_BUDGET", "10"))
METERED_AGENTS = frozenset({"evograd", "openevolve", "gepa", "kernelagent", "codex", "claude_code"})
# The agents that run the claude CLI, on Anthropic's API; GEAK is counted, not capped.
ANTHROPIC_AGENTS = frozenset({"claude_code", "geak"})


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=Path, required=True, help="task pack (python -m evograd.adbench_tools task)")
    p.add_argument("--run-dir", type=Path, required=True, help="fresh directory for this run")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="model API request cap per case (default $ADBENCH_BUDGET or 10; "
                        "GEAK's requests are only counted, see its --geak-budget)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM for every agent call")
    p.add_argument("--api-base", default=DEFAULT_API_BASE,
                   help="OpenAI-compatible endpoint (key from $OPENAI_API_KEY); for Claude Code "
                        "and GEAK --backend claude, Anthropic's (default $ADBENCH_ANTHROPIC_API_BASE or "
                        "https://api.anthropic.com, key from $ANTHROPIC_API_KEY)")
    p.add_argument("--seed", type=int, default=0, help="the agent's own random seed")
    p.add_argument("--no-score", action="store_true",
                   help="skip the final measurement (run `python -m evograd.adbench_tools score` later)")
    return p


def start(args, agent: str, config: dict):
    from evograd.adbench_tools.ledger import RunLedger
    from evograd.adbench_tools.task import load

    task = load(args.task)
    search_feedback(task)
    if (args.run_dir / "run.json").exists():
        raise SystemExit(f"{args.run_dir} already holds a run; use a fresh --run-dir")
    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")
    anthropic = config.get("api", "anthropic" if agent in ANTHROPIC_AGENTS else "openai") == "anthropic"
    if anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; the request proxy forwards with it")
    # A driver with its own budget unit (GEAK) overrides budget and budget_unit.
    config = {"budget": args.budget,
              "budget_unit": "api_requests" if agent in METERED_AGENTS else "none",
              "model": args.model, "api_base": args.api_base,
              "seed": args.seed, **config}
    ledger = RunLedger(args.run_dir, agent=agent, task=task, config=config)
    if agent in METERED_AGENTS or anthropic or agent == "geak":
        from evograd.adbench_tools.llm_budget import LLMBudgetProxy

        key = (os.environ["ANTHROPIC_API_KEY"] if anthropic else
               os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"))
        args._budget_proxy = LLMBudgetProxy(
            args.api_base, args.budget if agent in METERED_AGENTS else None,
            ledger.root / "api_calls.jsonl", api_key=key,
            api="anthropic" if anthropic else "openai")
        args._budget_proxy.start()
        args._upstream_api_base = args.api_base
        args.api_base = args._budget_proxy.base_url
        # Drivers and their subprocesses only receive the metered endpoint and
        # a placeholder credential. The upstream key stays in the proxy.
        if anthropic:
            # The claude CLI appends /v1/messages. Its documented switches turn
            # off traffic that bypasses the proxy (telemetry, update and
            # availability checks) and the background session-title request.
            # A launching Claude Code session's entrypoint (claude-vscode)
            # would make the CLI ignore ANTHROPIC_API_KEY, so it is unset.
            environment = {"ANTHROPIC_API_KEY": "adbench-local",
                           "ANTHROPIC_BASE_URL": args.api_base,
                           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                           "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
                           "CLAUDE_CODE_ENTRYPOINT": None}
        else:
            environment = {"OPENAI_API_KEY": "adbench-local", "CODEX_API_KEY": "adbench-local",
                           "OPENAI_BASE_URL": args.api_base, "OPENAI_API_BASE": args.api_base,
                           "ADBENCH_API_BASE": args.api_base}
        # Nor does the agent get the other provider's real key.
        environment.update({name: None for name in
                            (("OPENAI_API_KEY", "CODEX_API_KEY") if anthropic
                             else ("ANTHROPIC_API_KEY",))})
        for name in ("NO_PROXY", "no_proxy"):
            environment[name] = ",".join(filter(None, (os.environ.get(name), "127.0.0.1", "localhost")))
        args._budget_environment = {name: os.environ.get(name) for name in environment}
        _set_environment(environment)
        atexit.register(close_budget, args)
    print(f"[{agent}] task {task.task_id} -> {ledger.root}", flush=True)
    return task, ledger


def search_feedback(task) -> None:
    """Point every evaluator this run starts at the pack's search-correctness rule.

    Evaluators started through ``evograd.adbench_tools.evaluate`` read it from
    the pack; OpenEvolve's (``run_evolve``) inherit this process's environment.
    Under the payload rule the corpus and its named payloads must be there now:
    a run never starts on a rule it cannot apply, and never falls back.
    """
    from evograd.adbench_tools.evaluate import search_rule
    from evograd.verification.payloads.corpus import Corpus, CorpusError
    from evograd.verification.payloads.gate1 import SEARCH_CORRECTNESS_ENV, SEARCH_RULE

    if search_rule(task) != SEARCH_RULE:
        os.environ.pop(SEARCH_CORRECTNESS_ENV, None)
        return
    spec = task.spec["search_correctness"]
    try:
        corpus = Corpus.open(spec["corpus"]["manifest"])
        if corpus.corpus_id != spec["corpus"]["corpus_id"]:
            raise CorpusError(f"it is {corpus.corpus_id}, not {spec['corpus']['corpus_id']}")
        for payload_id in spec["payload_ids"]:
            corpus.payload(payload_id)
    except (CorpusError, OSError, ValueError) as exc:
        raise SystemExit(f"{task.task_id}: search feedback needs {spec['corpus']['manifest']}: {exc}")
    os.environ[SEARCH_CORRECTNESS_ENV] = str(task.root / "task.json")


def budget_exhausted(args) -> bool:
    proxy = getattr(args, "_budget_proxy", None)
    return proxy is not None and proxy.rejected_requests > 0


def close_budget(args) -> dict:
    proxy = getattr(args, "_budget_proxy", None)
    if proxy is None:
        return {}
    proxy.close()
    record = {"limit": proxy.budget, "used": proxy.calls_used,
              "rejected": proxy.rejected_requests, "unit": "api_requests"}
    _set_environment(args._budget_environment)
    args.api_base = args._upstream_api_base
    args._budget_proxy = None
    return record


def _set_environment(values: dict[str, str | None]) -> None:
    """Set each variable, or unset it where the value is ``None``."""
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def finish(args, task, ledger, best: Path | None, *, status: str | None = None,
           contract: Path | None = None, **extra) -> int:
    if budget_exhausted(args):
        status = "budget_exhausted"
    budget = close_budget(args)
    if budget:
        extra["llm_budget"] = budget
    status = status or ("completed" if best else "no_program")
    ledger.finish(status=status, best_program=best, **extra)
    if best is None:
        print(f"[{ledger.record['agent']}] produced no program", flush=True)
        return 1
    print(f"[{ledger.record['agent']}] best program: {best}", flush=True)
    if args.no_score:
        return 0
    from evograd.adbench_tools.score import score

    result = score(task.root, best, run_dir=ledger.root, agent=ledger.record["agent"],
                   contract=contract)
    print(json.dumps({k: result.get(k) for k in
                      ("agent", "accepted", "correct", "combined_score", "metrics")},
                     indent=2, default=str), flush=True)
    return 0 if result.get("accepted") and result.get("correct") else 1


def best_evaluated(ledger, fallback: Path | None = None) -> Path | None:
    """Recover a correct artifact, the native fallback, or the last attempted one."""
    from evograd.adbench_tools.ledger import EVALUATIONS_FILE, read_lines

    attempted = [entry for entry in read_lines(ledger.root / EVALUATIONS_FILE)
                 if entry.get("program") and Path(entry["program"]).is_file()]
    choices = [entry for entry in attempted if entry.get("correct")]
    if choices:
        entry = max(choices, key=lambda e: float(e.get("combined_score") or 0.0))
        return Path(entry["program"])
    if fallback is not None and fallback.is_file():
        return fallback
    return Path(attempted[-1]["program"]) if attempted else None


def unified_seed(task, ledger, *, eval_timeout: int = 180) -> Path | None:
    """evograd's frontend on the task: ``evograd seed unified`` at the task's
    exact configuration, from the pack's own declaration. No LLM call. Returns
    the accepted seed artifact, or ``None`` (``<run>/seed/failure.json`` says why)."""
    from dataclasses import replace

    from evograd.benchmark import get_task, load_task
    from evograd.pipelines.unified.synthesize import (
        UnifiedSeedConfig,
        select_workload,
        synthesize_unified_seed,
    )

    spec = task.spec
    if spec["base_declaration"]:
        # As `evograd seed unified --declaration` does: keep the absolute reference.
        base = replace(load_task(spec["base_declaration"]), declaration=spec["base_declaration"])
    else:
        base = get_task(task.op_name)
    workload = select_workload(base, dtype=spec["workload"]["dtype"], dims=spec["workload"]["dims"])
    seed_dir = ledger.root / "seed"
    status = synthesize_unified_seed(UnifiedSeedConfig(
        op=base, workload=workload, output_dir=seed_dir, eval_timeout=eval_timeout))
    seed = seed_dir / "initial_program_autograd_pair.py"
    if status != 0 or not seed.is_file():
        return None
    ledger.log_evaluation({"kind": "seed_acceptance", "origin": "unified", "correct": True,
                           "program": str(seed)})
    return seed


def import_evaluations_from_archive(ledger, archive: Path, origin: str) -> int:
    """Copy an OpenEvolve program archive's index into the run ledger."""
    from evograd.adbench_tools.ledger import read_lines

    count = 0
    for entry in read_lines(archive / "index.jsonl"):
        metrics = entry.get("metrics") or {}
        ledger.log_evaluation({
            "kind": "full", "origin": origin,
            "correct": float(metrics.get("correct", entry.get("correct", 0.0)) or 0.0) == 1.0,
            "combined_score": metrics.get("combined_score"),
            "program": str(archive / entry["file"]) if entry.get("file") else None,
            "duplicate": bool(entry.get("duplicate")),
        })
        count += 1
    return count
