"""adbench-tools: run other kernel agents on adbench, on the same terms as evograd.

This package is the agent-neutral part of that comparison. It uses the
benchmark and evaluator; it changes neither. Everything specific to one agent
(GEPA, GEAK, OpenEvolve without the frontend, KernelAgent, Claude Code, Codex,
and evograd itself) lives outside the package, under ``candidates/<agent>/``.

The fairness protocol it implements, in the order a run meets it:

1. **One task pack per operator configuration** (:mod:`.task`). It holds what
   the evograd frontend is given and nothing else: the declared PyTorch forward
   reference, its argument activity and shapes, and one fixed configuration
   (dims, dtype) selected exactly as ``evograd seed unified`` selects it. The
   declaration it writes is the frontend's own scoped declaration, so the
   oracle, the tolerances and the timed case are the ones evograd is measured
   on.
2. **One problem statement** (:mod:`.statement`). Its first section is the
   evograd agent's OpenEvolve system message for this operator, verbatim; the
   rest replaces what evograd's seed carries (the computation) with the
   forward reference source, and states the rules the evaluator already
   enforces. Each agent embeds this text in its own native prompt slot.
3. **One deliverable** (:mod:`.artifact`). Every agent produces the same
   module surface: ``<op>_forward_with_saved`` and ``<op>_backward_from_saved``,
   in Triton and/or PyTorch tensor code. The harness, not the agent, wraps it
   into an evograd artifact: the whole agent module goes inside one
   ``EVOLVE-BLOCK`` (so the static rules cover every line of it) followed by the
   declaration-generated deployment layer evograd's own pipelines emit.
4. **One evaluator** (:mod:`.evaluate`). In-loop feedback and the final score
   both come from ``evograd.evolve.evaluator_entry`` with the environment
   ``evograd evolve`` sets, run in a killable child. Agents with their own
   harness (GEAK's unittest, KernelAgent's test file, the coding agents'
   evaluate.py) call this module's CLI from inside that harness, so the
   verdict is never theirs.
5. **One ledger and one final measurement** (:mod:`.ledger`, :mod:`.score`).
   Every evaluation an agent requests is appended to ``evaluations.jsonl``,
   LLM usage is recorded where the agent exposes it, and the final program is
   re-measured with the same seed-acceptance gate the frontend's seed passes.
"""

TASK_SCHEMA = "adbench-tools-task/1"
RESULT_SCHEMA = "adbench-tools-result/1"
