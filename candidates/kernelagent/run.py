#!/usr/bin/env python3
"""KernelAgent (Meta, ``triton_kernel_agent``) on an adbench-tools task.

    pip install -e references/KernelAgent        # at e0647170da36ef9b059ac0bd3d60103aa4ed378b
    python candidates/kernelagent/run.py --task TASK --run-dir RUN --budget 30

KernelAgent's two native stages, unchanged, with evograd's evaluator as the
only oracle:

1. **Generation** -- ``TritonKernelAgent.generate_kernel``: parallel workers
   write ``kernel.py``, run the test file, and refine on its output until one
   passes. ``problem_description`` is the task's problem statement plus the
   KernelAgent glue below. The test file is ours (``generate_default_test=False``):
   it runs ``python -m evograd.adbench_tools evaluate --correctness-only`` on
   ``kernel.py`` and prints the evaluator's feedback, and it checks that
   ``kernel_function`` is the pair it claims to be.
2. **Optimization** -- ``OptimizationManager`` with KernelAgent's own
   ``beam_search`` configuration: NCU profiling, roofline and bottleneck
   analysis, LLM rewrites, our test as the correctness gate, and KernelAgent's
   timing of ``kernel_function`` against ``problem.py``. KernelAgent times one
   callable, so ``problem.py`` presents the task as one *training step*: given
   the arguments and upstream gradients, return outputs and input gradients
   (eager autograd on the reference), and ``kernel_function`` does the same
   with the pair. That is the forward+backward step evograd's score weighs.

Glue KernelAgent needs and the other agents do not: its harness calls
``kernel_function``, so the module must define it (verbatim, below), and
KernelAgent imports ``kernel.py`` directly, so a granted primitive is imported
from the task pack's ``adbench_primitives`` (packaging strips the import).
KernelAgent's own static bans (no ``torch.sigmoid`` etc.) are stricter than
evograd's and stay on: they are part of the agent.

The shared proxy caps actual model requests, including concurrent workers and
retries. Native stopping controls split the budget between stages:
generation runs ``--gen-workers`` (default
4, KernelAgent's default) workers for ``ceil(budget / 2 / workers)`` rounds;
optimization runs ``max(1, budget // 2 // 4)`` rounds of beam search (4 workers
per round). The ledger counts every evaluation either stage requested.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "kernelagent"
KERNELAGENT_COMMIT = "e0647170da36ef9b059ac0bd3d60103aa4ed378b"
DEFAULT_ROOT = _common.REPO / "references" / "KernelAgent"
#: The same GH100 die and SM count as GH200; KernelAgent's database has no GH200
#: entry and its roofline needs one. Override with --gpu-name.
DEFAULT_GPU = "NVIDIA H100 SXM5 80GB"


def kernel_function_source(task) -> str:
    abi = task.spec["abi"]
    args = [a["name"] for a in abi["arguments"]]
    scalars = [a["name"] for a in abi["arguments"] if a["kind"] == "scalar"]
    upstream = abi["upstream_grads"]
    params = ", ".join([*args, *upstream])
    grads_in = upstream[0] if len(upstream) == 1 else f"({', '.join(upstream)})"
    scalar_kw = "".join(f", {name}={name}" for name in scalars)
    return f'''def kernel_function({params}):
    """KernelAgent's entry: one forward+backward step through the pair."""
    outputs, saved = {abi['forward_fn']}({", ".join(args)})
    grads = {abi['backward_fn']}({grads_in}, saved{scalar_kw})
    outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    grads = grads if isinstance(grads, (tuple, list)) else (grads,)
    return torch.cat([t.float().reshape(-1) for t in (*outputs, *grads)])
'''


def problem_description(task) -> str:
    granted = task.allowed_primitives
    primitives = (f"\n- Import granted primitives as `from adbench_primitives import "
                  f"{', '.join(granted)}`.") if granted else ""
    return f"""{task.statement}

## KernelAgent harness requirements

- Write the module as `kernel.py`. Besides the two pair functions, it must
  define this function verbatim (it is how this harness times one training
  step; do not optimize it, optimize the pair):

```python
{kernel_function_source(task)}```
- `import torch`, `import triton`, `import triton.language as tl` at the top.{primitives}
- The test prints the evograd evaluator's verdict; fix what it reports.
"""


def render_problem(task) -> str:
    """KernelBench-style problem: Model = one eager autograd step of the reference."""
    abi = task.spec["abi"]
    return f'''"""One forward+backward step of {task.op_name}, KernelBench style (adbench-tools).

Model(*inputs) takes the declared arguments then the upstream gradients and
returns outputs and input gradients, flattened to one float32 vector, computed
by eager PyTorch autograd on the runtime reference. Inputs are the evaluator's
own inputs at the task's configuration. Generated; do not edit.
"""
import sys

sys.path.insert(0, {str(_common.REPO / "src")!r})
import torch

from evograd.adbench_tools.task import load as _load
from evograd.opdecl.importing import resolve_callable as _resolve
from evograd.opdecl.inputs import make_case_inputs as _inputs

_TASK = _load({str(task.root)!r})
_OP = _TASK.op()
_REFERENCE = _resolve(_OP.runtime_forward or _OP.forward)
_ARGS = {[a["name"] for a in abi["arguments"]]!r}
_ACTIVE = {{a.name: a.grad_name for a in _OP.active_args()}}
_UPSTREAM = {abi["upstream_grads"]!r}
_GRAD_ORDER = {abi["grad_order"]!r}


class Model(torch.nn.Module):
    def forward(self, *inputs):
        args = list(inputs[: len(_ARGS)])
        upstream = inputs[len(_ARGS):]
        leaves = {{}}
        for i, name in enumerate(_ARGS):
            if name in _ACTIVE:
                args[i] = args[i].detach().requires_grad_(True)
                leaves[_ACTIVE[name]] = args[i]
        with torch.enable_grad():
            out = _REFERENCE(*args)
            outs = tuple(out) if isinstance(out, (tuple, list)) else (out,)
            order = [leaves[g] for g in _GRAD_ORDER]
            grads = torch.autograd.grad(outs, order, upstream)
        return torch.cat([t.detach().float().reshape(-1) for t in (*outs, *grads)])


def get_inputs():
    values = _inputs(_OP, _OP.benchmark[0], device="cuda")
    return [values[name] for name in _ARGS] + [values[name] for name in _UPSTREAM]


def get_init_inputs():
    return []
'''


def render_test(task, run_dir: Path) -> str:
    abi = task.spec["abi"]
    return f'''"""evograd's oracle for this KernelAgent run (adbench-tools). Do not edit.

1. The evograd evaluator's correctness gate on kernel.py (static rules, every
   output and gradient against torch.autograd, the timed shape).
2. kernel_function must be the forward+backward step of the pair it defines.
"""
import json
import subprocess
import sys

TASK = {str(task.root)!r}
RUN = {str(run_dir)!r}

proc = subprocess.run(
    [sys.executable, "-m", "evograd.adbench_tools", "evaluate", "--task", TASK, "--run-dir", RUN,
     "--correctness-only", "--origin", "kernelagent", "kernel.py"],
    capture_output=True, text=True)
try:
    report = json.loads(proc.stdout)
except json.JSONDecodeError:
    print("evaluator did not return a report:\\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
    sys.exit(1)
print(report["feedback"])
if not report["correct"]:
    print("FAIL: the evograd evaluator rejected kernel.py")
    sys.exit(1)

sys.path.insert(0, TASK)
import torch
import kernel
from problem import get_inputs

inputs = get_inputs()
names = {[a["name"] for a in abi["arguments"]]!r}
upstream = inputs[len(names):]
outputs, saved = kernel.{abi['forward_fn']}(*inputs[: len(names)])
grads = kernel.{abi['backward_fn']}(upstream[0] if len(upstream) == 1 else tuple(upstream), saved)
outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
grads = grads if isinstance(grads, (tuple, list)) else (grads,)
expected = torch.cat([t.float().reshape(-1) for t in (*outputs, *grads)])
actual = kernel.kernel_function(*inputs)
error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12))
if actual.shape != expected.shape or not error < 1e-2:
    print(f"FAIL: kernel_function is not the pair's forward+backward step (relative error {{error:.3g}})")
    sys.exit(1)
print("PASS")
sys.exit(0)
'''


def register_model(model: str, api_base: str) -> None:
    """Route ``model`` to KernelAgent's OpenAI provider at ``api_base``."""
    from utils.providers import models
    from utils.providers.model_config import ModelConfig
    from utils.providers.openai_provider import OpenAIProvider

    os.environ["OPENAI_BASE_URL"] = api_base
    table = models._get_model_name_to_config()
    if model not in table:
        table[model] = ModelConfig(name=model, provider_classes=[OpenAIProvider],
                                   description="registered by adbench-tools")


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--kernelagent-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gen-workers", type=int, default=4)
    p.add_argument("--opt-strategy", default="beam_search",
                   help="a config name under KernelAgent's examples/configs (default beam_search)")
    p.add_argument("--gpu-name", default=DEFAULT_GPU)
    p.add_argument("--test-timeout", type=int, default=1200)
    p.add_argument("--skip-optimization", action="store_true")
    args = p.parse_args(argv)
    gen_rounds = max(1, math.ceil(args.budget / 2 / args.gen_workers))
    opt_rounds = max(1, args.budget // 2 // 4)
    task, ledger = _common.start(args, AGENT, {
        "kernelagent_commit": KERNELAGENT_COMMIT, "gen_workers": args.gen_workers,
        "gen_rounds": gen_rounds, "opt_strategy": args.opt_strategy, "opt_rounds": opt_rounds,
        "gpu_name": args.gpu_name, "skip_optimization": args.skip_optimization,
    })

    root = args.kernelagent_root.resolve()
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_common.REPO / "src"), str(task.root), str(ledger.root), str(root),
         os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    register_model(args.model, args.api_base)

    problem = ledger.root / "problem.py"
    problem.write_text(render_problem(task), encoding="utf-8")
    test = render_test(task, ledger.root)
    (ledger.root / "test_adbench_tools.py").write_text(test, encoding="utf-8")
    description = problem_description(task)
    (ledger.root / "problem_description.md").write_text(description, encoding="utf-8")

    from triton_kernel_agent import TritonKernelAgent

    agent = TritonKernelAgent(num_workers=args.gen_workers, max_rounds=gen_rounds,
                              log_dir=str(ledger.root / "generation"), model_name=args.model,
                              test_timeout_s=args.test_timeout)
    generated = {}
    try:
        generated = agent.generate_kernel(problem_description=description, test_code=test,
                                          generate_default_test=False)
    except Exception:
        if not _common.budget_exhausted(args):
            raise
    finally:
        agent.cleanup()
    (ledger.root / "generation.json").write_text(
        json.dumps({k: v for k, v in generated.items() if k != "kernel_code"}, indent=2,
                   default=str), encoding="utf-8")
    if not generated.get("success"):
        if _common.budget_exhausted(args):
            return _common.finish(args, task, ledger, _common.best_evaluated(ledger))
        return _common.finish(args, task, ledger, None, status="generation_failed")
    best = ledger.root / "generated_kernel.py"
    best.write_text(generated["kernel_code"], encoding="utf-8")

    if not args.skip_optimization:
        from triton_kernel_agent.opt_manager import OptimizationManager

        config = root / "examples" / "configs" / f"{args.opt_strategy}.yaml"
        manager = OptimizationManager(
            config=str(config), max_rounds=opt_rounds, log_dir=ledger.root / "optimization",
            database_path=ledger.root / "optimization" / "program_db.json",
            openai_model=args.model, gpu_name=args.gpu_name,
        )
        try:
            optimized = manager.run_optimization(initial_kernel=generated["kernel_code"],
                                                 problem_file=problem, test_code=[test])
        except Exception:
            if not _common.budget_exhausted(args):
                raise
            optimized = {"success": False, "stop_reason": "api_budget"}
        (ledger.root / "optimization.json").write_text(
            json.dumps({k: v for k, v in optimized.items() if k != "kernel_code"}, indent=2,
                       default=str), encoding="utf-8")
        if optimized.get("success") and optimized.get("kernel_code"):
            best = ledger.root / "optimized_kernel.py"
            best.write_text(optimized["kernel_code"], encoding="utf-8")
    return _common.finish(args, task, ledger, best)


if __name__ == "__main__":
    raise SystemExit(main())
