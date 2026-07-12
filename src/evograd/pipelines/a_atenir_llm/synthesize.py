"""Pipeline A: AtenIR-grounded LLM autograd-pair synthesis.

forward reference -> AtenIR backward graph summary -> LLM plan -> LLM codegen
-> oracle verification (retry with repair prompts).

Everything operator-specific comes from the :class:`OpDecl`: the forward
reference path, the extraction example input, the prompt contract, and the
verification oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evograd.opdecl.activity import OpDecl
from evograd.opdecl.compat import to_operator_spec
from evograd.pipelines.a_atenir_llm.prompts import (
    SYSTEM_MESSAGE,
    render_codegen_prompt,
    render_plan_prompt,
    render_repair_prompt,
)
from evograd.pipelines.shared.graph_summary import summarize_backward_graph_file
from evograd.pipelines.shared.llm_client import generate_with_openai_compatible_api
from evograd.pipelines.shared.runner import (
    extract_graph,
    report_passed,
    strip_code_fence,
    verify_candidate,
)


@dataclass(frozen=True)
class AutogradPairConfig:
    op: OpDecl
    forward: str
    example_input: str
    output_dir: Path
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    python: str
    lowering_context: str | None = None
    dry_run: bool = False
    # Skip oracle verification (accept the first attempt). Only useful on
    # machines without CUDA; verified runs are the default.
    skip_verify: bool = False


def _verify(config: AutogradPairConfig, program_path: Path) -> dict:
    if config.skip_verify:
        return {"metrics": {"correct": 1.0}, "verification": "skipped"}
    return verify_candidate(
        python=config.python,
        op_name=config.op.name,
        program_path=program_path,
        log_dir=program_path.parent,
    )


def synthesize_autograd_pair(config: AutogradPairConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    spec = to_operator_spec(config.op)

    print("Extract: AtenIR reference backward graph")
    graph_path = extract_graph(
        python=config.python,
        forward=config.forward,
        example_input=config.example_input,
        out_path=config.output_dir / "atenir_graph.json",
        log_dir=config.output_dir,
    )
    graph_summary = summarize_backward_graph_file(graph_path)
    (config.output_dir / "graph_summary.md").write_text(graph_summary, encoding="utf-8")

    lowering_context = config.lowering_context or ""
    if lowering_context:
        (config.output_dir / "lowering_context.md").write_text(lowering_context, encoding="utf-8")

    plan_prompt = render_plan_prompt(
        forward=config.forward,
        graph_summary=graph_summary,
        lowering_context=lowering_context,
        spec=spec,
    )
    (config.output_dir / "autograd_pair_plan_prompt.md").write_text(plan_prompt, encoding="utf-8")

    if config.dry_run:
        dry_dir = config.output_dir / "attempt_001"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "codegen_prompt.md").write_text(
            render_codegen_prompt(
                graph_summary=graph_summary,
                pair_plan="{AUTOGRAD_PAIR_PLAN_FROM_LLM}",
                lowering_context=lowering_context,
                spec=spec,
            ),
            encoding="utf-8",
        )
        print(f"dry-run wrote {config.output_dir}")
        return 0

    print("Autograd-pair: plan synthesis")
    pair_plan = generate_with_openai_compatible_api(
        prompt=plan_prompt,
        system_message=SYSTEM_MESSAGE,
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    (config.output_dir / "autograd_pair_plan.md").write_text(pair_plan, encoding="utf-8")

    prompt = render_codegen_prompt(
        graph_summary=graph_summary,
        pair_plan=pair_plan,
        lowering_context=lowering_context,
        spec=spec,
    )
    previous_code = ""
    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "codegen_prompt.md").write_text(prompt, encoding="utf-8")
        print(f"Autograd-pair: codegen attempt {attempt}/{config.max_attempts}")
        response = generate_with_openai_compatible_api(
            prompt=prompt,
            system_message=SYSTEM_MESSAGE,
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )
        code = strip_code_fence(response)
        program_path = attempt_dir / "program.py"
        program_path.write_text(code, encoding="utf-8")
        (attempt_dir / "response.txt").write_text(response, encoding="utf-8")

        report = _verify(config, program_path)
        (attempt_dir / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if report_passed(report):
            best_dir = config.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "initial_program_autograd_pair.py"
            best_path.write_text(code, encoding="utf-8")
            (best_dir / "verification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Autograd-pair synthesis passed. Best program: {best_path}")
            return 0

        if config.skip_verify:
            break

        repair_prompt = render_repair_prompt(
            graph_summary=graph_summary,
            pair_plan=pair_plan,
            previous_code=code or previous_code,
            verifier_report=json.dumps(report, indent=2, sort_keys=True),
            lowering_context=lowering_context,
            spec=spec,
        )
        (attempt_dir / "repair_prompt.md").write_text(repair_prompt, encoding="utf-8")
        prompt = repair_prompt
        previous_code = code
        print(f"attempt {attempt} failed; wrote repair prompt")

    print(f"Autograd-pair synthesis failed after {config.max_attempts} attempts")
    return 1
