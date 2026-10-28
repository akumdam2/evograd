"""Every observed case's scalar arguments, as its task pack states and generates
them, against what the model's recorded training calls pass.

A pack's scalar (a norm's eps, a convolution's stride) reaches an agent as a
signature default and an argument line, and reaches every generated-input
evaluation and evograd's unified seed through the declaration's input
generator. Both must be what the model passes, or a candidate that is correct
on the pack is wrong on the real calls. Only the corpus manifests are read
(no tensor shard): set ``EVOGRAD_PAYLOAD_CORPORA`` to a directory holding
``<model>/<corpus>/manifest.json``; skipped otherwise.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from evograd.verification.integrations import PACKAGES  # noqa: E402

ROOT = os.environ.get("EVOGRAD_PAYLOAD_CORPORA")
pytestmark = pytest.mark.skipif(not ROOT, reason="EVOGRAD_PAYLOAD_CORPORA is not set")


def _recorded(model: str) -> tuple[set[str], dict[str, dict[str, set]]]:
    """Case ids with recorded calls, and each case's recorded scalar values."""
    cases, scalars = set(), {}
    for manifest in sorted(Path(ROOT, model).glob("*/manifest.json")):
        for payload in json.loads(manifest.read_text())["payloads"]:
            cases.add(payload["case_id"])
            for name, value in payload["args"].items():
                if isinstance(value, dict) and "scalar" in value:
                    scalars.setdefault(payload["case_id"], {}).setdefault(name, set()).add(
                        value["scalar"])
    return cases, scalars


@pytest.mark.parametrize("model", sorted(PACKAGES))
def test_pack_scalars_are_what_the_recorded_calls_pass(tmp_path, model):
    from evograd.adbench_tools.task import _write_model_declaration, describe_abi
    from evograd.benchmark import load_task
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import observed_cases

    cases, recorded = _recorded(model)
    if not cases:
        pytest.skip(f"no corpus for {model} under {ROOT}")
    mismatches, checked = [], 0
    for index, case in enumerate(observed_cases(load(model))):
        if case.case_id not in cases:
            continue
        path = tmp_path / str(index) / "declaration.py"
        path.parent.mkdir()
        _write_model_declaration(path, model, case.task, case.workload)
        op = load_task(f"{path}:op")
        workload = op.correctness[0]
        generated = make_case_inputs(op, workload, device="meta")
        abi = {e["name"]: e["default"] for e in describe_abi(op, workload)["arguments"]
               if e["kind"] == "scalar"}
        truth = recorded.get(case.case_id, {})
        names = {a.name for a in op.scalar_inactive_args()}
        if names != set(truth):
            mismatches.append((case.case_id, "declared", sorted(names), "recorded", sorted(truth)))
            continue
        for arg in op.scalar_inactive_args():
            checked += 1
            values = truth[arg.name]
            views = {"default": arg.default, "generated": generated.get(arg.name, arg.default),
                     "abi": abi[arg.name]}
            if len(values) != 1 or any(v != next(iter(values)) for v in views.values()):
                mismatches.append((case.case_id, arg.name, sorted(values), views))
    assert not mismatches, mismatches
    if not checked:
        pytest.skip(f"{model}: no recorded case takes a scalar argument")
