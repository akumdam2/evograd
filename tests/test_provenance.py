"""Results record the evograd commit and the host that produced them."""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

from evograd.provenance import provenance


def test_a_run_and_an_evaluation_record_the_commit_and_host(tmp_path):
    from evograd.adbench_tools.ledger import RunLedger
    from evograd.evaluation.protocol import _environment

    recorded = provenance()
    assert len(recorded["commit"]) == 40 and recorded["host"] == socket.gethostname()
    task = SimpleNamespace(task_id="t", root=tmp_path, spec={})
    RunLedger(tmp_path / "run", agent="a", task=task)
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["evograd"]["commit"] == recorded["commit"] == _environment()["evograd"]["commit"]
