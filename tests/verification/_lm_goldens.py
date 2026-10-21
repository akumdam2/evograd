"""Golden fingerprints of the language-model verification paths, on tiny CPU models.

The gates are being generalized beyond language models. Every existing Qwen3,
Qwen3.5 and Llama path must stay bit-for-bit what it was, because corpora,
Gate 2 policies and Gate 3 control runs are bound to it. This module computes
what those paths produce on tiny random models:

* the payload records and shard digests of a tiny Qwen3 and Qwen3.5 capture
  (the corpus' own ``corpus_sha256`` carries a timestamp, so it is not used);
* ``WorkloadModelScope`` on tiny Qwen3 and Llama: per-sample results digest,
  sample fingerprint, result contract digest, and the calibrated thresholds;
* a 6-step Gate 3 measurement of tiny Llama: every objective and the run
  identity (without the environment fingerprint, recorded separately).

``python -m tests.verification._lm_goldens`` rewrites ``_golden_lm.json``;
``test_lm_invariance`` recomputes and compares.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import tempfile
import warnings
from pathlib import Path

import torch

GOLDEN = Path(__file__).with_name("_golden_lm.json")
FAST = "aot_eager"
ARCH = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "head_dim": 16, "vocab_size": 128,
        "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1}


def environment() -> dict:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "machine": platform.machine()}


def _payload_digest(writer) -> dict:
    from evograd.verification.common.contracts import stable_hash

    shards = []
    for path in sorted((writer.root / "shards").glob("shard_*.bin")):
        from evograd.verification.integrations.base import sha_file

        shards.append(sha_file(path))
    return {"payloads": len(writer.payloads), "records": stable_hash(writer.payloads),
            "storages": stable_hash(writer.storages), "shards": shards}


def qwen3_capture() -> dict:
    from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
    from evograd.verification.integrations import base as Q
    from evograd.verification.integrations.qwen3_0_6b import capture as C
    from evograd.verification.payloads import CorpusWriter
    from tests.verification import test_integration_qwen3 as T

    workload = Qwen3Workload(device="cpu", dtype="bfloat16", batch_size=T.B, seq_len=T.T,
                             arch_overrides=T.ARCH)

    def build():
        model = Q.build_model(workload)
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for name, p in sorted(model.named_parameters()):
                if name.endswith(("norm.weight", "layernorm.weight")):
                    p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
        return model

    cases = T.tiny_cases()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        writer = CorpusWriter(Path(tmp))
        for order, seed in ((101, 7), (303, 8)):
            ids = torch.randint(0, T.V, (T.B, T.T), generator=torch.Generator().manual_seed(seed))
            model = build()
            recorder = C.LiveCapture(model).install()
            try:
                output = Q.run_step(model, ids, ids.clone())
            finally:
                recorder.remove()
            out[f"step_{order}"] = {k: v for k, v in Q.step_record(model, output).items()
                                    if k != "loss"}
            Q.record_step(writer, C.invocations(recorder, model, cases),
                          {"order": order, "sha256": str(order)})
        writer.finish(identity={"name": "golden"}, splits=Q.splits(writer.payloads),
                      coverage=Q.coverage(cases, writer.payloads))
        out["corpus"] = _payload_digest(writer)
        out["state"] = Q.parameter_state(build())
    return out


def qwen3_5_capture() -> dict:
    from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload
    from evograd.verification.integrations import base as Q
    from evograd.verification.integrations.qwen3_5_4b import capture as C
    from evograd.verification.payloads import CorpusWriter
    from tests.qwen3_5_4b._tiny import TINY_OVERRIDES
    from tests.verification import test_integration_qwen3_5 as T

    workload = Qwen3_5Workload(device="cpu", dtype="bfloat16", batch_size=1, seq_len=T.T,
                               arch_overrides={**with_layers(QWEN3_5_4B, 8), **TINY_OVERRIDES})

    def build():
        model = Q.build_model(workload)
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for name, p in sorted(model.named_parameters()):
                if name.endswith("norm.weight"):
                    p.copy_(0.1 * torch.randn(p.shape, generator=g))
        return model

    cases = T.tiny_cases()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        writer = CorpusWriter(Path(tmp))
        for order, seed in ((101, 7), (303, 8)):
            ids = torch.randint(0, T.V, (1, T.T), generator=torch.Generator().manual_seed(seed))
            model = build()
            recorder = C.LiveCapture(model).install()
            try:
                output = Q.run_step(model, ids, ids.clone())
            finally:
                recorder.remove()
            out[f"step_{order}"] = {k: v for k, v in Q.step_record(model, output).items()
                                    if k != "loss"}
            Q.record_step(writer, C.invocations(recorder, model, cases),
                          {"order": order, "sha256": str(order)})
        writer.finish(identity={"name": "golden"}, splits=Q.splits(writer.payloads),
                      coverage=Q.coverage(cases, writer.payloads))
        out["corpus"] = _payload_digest(writer)
    return out


def _model_scope(workload, site: str) -> dict:
    from evograd.verification.common.execution import Deployment, Sample
    from evograd.verification.gate2.adapters.model import WorkloadModelScope
    from evograd.verification.gate2.calibration import calibrate
    from evograd.verification.gate2.runner import execute, integration_contract

    scope = WorkloadModelScope(workload)
    sample = Sample("golden/seed1", 1)
    ex = execute(scope, Deployment.native(), sample)
    if ex.error:
        raise RuntimeError(ex.error)
    cal = calibrate(scope, patch_set=(site,), samples=[Sample("golden/seed1", 1),
                                                       Sample("golden/seed2", 2)],
                    compile_backend=FAST)
    if cal.policy is None:
        raise RuntimeError([f.to_dict() for f in cal.findings])
    policy = cal.policy.to_dict()
    return {"results_digest": ex.results_digest,
            "sample_fingerprint": ex.sample_identity["fingerprint"],
            "contract_digest": integration_contract(ex).digest(),
            "identity": json.loads(json.dumps(scope.identity(), default=str)),
            "thresholds": policy.get("thresholds_by_sample") or policy.get("thresholds"),
            "control_measurements": cal.evidence["control_measurements"]}


def model_scopes() -> dict:
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload
    from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

    llama = Llama3Workload.from_config({
        "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
        "arch_overrides": {**ARCH, "num_key_value_heads": 1}})
    qwen3 = Qwen3Workload(device="cpu", dtype="float32", batch_size=1, seq_len=16,
                          arch_overrides={**ARCH, "num_key_value_heads": 2})
    return {"llama_3_2_1b": _model_scope(llama, "swiglu_mlp"),
            "qwen3_0_6b": _model_scope(qwen3, "swiglu_mlp")}


def gate3_llama() -> dict:
    from evograd.evaluation.tier3.workloads.llama3_2_1b.training import SyntheticBatches
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload
    from evograd.verification import gate3
    from evograd.verification.common.contracts import stable_hash
    from evograd.verification.common.execution import Deployment

    workload = Llama3Workload.from_config({
        "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
        "arch_overrides": {**ARCH, "num_key_value_heads": 1}})
    adapter = gate3.workload_training(
        workload, train_data=SyntheticBatches(workload, offset=1000, count=6, role="train"),
        train_count=6, optimizer=gate3.adamw(lr=1e-3))
    plan = gate3.TrainingPlan(steps=6, window=1, checkpoints=(), validation=False)
    run = gate3.measure_training(adapter, Deployment.native(), plan, seed=0,
                                 protocol_context={"golden": True})
    if not run.completed:
        raise RuntimeError([f.to_dict() for f in run.findings])
    identity = {k: v for k, v in run.to_dict()["identity"].items() if k != "environment"}
    return {"objectives": [s["objective"] for s in run.train_steps],
            "identity_hash": stable_hash(identity), "identity": identity}


def compute() -> dict:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return {"environment": environment(), "qwen3_capture": qwen3_capture(),
            "qwen3_5_capture": qwen3_5_capture(), "model_scope": model_scopes(),
            "gate3_llama": gate3_llama()}


def main() -> int:
    data = compute()
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
