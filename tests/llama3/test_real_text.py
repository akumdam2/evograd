"""Llama's real-text identity and batch routing, without network or GPU access.

The checkpoint and WikiText are replaced by small packed splits, so these check
the wiring -- identity, flags, which stream each consumer reads -- and not the
download or the tokenizer.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evograd.benchmark.topdown.llama3_2_1b.levels.level3.spec import (
    CANONICAL, PRETRAINED, REAL_TEXT, WorkloadSpecError,
)
from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload
from evograd.evaluation.tier3.workloads.qwen3_0_6b.textdata import PackedSplit, TextBatches

SNAPSHOT = (Path(__file__).parents[2]
            / "src/evograd/benchmark/topdown/llama3_2_1b/harvest/snapshot.json")


def _split(name, first_tokens, block_len=8):
    blocks = torch.tensor([[token] * block_len for token in first_tokens])
    return PackedSplit(name, blocks, blocks.numel(), 0, len(first_tokens), name[0] * 64)


def _real_workload(monkeypatch, **overrides):
    workload = Llama3Workload(batch_size=2, seq_len=8, device="cpu",
                              weights=PRETRAINED, data=REAL_TEXT, **overrides)
    text = {"train": _split("train", [1, 2, 3, 4]),
            "validation": _split("validation", [7, 8]),
            "identity": {"dataset": "fake", "revision": "fake"}}
    monkeypatch.setattr(workload, "_text", lambda: text)
    return workload


# ── identity ─────────────────────────────────────────────────────────────────


def test_the_canonical_identity_is_unchanged():
    """Adding the two fields must not move the tracked snapshot's binding."""
    snapshot = json.loads(SNAPSHOT.read_text())
    assert CANONICAL.workload_id == snapshot["workload_id"]
    assert "weights" not in CANONICAL.to_dict() and "data" not in CANONICAL.to_dict()


def test_real_text_is_its_own_workload_identity():
    real = CANONICAL.replace(weights=PRETRAINED, data=REAL_TEXT)
    assert real.real_text and real.pretrained
    assert real.workload_id != CANONICAL.workload_id
    assert ".pretrained.realtext." in real.workload_id
    assert real.workload_hash != CANONICAL.workload_hash
    assert real.pretrained_repo == ("meta-llama/Llama-3.2-1B",
                                    "4e20de362430cd3b72f300e6b0f18e50e7166e08")


@pytest.mark.parametrize("overrides", [
    {"data": REAL_TEXT},                       # real text on random weights
    {"weights": PRETRAINED},                   # checkpoint on synthetic tokens
    {"weights": "pretrained:meta-llama/Llama-3.2-1B@main", "data": REAL_TEXT},
    {"weights": PRETRAINED, "data": "wikitext-2-raw:main"},
])
def test_only_the_two_pinned_identities_exist(overrides):
    with pytest.raises(WorkloadSpecError, match="requires the pinned checkpoint"):
        CANONICAL.replace(**overrides)


def test_the_checkpoint_exists_only_at_the_published_architecture():
    with pytest.raises(WorkloadSpecError, match="published architecture"):
        CANONICAL.replace(weights=PRETRAINED, data=REAL_TEXT, arch={"num_hidden_layers": 2})
    with pytest.raises(WorkloadSpecError):
        Llama3Workload(device="cpu", weights=PRETRAINED, data=REAL_TEXT,
                       arch_overrides={"num_hidden_layers": 2})


def test_the_config_round_trips_through_a_child_process_boundary():
    workload = Llama3Workload(device="cpu", weights=PRETRAINED, data=REAL_TEXT, data_seed=3)
    rebuilt = Llama3Workload.from_config(json.loads(json.dumps(workload.to_config())))
    assert rebuilt.spec == workload.spec
    assert rebuilt.spec.workload_id == workload.spec.workload_id


# ── flags ────────────────────────────────────────────────────────────────────


def _cli_args(**overrides):
    args = dict(device="cpu", seed=0, data_seed=0, dtype=None, batch=None, tokens=None,
                layers=None, calibration=None, real_text=False)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_the_tier3_adapter_selects_real_text_only_when_asked():
    from evograd.evaluation.tier3.workloads.llama3_2_1b.adapter import ADAPTER, build

    assert "real_text" in ADAPTER.options
    assert build(_cli_args()).spec == CANONICAL.replace(device="cpu")
    assert build(_cli_args(real_text=True)).spec.real_text


def test_calibrate_passes_real_text_to_its_workload():
    from evograd.evaluation.tier3.workloads.llama3_2_1b import calibrate

    args = calibrate.build_parser().parse_args(["run", "--device", "cpu", "--real-text"])
    assert calibrate._workload(args).spec.real_text
    args = calibrate.build_parser().parse_args(["run", "--device", "cpu"])
    assert not calibrate._workload(args).spec.real_text


def test_controls_refuse_a_calibration_from_the_other_identity(monkeypatch):
    from evograd.evaluation.tier3.workloads.llama3_2_1b import controls

    args = controls.build_parser().parse_args(["--device", "cpu", "--real-text"])
    config = controls._workload_config(args)
    real_id = Llama3Workload.from_config(config).spec.workload_id
    monkeypatch.setattr(controls, "load_policy",
                        lambda path: SimpleNamespace(workload_id=CANONICAL.workload_id))
    with pytest.raises(SystemExit, match="--real-text"):
        controls._bound_policy(args, config)
    monkeypatch.setattr(controls, "load_policy",
                        lambda path: SimpleNamespace(workload_id=real_id))
    assert controls._bound_policy(args, config).workload_id == real_id


# ── which stream each consumer reads ─────────────────────────────────────────


def test_single_step_batches_come_from_the_training_split(monkeypatch):
    workload = _real_workload(monkeypatch)
    ids, labels = workload.batch_for(seed=0)
    assert torch.equal(ids, labels)
    assert ids.shape == (2, 8)
    assert set(ids[:, 0].tolist()) <= {1, 2, 3, 4}
    # The first batch of data order data_seed + seed, as a training run's step 1.
    expected, _ = TextBatches(workload._text()["train"], batch_size=2, seed=0,
                              device="cpu").batch(0)
    assert torch.equal(ids, expected)


def test_validation_is_the_held_out_split_in_a_fixed_order(monkeypatch):
    workload = _real_workload(monkeypatch)
    first, _ = workload.validation_batches().batch(0)
    again, _ = workload.validation_batches().batch(0)
    assert set(first[:, 0].tolist()) == {7, 8}
    assert torch.equal(first, again)


def test_the_training_loop_reads_real_text(monkeypatch):
    from evograd.evaluation.tier3.workloads.llama3_2_1b import training

    workload = _real_workload(monkeypatch, data_seed=5)
    train = training.train_batches(workload, steps=10)
    validation = training.validation_batches(workload)
    assert isinstance(train, TextBatches) and train.seed == 5
    assert set(validation.batch(0)[0][:, 0].tolist()) == {7, 8}
    synthetic = training.train_batches(Llama3Workload(device="cpu"), steps=10)
    assert isinstance(synthetic, training.SyntheticBatches)


def test_the_input_checksum_follows_the_real_token_stream(monkeypatch):
    workload = _real_workload(monkeypatch)
    assert workload.input_checksum(seed=0) == workload.input_checksum(seed=0)
    assert workload.input_checksum(seed=0).endswith(":(2, 8)")
    seeds = {workload.input_checksum(seed=s) for s in range(6)}
    assert len(seeds) > 1


def test_a_synthetic_workload_has_no_text():
    with pytest.raises(ValueError, match="synthetic data"):
        Llama3Workload(device="cpu")._text()
