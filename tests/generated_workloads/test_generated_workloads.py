"""Reduced execution checks; these cases are never published as timing suites."""

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.common.generated.harvest import collect, snapshot
from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload


SMALL = dict(input_size=8, hidden_size=24, depth=2, num_heads=4,
             num_classes=10, diffusion_steps=100)


def test_generated_harvest():
    spec = GeneratedSpec(model="dit_s_2", batch_size=2, device="cpu", arch_overrides=SMALL)
    manifest = collect(spec)
    snap = snapshot(manifest)
    assert manifest["backward_verified"]
    depth = spec.arch["depth"]
    for task in ("dit_mlp", "dit_modulation", "dit_attention"):
        assert snap["tasks"]["dit_s_2_" + task]["frequency"] == depth
    # Two residual adds per block; the last feeds the final layer (its own configuration).
    residual = snap["tasks"]["dit_s_2_dit_adaln_residual"]
    assert [(c["provenance"]["component"], c["frequency"]) for c in residual["configurations"]] == [
        ("blocks.residual", 2 * depth - 1), ("final_residual", 1)]
    assert snap["tasks"]["dit_s_2_dit_attention"]["configurations"][0]["provenance"]["free"] == {
        "heads": spec.arch["num_heads"]}
    assert not spec.is_canonical
    # The tracked suite is the published run's, never this reduced one.
    assert snap["tasks"]["dit_s_2_dit_mlp"]["configurations"][0]["dims"] not in [
        dict(w.dims) for w in TASKS["dit_s_2_dit_mlp"].benchmark]


def test_canonical_architecture_size():
    from evograd.benchmark.topdown.common.generated.model import DiTModel, DLRMModel

    dit = DiTModel(canonical("dit_s_2").arch)
    assert 32_000_000 < sum(p.numel() for p in dit.parameters()) < 34_000_000
    dlrm = DLRMModel(canonical("torchrec_dlrm").arch)
    assert sum(p.numel() for p in dlrm.parameters()) > 160_000_000


def test_dlrm_generated_harvest():
    from evograd.ops.level1.embedding_bag.reference import forward as embedding_bag
    import pytest
    import torch

    small = dict(num_sparse_features=3, embedding_dim=8,
                 num_embeddings_per_feature=[40] * 3,
                 dense_arch_layer_sizes=[16, 8],
                 over_arch_layer_sizes=[16, 8, 1],
                 multi_hot_sizes=[1, 2, 1])
    spec = GeneratedSpec(model="torchrec_dlrm", batch_size=2, device="cpu",
                         arch_overrides=small)
    manifest = collect(spec)
    snap = snapshot(manifest)
    assert manifest["backward_verified"]
    assert snap["tasks"]["torchrec_dlrm_dlrm_interaction"]["frequency"] == 1
    assert snap["level1"]["embedding_bag"]["frequency"] == 3
    assert snap["tasks"]["torchrec_dlrm_dlrm_interaction"]["configurations"][0]["dims"] not in [
        dict(w.dims) for w in TASKS["torchrec_dlrm_dlrm_interaction"].benchmark]

    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
    output = embedding_bag(weight, torch.tensor([2, 2, 4, 1, 2, 1]),
                           torch.tensor([0, 3]))
    output.sum().backward()
    assert torch.equal(weight.grad[:, 0], torch.tensor([0., 2., 3., 0., 1.]))


DLRM_SMALL = dict(num_sparse_features=3, embedding_dim=8, num_embeddings_per_feature=[40] * 3,
                  dense_arch_layer_sizes=[16, 8], over_arch_layer_sizes=[16, 8, 1],
                  multi_hot_sizes=[1, 2, 1])


def test_real_run_observes_the_verification_step_and_restores_the_model(tmp_path):
    """The harvest observes ``base.real_run``'s step, records its identity as
    ``run`` and leaves the model's site functions as it found them."""
    from unittest import mock
    from evograd.benchmark.topdown.common.generated import harvest
    from evograd.benchmark.topdown.common.generated import model as dit
    from evograd.benchmark.topdown.common.generated.model import make_inputs
    from evograd.verification.integrations import base, load

    for name, small, real_id in (("dit_s_2", SMALL, "dit_s_2-f06efa7f62b01cf5"),
                                 ("torchrec_dlrm", DLRM_SMALL, "torchrec_dlrm-1394f3cc47559de8")):
        spec = GeneratedSpec(model=name, batch_size=2, device="cpu", arch_overrides=small)
        workload = GeneratedWorkload(spec)
        model = base.build_model(workload)
        run = {"model": name, "workload": {"model": name}, "data": {}, "order": 101,
               "batch": {"batch_sha256": "b"}, "parameter_state": {"state_sha256": "s"},
               "execution": {}}
        with mock.patch("evograd.verification.integrations.base.real_run",
                        return_value=(workload, model, make_inputs(spec, 0), run)) as real_run:
            manifest = harvest.real_run(name)
        assert real_run.call_args.args == (load(name),)
        assert manifest["run"] == run
        assert snapshot(manifest)["run"] == manifest["run"]
        assert snapshot(manifest)["schema_version"] == f"evograd-{name}-snapshot/2"
        if name == "dit_s_2":  # sites put back
            assert all((b.mod_fn, b.attn_fn, b.residual_fn, b.mlp_fn) == (
                dit.dit_modulation, dit.dit_attention, dit.dit_adaln_residual, dit.dit_mlp)
                for b in model.blocks)
        real = load(name).load_workload().spec
        assert real.input_mode == "real" and real.workload_id == real_id


def test_dlrm_harvest_records_the_run_its_capture_repeats(tmp_path, monkeypatch):
    """Unmocked on CPU over a small Criteo fixture: the harvest's ``run`` is the
    identity a capture of the same model computes, and the publish rule takes it."""
    import json
    import numpy as np
    import pytest
    from unittest import mock
    from evograd.benchmark.topdown.common.generated import harvest
    from evograd.verification.integrations import base
    from evograd.verification.integrations.generated_model import make_model
    from tests.verification._toy_integration import configured

    rows = 64
    np.save(tmp_path / "train_dense.npy", np.arange(rows * 13, dtype=np.float32).reshape(rows, 13) / 10)
    np.save(tmp_path / "train_sparse.npy", np.arange(rows * 26, dtype=np.int64).reshape(rows, 26) + 1000)
    np.save(tmp_path / "train_labels.npy", (np.arange(rows, dtype=np.int64) % 2).reshape(rows, 1))
    monkeypatch.setenv("EVOGRAD_DLRM_CRITEO_DIR", str(tmp_path))
    model = make_model("torchrec_dlrm", {
        "model": "torchrec_dlrm", "device": "cpu", "input_mode": "real", "stream": True,
        "batch_size": 4, "arch_overrides": {**DLRM_SMALL, "num_sparse_features": 26,
                                            "num_embeddings_per_feature": [50] * 26,
                                            "multi_hot_sizes": [1] * 26}})
    with configured(), mock.patch("evograd.verification.integrations.load", return_value=model):
        manifest = json.loads(json.dumps(harvest.real_run("torchrec_dlrm", device="cpu")))
        workload = model.load_workload()
        _, batches = model.train_batches(workload, base.ORDERS)
        state = base.objective_of(model).parameter_state(base.build_model(workload))
    capture = base.run_identity(model, state, batches)
    published = harvest.snapshot(manifest)
    assert published["run"] == manifest["run"]
    base.require_real_run(model, manifest)
    with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: published):
        base.require_harvested_run(model, capture)
        other = {**capture, "batch": batches["batches"]["202"]}
        with pytest.raises(RuntimeError, match="another run"):
            base.require_harvested_run(model, other)


def test_dlrm_criteo_real_input_artifact(tmp_path, monkeypatch):
    """A small TorchRec-format fixture exercises mapping and replay."""
    import numpy as np
    import pytest
    import torch
    from types import SimpleNamespace
    from evograd.benchmark.topdown.common.generated.data import ensure_pinned_data, load_data

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVOGRAD_DLRM_CRITEO_DIR", raising=False)
    source = tmp_path / "data" / "torchrec_dlrm" / "processed"
    source.mkdir(parents=True)
    dense = np.arange(20 * 13, dtype=np.float32).reshape(20, 13) / 10
    sparse = (np.arange(20 * 26, dtype=np.int64).reshape(20, 26) + 1000)
    labels = (np.arange(20, dtype=np.int32) % 2).reshape(20, 1)
    np.save(source / "train_dense.npy", dense)
    np.save(source / "train_sparse.npy", sparse)
    np.save(source / "train_labels.npy", labels)

    small = dict(embedding_dim=8, num_embeddings_per_feature=[32] * 26,
                 dense_arch_layer_sizes=[16, 8], over_arch_layer_sizes=[16, 8, 1])
    spec = GeneratedSpec(model="torchrec_dlrm", batch_size=2, device="cpu",
                         input_mode="real", arch_overrides=small)
    path = tmp_path / "prepared.pt"
    ensure_pinned_data(spec, path, batch_count=4)
    data, checksum = load_data(SimpleNamespace(spec=spec, data_path=str(path)))
    assert len(checksum) == 64
    assert len(data["batches"]) == 4
    assert len(set(data["selection"])) == 8
    assert all(row < 10 for row in data["selection"])
    first = data["batches"][0]
    selected = data["selection"][:2]
    assert torch.equal(first["dense"], torch.from_numpy(dense[selected]))
    assert torch.equal(first["sparse_ids"][:, :, 0],
                       torch.from_numpy((sparse[selected] % 32).T.copy()))
    assert first["labels"].tolist() == labels[selected, 0].tolist()
    assert ensure_pinned_data(spec, path, batch_count=4) == str(path)
    assert ensure_pinned_data(spec, path, source_dir=source, batch_count=4) == str(path)
    assert collect(spec, batch={k: v.to(spec.device) for k, v in first.items()})["backward_verified"]

    workload = GeneratedWorkload(spec, data_path=str(path))
    assert workload.describe()["data"] == "pinned_real_data"

    changed_source = np.load(source / "train_sparse.npy", mmap_mode="r+")
    changed_source[selected[0], 0] += 1
    changed_source.flush()
    with pytest.raises(ValueError, match="differs from the supplied Criteo source"):
        ensure_pinned_data(spec, path, source_dir=source, batch_count=4)

    data["batches"][0]["sparse_ids"][0, 0, 0] = 32
    torch.save(data, path)
    with pytest.raises(ValueError, match="specialist contract"):
        load_data(SimpleNamespace(spec=spec, data_path=str(path)))
    with pytest.raises(ValueError, match="one-hot"):
        spec.replace(arch_overrides={**small, "multi_hot_sizes": [2] * 26})


def test_real_input_artifact_contract(tmp_path):
    import torch
    from evograd.benchmark.topdown.common.generated.data import (
        CLASS_IDS, DATASET, SCHEMA, VAE, load_data,
    )

    spec = GeneratedSpec(model="dit_s_2", batch_size=2, device="cpu", input_mode="real")
    shape = (2, 4, 32, 32)
    batch = {"latent": torch.zeros(shape), "noise": torch.ones(shape),
             "timesteps": torch.tensor([0, 999]),
             "labels": torch.tensor([CLASS_IDS[0], CLASS_IDS[1]]),
             "drop_mask": torch.tensor([0, 1])}
    artifact = {"schema_version": SCHEMA, "model": spec.model,
                "workload_id": spec.workload_id,
                "dataset_repo": DATASET[0], "dataset_revision": DATASET[1],
                "dataset_file": DATASET[2], "vae_repo": VAE[0],
                "vae_revision": VAE[1], "vae_scale": 0.18215,
                "examples": 2,
                "selection": [{"row": 0, "label": 0, "flipped": False, "image_sha256": "a" * 64},
                              {"row": 1, "label": 1, "flipped": True, "image_sha256": "b" * 64}],
                "batches": [batch]}
    path = tmp_path / "prepared.pt"
    torch.save(artifact, path)
    workload = GeneratedWorkload(spec, data_path=str(path))
    loaded, checksum = load_data(workload)
    assert loaded["batches"][0]["labels"].tolist() == [CLASS_IDS[0], CLASS_IDS[1]]
    assert len(checksum) == 64
    assert workload.describe()["data"] == "pinned_real_data"
    assert workload.batch_for(seed=1)["latent"].shape == shape
    assert spec.workload_hash != spec.replace(input_mode="generated").workload_hash
    artifact["vae_revision"] = "wrong"
    torch.save(artifact, path)
    import pytest
    with pytest.raises(ValueError, match="pinned real-input contract"):
        load_data(workload)
