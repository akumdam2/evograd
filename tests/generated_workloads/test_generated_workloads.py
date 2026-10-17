"""Reduced execution checks; these cases are never published as timing suites."""

import json

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.common.generated.harvest import collect, snapshot
from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from evograd.evaluation.tier3.workloads.common.generated.workload import (
    GeneratedWorkload, structural_identity_kernels,
)
from evograd.evaluation.tier3.workloads.common.vision.gate import calibrate, evaluate


SMALL = dict(input_size=8, hidden_size=24, depth=2, num_heads=4,
             num_classes=10, diffusion_steps=100)


def test_generated_harvest_and_model_gate(tmp_path):
    spec = GeneratedSpec(model="dit_s_2", batch_size=2, device="cpu", arch_overrides=SMALL)
    manifest = collect(spec)
    snap = snapshot(manifest)
    assert manifest["backward_verified"]
    assert snap["tasks"]["dit_s_2_dit_mlp"]["frequency"] == spec.arch["depth"]
    assert snap["tasks"]["dit_s_2_dit_modulation"]["frequency"] == spec.arch["depth"]
    assert not spec.is_canonical
    assert not TASKS["dit_s_2_dit_mlp"].benchmark

    path = tmp_path / "calibration.json"
    workload = GeneratedWorkload(spec, str(path))
    sites = ["dit_modulation", "dit_mlp"]
    path.write_text(json.dumps(calibrate(workload, sites, steps=2, backend="aot_eager")))
    result = evaluate(workload, structural_identity_kernels(workload.site_registry))
    assert result["ok"], result
    assert result["evaluation_complete"]
    assert result["trajectory"] == {"rms": 0.0, "max_abs": 0.0}


def test_canonical_architecture_size():
    from evograd.benchmark.topdown.common.generated.model import DiTModel, DLRMModel

    dit = DiTModel(canonical("dit_s_2").arch)
    assert 32_000_000 < sum(p.numel() for p in dit.parameters()) < 34_000_000
    dlrm = DLRMModel(canonical("torchrec_dlrm").arch)
    assert sum(p.numel() for p in dlrm.parameters()) > 160_000_000


def test_dlrm_generated_harvest_and_identity_gate(tmp_path):
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
    assert not TASKS["torchrec_dlrm_dlrm_interaction"].benchmark

    path = tmp_path / "dlrm_calibration.json"
    workload = GeneratedWorkload(spec, str(path))
    sites = ["embedding_bag", "dlrm_interaction"]
    path.write_text(json.dumps(calibrate(workload, sites, steps=2, backend="aot_eager")))
    result = evaluate(workload, structural_identity_kernels(workload.site_registry))
    assert result["ok"], result
    assert result["evaluation_complete"]
    assert result["trajectory"] == {"rms": 0.0, "max_abs": 0.0}

    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
    output = embedding_bag(weight, torch.tensor([2, 2, 4, 1, 2, 1]),
                           torch.tensor([0, 3]))
    output.sum().backward()
    assert torch.equal(weight.grad[:, 0], torch.tensor([0., 2., 3., 0., 1.]))


def test_dlrm_criteo_real_input_artifact_and_gate(tmp_path, monkeypatch):
    """A small TorchRec-format fixture exercises mapping, replay, and the gate."""
    import numpy as np
    import pytest
    import torch
    from types import SimpleNamespace
    from evograd.benchmark.topdown.common.generated.data import ensure_pinned_data, load_data

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVOGRAD_DLRM_CRITEO_DIR", raising=False)
    source = tmp_path / "data" / "criteo_kaggle"
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
    assert collect(spec, {k: v.to(spec.device) for k, v in first.items()})["backward_verified"]

    calibration = tmp_path / "real_calibration.json"
    workload = GeneratedWorkload(spec, str(calibration), str(path))
    sites = ["embedding_bag", "dlrm_interaction"]
    calibration.write_text(json.dumps(calibrate(workload, sites, steps=2, backend="aot_eager")))
    result = evaluate(workload, structural_identity_kernels(workload.site_registry))
    assert result["ok"], result
    assert result["evaluation_complete"]
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
