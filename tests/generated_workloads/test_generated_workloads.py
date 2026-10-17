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

    with pytest.raises(NotImplementedError, match="reserved"):
        spec.replace(input_mode="real")

    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
    output = embedding_bag(weight, torch.tensor([2, 2, 4, 1, 2, 1]),
                           torch.tensor([0, 3]))
    output.sum().backward()
    assert torch.equal(weight.grad[:, 0], torch.tensor([0., 2., 3., 0., 1.]))


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
