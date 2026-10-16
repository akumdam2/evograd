import torch
import pytest

from evograd.benchmark.topdown.common.vision.data import (
    DATASETS, IMAGENETTE_CLASS_IDS, _answer, load_data,
)
from evograd.benchmark.topdown.common.vision.model import make_inputs
from evograd.benchmark.topdown.common.vision.spec import REVISIONS, canonical
from evograd.evaluation.tier3.workloads.common.vision.workload import VisionWorkload


def test_dataset_label_and_answer_policy():
    assert IMAGENETTE_CLASS_IDS == (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
    assert _answer(["nokia", "toshiba", "nokia"]) == "nokia"
    assert _answer(["a", "b"]) == "a"
    with pytest.raises(ValueError, match="no reference answer"):
        _answer(["", " "])


@pytest.mark.parametrize("model", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_pinned_artifact_identity_and_supervision(tmp_path, model):
    spec = canonical(model).replace(device="cpu", weights="pretrained")
    batch = make_inputs(spec)
    if model == "convnextv2_tiny":
        batch["labels"][:] = IMAGENETTE_CLASS_IDS[0]
        selection = [{"row": i, "label": 0} for i in range(spec.batch_size)]
    else:
        start = int((batch["labels"][0] == -100).sum())
        selection = [{"row": 0, "answer_start": start, "answer_end": spec.seq_len}]
    repo, revision, filename = DATASETS[model]
    data = {
        "schema_version": "evograd-vision-data/2", "model": model,
        "source": f"{repo}@{revision}/{filename}:train[:{spec.batch_size}]",
        "dataset_repo": repo, "dataset_revision": revision, "dataset_file": filename,
        "processor": spec.model_name, "processor_revision": REVISIONS[model],
        "examples": spec.batch_size, "selection": selection,
        "batches": [batch],
    }
    path = tmp_path / "data.pt"
    torch.save(data, path)
    workload = VisionWorkload(spec, data_path=str(path))
    assert load_data(workload)[0]["dataset_revision"] == revision
    data["dataset_revision"] = "other"
    torch.save(data, path)
    with pytest.raises(ValueError, match="pinned dataset"):
        load_data(workload)


def test_pretrained_requires_real_batches():
    with pytest.raises(ValueError, match="requires pinned"):
        VisionWorkload(canonical("convnextv2_tiny").replace(weights="pretrained"))
