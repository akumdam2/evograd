import copy
import json
import pytest
import torch
from evograd.benchmark.topdown.common.vision.spec import canonical
from evograd.evaluation.tier3.workloads.common.vision.workload import (
    VisionWorkload,
    structural_identity_kernels,
)
from evograd.benchmark.topdown.common.vision.harvest import collect, snapshot
from evograd.evaluation.tier3.patch import KernelSet


def small_spec(model):
    if model == "convnextv2_tiny":
        overrides = {"depths": [1, 1, 1, 1], "hidden_sizes": [8, 16, 24, 32], "num_labels": 7}
        image, tokens = 32, 0
    else:
        overrides = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "vision_start_token_id": 120,
            "vision_end_token_id": 121,
            "image_token_id": 122,
            "video_token_id": 123,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
            "vision_config": {"depth": 1, "embed_dim": 32, "num_heads": 4, "hidden_size": 32},
        }
        image, tokens = 28, 8
    return canonical(model).replace(
        device="cpu",
        dtype="float32",
        batch_size=1,
        image_size=image,
        seq_len=tokens,
        arch_overrides=overrides,
    )


@pytest.mark.parametrize("name", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_full_backward_and_structural_identity(name):
    spec = small_spec(name)
    workload = VisionWorkload(spec)
    assert VisionWorkload.from_config(json.loads(json.dumps(workload.to_config()))).spec == spec
    assert not spec.is_canonical
    eager = workload.build(KernelSet(registry=workload.site_registry))
    patched, provenance = workload.build_patched(
        structural_identity_kernels(workload.site_registry)
    )
    assert set(provenance.actual_sites) == set(workload.site_registry.names)
    before = dict(eager.named_parameters())
    after = dict(patched.named_parameters())
    assert before.keys() == after.keys()
    for name, p in before.items():
        torch.testing.assert_close(p, after[name], rtol=0, atol=0)
    batch = workload.batch_for(seed=9)
    a = workload.loss(eager, batch)
    b = workload.loss(patched, batch)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for name, p in before.items():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, after[name].grad, rtol=1e-5, atol=1e-7)
    if spec.model == "qwen2_vl_2b":
        assert eager.model.language_model.embed_tokens.weight is eager.lm_head.weight
        assert patched.model.language_model.embed_tokens.weight is patched.lm_head.weight
        assert any("visual" in n and p.grad.abs().sum() > 0 for n, p in before.items())
        assert any("merger" in n and p.grad.abs().sum() > 0 for n, p in before.items())


@pytest.mark.parametrize("name", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_harvest_is_executed_and_deduplicated(name):
    spec = small_spec(name)
    manifest = collect(spec)
    result = snapshot(manifest)
    assert manifest["backward_verified"]
    assert result["schema_version"] == f"evograd-{name}-snapshot/2" and result["run"] is None
    assert sum(e["frequency"] for e in result["tasks"].values()) == (
        4 if name == "convnextv2_tiny" else 2
    )
    assert result["level1"]["linear"]["frequency"] > 0
    assert result["not_bound"]


@pytest.mark.parametrize("padded", [False, True])
def test_qwen2_vl_harvest_sees_the_batch_padding_mask(padded):
    """Right-padded rows (the real TextVQA pool) give the text SDPA an explicit mask."""
    from evograd.benchmark.topdown.common.vision.model import make_inputs

    spec = small_spec("qwen2_vl_2b")
    batch = make_inputs(spec)
    if padded:
        batch["attention_mask"][:, -2:] = 0
        batch["labels"][:, -2:] = -100
    manifest = collect(spec, batch=batch)
    mask = {"shape": [1, 1, 8, 8], "stride": [64, 64, 8, 1], "dtype": "torch.bool",
            "requires_grad": False} if padded else None
    assert manifest["text_attention_masks"] == [{"mask": mask, "calls": 1}]
    assert snapshot(manifest)["text_attention_masks"] == manifest["text_attention_masks"]


def test_real_run_records_its_identity_and_only_the_verification_run_publishes():
    from unittest import mock
    from evograd.benchmark.topdown.common.vision import harvest
    from evograd.benchmark.topdown.common.vision.model import build_model, make_inputs
    from evograd.verification.integrations import load

    spec = small_spec("convnextv2_tiny")
    run = {"data": {"data": {"pool_sha256": "p"}}, "order": 101, "batch_digest": "b",
           "parameter_state_sha256": "s"}
    step = (VisionWorkload(spec), build_model(spec), make_inputs(spec), run, {"threads": 8})
    with mock.patch("evograd.verification.integrations.harvest_run.first_step",
                    return_value=step) as first_step:
        manifest = harvest.real_run("convnextv2_tiny")
    assert first_step.call_args.args == (load("convnextv2_tiny"),)
    assert manifest["run"] == {
        "weights": "pretrained:facebook/convnextv2-tiny-1k-224@f4db009e63145e02b3c075aa64d90ce41bcca4b1",
        **run}
    assert manifest["determinism"] == {"threads": 8}
    assert snapshot(manifest)["run"] == manifest["run"]
    with pytest.raises(ValueError, match="verification run"):
        harvest.require_real_run(manifest, "convnextv2_tiny")  # a reduced spec never publishes
    real = load("convnextv2_tiny").load_workload().spec
    assert real.weights == "pretrained" and real.workload_id == "convnextv2_tiny-023afe318b1f76e4"
    published = json.loads(json.dumps({**manifest, "config": real.to_config(),
                                       "workload_id": real.workload_id, "architecture": real.arch}))
    harvest.require_real_run(published, "convnextv2_tiny")
    with pytest.raises(ValueError, match="verification run"):
        harvest.require_real_run({**published, "run": None}, "convnextv2_tiny")


def test_configuration_and_inputs():
    from evograd.benchmark.topdown.common.vision.model import make_inputs

    s = canonical("qwen2_vl_2b").replace(device="cpu")
    batch = make_inputs(s)
    assert batch["pixel_values"].shape == (256, 1176)
    assert (batch["input_ids"] == s.arch["image_token_id"]).sum() == 64
    assert batch["labels"][0, :66].eq(-100).all()
    assert batch["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert all(torch.equal(v, make_inputs(s)[k]) for k, v in batch.items())
    assert not torch.equal(batch["pixel_values"], make_inputs(s, 1)["pixel_values"])
    with pytest.raises(ValueError):
        s.replace(image_size=225)
    with pytest.raises(ValueError):
        s.replace(seq_len=64)


def test_canonical_parameter_counts():
    from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    from evograd.benchmark.topdown.common.vision.spec import architecture

    for name, config, model, expected in (
        ("convnextv2_tiny", ConvNextV2Config, ConvNextV2ForImageClassification, 28635496),
        ("qwen2_vl_2b", Qwen2VLConfig, Qwen2VLForConditionalGeneration, 2208985600),
    ):
        with torch.device("meta"):
            instance = model(config.from_dict(architecture(name)))
        assert sum(p.numel() for p in instance.parameters()) == expected


def test_preflight_tracks_specialist_configuration():
    from evograd.evaluation.tier3.workloads.common.vision.sites import preflight_cases

    small = small_spec("qwen2_vl_2b")
    assert preflight_cases(small, "text_mlp")[0].dims == dict(B=1, T=8, H=32, I=64)
    large = small.replace(batch_size=2, seq_len=16)
    case = preflight_cases(large, "text_mlp")[0]
    assert case.dims["B"] == 2 and case.dims["T"] == 16
    assert case.provenance.source == "specification"


def test_convnext_specialist_patch_touches_only_selected_stage():
    from evograd.evaluation.tier3.patch import patch

    workload = VisionWorkload(small_spec("convnextv2_tiny"))
    site = "convnext_mlp_stage2"
    kernels = patch(
        KernelSet(registry=workload.site_registry),
        site,
        workload.site_registry.require(site).default,
    )
    model, provenance = workload.build_patched(kernels)
    assert provenance.actual_sites == (site,)
    assert all(".stages.2." in path for path in provenance.paths[site])
    assert len(workload.site_registry.require(site).preflight) == 1
    assert workload.site_registry.require(site).preflight[0].dims["H"] == 24


@pytest.mark.parametrize("padded", [False, True])
def test_qwen2_vl_host_inputs_are_the_hf_step_bitwise(padded):
    """The host-computed inputs (positions, masks, vision segments, image rows) give HF's
    own loss and gradients exactly, and are computed once per unchanged batch."""
    from evograd.benchmark.topdown.common.vision import model as vision

    spec = small_spec("qwen2_vl_2b")
    model = VisionWorkload(spec).build(KernelSet(registry=VisionWorkload(spec).site_registry))
    batch = VisionWorkload(spec).batch_for(seed=3)
    if padded:
        batch["attention_mask"][:, -2:] = 0

    def step(call):
        model.zero_grad(set_to_none=True)
        loss = call()
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}

    hf_loss, hf_grads = step(lambda: model(**batch, use_cache=False).loss)
    loss, grads = step(lambda: vision.forward(model, batch, spec.model).loss)
    assert torch.equal(loss, hf_loss)
    assert grads.keys() == hf_grads.keys()
    for name, grad in grads.items():
        assert torch.equal(grad, hf_grads[name]), name

    first = vision._host_inputs(model, batch)
    assert vision._host_inputs(model, batch) is first
    assert first["image_cu_seqlens"].device.type == "cpu"
    assert first["image_grid_thw"].device.type == "cpu"
    batch["input_ids"] = batch["input_ids"].clone()
    assert vision._host_inputs(model, batch) is not first

    # The evaluation's model cache saves the model and hands out deep copies.
    import io

    saved = io.BytesIO()
    torch.save(model, saved)
    saved.seek(0)
    for clone in (copy.deepcopy(model), torch.load(saved, weights_only=False)):
        assert clone.model.forward.args[0] is clone.model
        assert torch.equal(vision.forward(clone, batch, spec.model).loss, hf_loss)
        assert torch.equal(clone(**batch, use_cache=False).loss, hf_loss)
