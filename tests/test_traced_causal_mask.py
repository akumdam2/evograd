"""A traced or captured step gets the attention eager runs (SDPA ``is_causal``).

transformers refuses the ``is_causal`` shortcut while ``is_tracing()`` -- under dynamo
and during CUDA-graph capture -- and materialises a 4D mask instead; see
``common.model.traced_causal_mask_as_eager``.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from evograd.benchmark.topdown.common.model import traced_causal_mask_as_eager  # noqa: E402

B, T = 2, 16


def _config():
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8,
                                      attn_implementation="sdpa")
    return config


def _mask(attention_mask=None, position_ids=None):
    import transformers.masking_utils as masking

    positions = torch.arange(T).expand(B, -1) if position_ids is None else position_ids
    return masking.create_causal_mask(config=_config(), inputs_embeds=torch.zeros(B, T, 32),
                                      attention_mask=attention_mask, past_key_values=None,
                                      position_ids=positions)


@pytest.fixture
def tracing(monkeypatch):
    import transformers.masking_utils as masking

    traced_causal_mask_as_eager()
    monkeypatch.setattr(masking, "is_tracing", lambda tensor=None: True)


def test_eager_is_unchanged():
    traced_causal_mask_as_eager()
    assert _mask() is None  # unpadded and unpacked: SDPA gets is_causal=True
    packed = torch.tensor([list(range(8)) + list(range(8))] * B)
    assert torch.is_tensor(_mask(position_ids=packed))  # eager still detects packing
    padded = torch.ones(B, T, dtype=torch.long)
    padded[0, -3:] = 0
    assert torch.is_tensor(_mask(attention_mask=padded))


def test_traced_unpadded_step_takes_the_is_causal_shortcut(tracing):
    assert _mask() is None


def test_traced_padding_still_gets_its_mask(tracing):
    padded = torch.ones(B, T, dtype=torch.long)
    padded[0, -3:] = 0
    assert torch.is_tensor(_mask(attention_mask=padded))


def test_installed_once():
    import transformers.masking_utils as masking

    traced_causal_mask_as_eager()
    first = masking.find_packed_sequence_indices
    traced_causal_mask_as_eager()
    assert masking.find_packed_sequence_indices is first


def test_compiled_model_calls_sdpa_with_is_causal():
    """Real dynamo on CPU: the traced graph's SDPA has no mask and is_causal=True."""
    import torch._dynamo

    from evograd.benchmark.topdown.common.model import build_model
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import CLASSES
    from tests.qwen3.test_level3_workload import tiny_spec

    model = build_model(tiny_spec(), CLASSES)
    calls = []

    def backend(gm, example_inputs):
        calls.extend(node for node in gm.graph.nodes
                     if "scaled_dot_product_attention" in str(node.target))
        return gm.forward

    torch._dynamo.reset()
    ids = torch.randint(0, 256, (2, 16))
    torch.compile(model, backend=backend)(input_ids=ids, labels=ids, use_cache=False)
    torch._dynamo.reset()
    assert calls, "no SDPA call traced"
    for node in calls:
        assert node.kwargs.get("is_causal") is True, node.format_node()
        mask = node.kwargs.get("attn_mask", node.args[3] if len(node.args) > 3 else None)
        assert mask is None, node.format_node()


def test_configure_installs_it_for_models_loaded_from_a_snapshot():
    """Evaluation children load a job-local snapshot without constructing the model."""
    import os
    import subprocess
    import sys

    code = ("from evograd.verification.integrations.base import configure; configure(); "
            "import transformers.masking_utils as m; assert m._evograd_traced_causal_mask")
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
