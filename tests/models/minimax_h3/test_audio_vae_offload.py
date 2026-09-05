import importlib.util
import json
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn.utils.weight_norm import WeightNorm

REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture()
def audio_module(monkeypatch):
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    platform = types.ModuleType("lightx2v_platform.base.global_var")
    platform.AI_DEVICE = "cpu"
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    load("lightx2v.models.video_encoders.hf.minimax_h3.weights", "lightx2v/models/video_encoders/hf/minimax_h3/weights.py")
    return load("h3_audio_offload_under_test", "lightx2v/models/audio_encoders/hf/minimax_h3/audio_vae.py")


def tiny_model(audio_module):
    model = audio_module.MiniMaxH3AudioVAE(
        {
            "encoder_dim": 4,
            "latent_dim": 8,
            "latent_channels": 2,
            "num_attention_heads": 2,
            "decoder_dim": 8,
            "encoder_rates": [2],
            "decoder_rates": [2],
            "decoder_kernel_sizes": [4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
        },
        device="cpu",
        cpu_offload=True,
    )
    return model.eval().requires_grad_(False)


def test_repeated_public_decode_and_checkpoint_contract(audio_module, tmp_path):
    model = tiny_model(audio_module)
    latent = torch.randn(2, 2, 16)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    hooks = [(module, key, hook) for module in model.modules() for key, hook in module._forward_pre_hooks.items() if isinstance(hook, WeightNorm)]
    assert hooks
    weights = [weakref.ref(getattr(module, hook.name)) for module, _, hook in hooks]
    outputs = [model.decode(latent) for _ in range(3)]
    assert all(ref() is None for ref in weights)
    assert all(torch.equal(outputs[0], output) for output in outputs)
    assert outputs[0].shape == (1, 2, 32)
    for module, key, hook in hooks:
        assert module._forward_pre_hooks[key] is hook
        assert hook.name not in vars(module)
        assert hook.name + "_g" in module._parameters
        assert hook.name + "_v" in module._parameters
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    model.offload()  # Cleanup is also safe when the derived attribute is absent.
    model.to("cpu")
    model.load_state_dict(state, strict=True)
    assert torch.equal(outputs[0], model.decode(latent))

    component = tmp_path / "audio_vae"
    component.mkdir()
    (component / "config.json").write_text(json.dumps(model.config))
    save_file(model.state_dict(), component / "model.safetensors")
    loaded = audio_module.MiniMaxH3AudioVAE.from_pretrained(tmp_path, device="cpu", cpu_offload=True)
    assert set(loaded.load_report.loaded_keys) == set(state)
    assert torch.equal(outputs[0], loaded.decode(latent))
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert all(buffer.device.type == "cpu" for buffer in loaded.buffers())


def test_cleanup_only_targets_legacy_hook_attributes(audio_module):
    model = tiny_model(audio_module)
    plain = torch.nn.Linear(2, 2)
    plain.cache = torch.ones(2)
    model.unrelated = plain
    custom = torch.nn.Module()
    custom.register_parameter("kernel", torch.nn.Parameter(torch.randn(2, 2)))
    torch.nn.utils.weight_norm(custom, name="kernel", dim=1)
    model.custom = custom
    parameter = plain.weight
    cache = plain.cache
    model.offload()
    assert plain.weight is parameter
    assert plain.cache is cache
    assert "kernel" not in vars(custom)
    assert set(custom.state_dict()) == {"kernel_g", "kernel_v"}
