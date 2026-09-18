"""Regression coverage for shared offload after the upstream H3 API changes."""

import argparse
import json

import pytest
import torch
from safetensors.torch import save_file

from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE
from lightx2v.utils.set_config import build_cli_inputs
from tools.cache_minimax_h3_adaln.cache_minimax_h3_adaln import parse_args


@pytest.mark.parametrize("task", ["t2av", "i2av", "l2av", "fl2av", "ref2av"])
def test_model_variant_and_request_task_remain_separate(tmp_path, task):
    for directory in ("transformer", "transformer_ref"):
        component = tmp_path / directory
        component.mkdir()
        (component / "config.json").write_text(json.dumps({"hidden_size": 64, "num_layers": 2}))
    config_file = tmp_path / "offload.json"
    config_file.write_text(json.dumps({"size": [544, 960], "num_frames": 124, "cpu_offload": True, "shared_cpu_weights": True}))
    variant = "ref2av" if task == "ref2av" else "fl2av"
    config, request = build_cli_inputs(argparse.Namespace(model_cls="minimax_h3", model_variant=variant, task=task, model_path=str(tmp_path), config_json=str(config_file), prompt="a fox"))
    assert config["model_variant"] == variant
    assert "task" not in config
    assert config["dit_original_ckpt"] == str(tmp_path / ("transformer_ref" if task == "ref2av" else "transformer"))
    assert request == {"task": task, "prompt": "a fox"}
    assert config["size"] == [544, 960] and config["num_frames"] == 124


@pytest.mark.parametrize("variant", ["fl2av", "ref2av"])
def test_adaln_cache_cli_accepts_model_variant(monkeypatch, variant):
    monkeypatch.setattr("sys.argv", ["cache", "--model_path", "/model", "--config_json", "/config", "--model-variant", variant])
    assert parse_args().model_variant == variant


@pytest.fixture
def vae_checkpoint(tmp_path):
    config = {
        "latent_channels": 4,
        "block_out_channels": [32, 32],
        "layers_per_block": 1,
        "spatial_downsample_factors": [2, 2],
        "temporal_downsample_factors": [1, 1],
        "norm_num_groups": 32,
        "decoder_num_layers": 3,
        "decoder_num_attention_heads": 4,
        "decoder_attention_head_dim": 16,
        "decoder_ffn_mult": 2,
        "clip_length": 1,
        "token_drop": 0,
    }
    model = MiniMaxH3VideoVAE(config, device="cpu")
    directory = tmp_path / "vae"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    save_file(model.state_dict(), directory / "model.safetensors")
    return tmp_path


def test_private_vae_preserves_upstream_channels_last(vae_checkpoint):
    model = MiniMaxH3VideoVAE.from_pretrained(vae_checkpoint, device="cpu", cpu_offload=True, encoder_conv_mode="torch_channels_last")
    assert model._use_channels_last_encoder_input
    for module in model.encoder.down_blocks.modules():
        if isinstance(module, torch.nn.Conv3d):
            assert module.weight.dtype == torch.float16
            assert module.weight.is_contiguous(memory_format=torch.channels_last_3d)


def test_shared_vae_rejects_unrepresented_layout_before_loading(tmp_path):
    with pytest.raises(ValueError, match="shared manifest"):
        MiniMaxH3VideoVAE.from_pretrained(tmp_path, cpu_offload=True, shared_cpu_config={}, encoder_conv_mode="torch_channels_last")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared pinned weights require CUDA")
def test_shared_vae_prepares_weights_on_meta_and_preserves_storage(vae_checkpoint):
    baseline = MiniMaxH3VideoVAE.from_pretrained(vae_checkpoint, device="cuda:0", cpu_offload=True)
    shared = MiniMaxH3VideoVAE.from_pretrained(
        vae_checkpoint,
        device="cuda:0",
        cpu_offload=True,
        offload_granularity="block",
        shared_cpu_config={"shared_cpu_weights": True, "shared_cpu_weight_backend": "sysv", "shared_cpu_weight_scope": "host"},
    )
    try:
        baseline.disable_tiling()
        shared.disable_tiling()
        expected_weights = baseline.state_dict()
        pointers = {}
        arena = shared.shared_cpu_weight_owner.arena
        for name, tensor in shared.state_dict().items():
            assert tensor.dtype == expected_weights[name].dtype
            assert torch.equal(tensor, expected_weights[name])
            assert tensor.is_pinned()
            assert tensor.data_ptr() == arena.address + arena.manifest.by_name[name].offset
            pointers[name] = tensor.data_ptr()
        latents = torch.randn((1, 4, 1, 4, 4), generator=torch.Generator().manual_seed(42))
        expected = baseline.decode(latents)
        for _ in range(2):
            torch.testing.assert_close(shared.decode(latents), expected, rtol=0, atol=0)
            assert {name: tensor.data_ptr() for name, tensor in shared.state_dict().items()} == pointers
    finally:
        shared.close_shared_cpu_weights()
