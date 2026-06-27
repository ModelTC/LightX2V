import json
from pathlib import Path
from types import SimpleNamespace

from lightx2v.models.runners.hunyuan_video import hunyuan_video_15_runner as runner_mod
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner


def _runner(align_single_card_shape=None):
    runner = HunyuanVideo15Runner.__new__(HunyuanVideo15Runner)
    runner.config = {
        "aspect_ratio": "16:9",
        "transformer_model_name": "480p_i2v",
        "target_video_length": 121,
        "vae_stride": [4, 16, 16],
    }
    if align_single_card_shape is not None:
        runner.config["align_single_card_shape"] = align_single_card_shape
    runner.target_size_config = {
        "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
        "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
        "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
        "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
    }
    runner.vae_decoder = SimpleNamespace(world_size_h="unset", world_size_w="unset")
    return runner


def test_hy15_multicard_keeps_single_card_latent_shape_by_default(monkeypatch):
    monkeypatch.setattr(runner_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner_mod.dist, "get_world_size", lambda: 8)

    runner = _runner()

    latent_shape = runner.get_latent_shape_with_target_hw()

    assert latent_shape == [32, 31, 30, 53]
    assert runner.target_height == 480
    assert runner.target_width == 848
    assert runner.vae_decoder.world_size_h is None
    assert runner.vae_decoder.world_size_w is None


def test_hy15_can_opt_into_distributed_vae_decode_shape_adjustment(monkeypatch):
    monkeypatch.setattr(runner_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner_mod.dist, "get_world_size", lambda: 8)

    runner = _runner(align_single_card_shape=False)

    latent_shape = runner.get_latent_shape_with_target_hw()

    assert latent_shape == [32, 31, 32, 53]
    assert runner.target_height == 512
    assert runner.target_width == 848
    assert runner.vae_decoder.world_size_h == 8
    assert runner.vae_decoder.world_size_w == 1


def test_hy15_dist8_config_and_launcher_default_to_single_card_alignment():
    config = json.loads(Path("configs/platforms/nvidia/dist_8/hunyuan_video_t2v_480p.json").read_text(encoding="utf-8"))
    script = Path("scripts/platforms/nvidia/dist_8/run_hy15_t2v_480p.sh").read_text(encoding="utf-8")

    assert config["align_single_card_shape"] is True
    assert "gpus=${GPUS:-8}" in script
