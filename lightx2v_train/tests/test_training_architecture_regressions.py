from copy import deepcopy

import pytest
import torch
from lightx2v_train.data.training_cache import training_cache_info
from lightx2v_train.model_zoo.wan.wan_t2v import WanT2VModel
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler


def test_clean_augmentation_samples_from_clean_end_of_schedule(monkeypatch):
    scheduler = CausalForcingFlowMatchScheduler(
        num_train_timesteps=1000,
        time_shift_settings={"do_time_shift": False},
    )
    randint_bounds = {}

    def fake_randint(low, high, size, **kwargs):
        randint_bounds.update(low=low, high=high)
        return torch.full(size, low, **kwargs)

    monkeypatch.setattr(torch, "randint", fake_randint)
    sigmas = scheduler.sample_clean_augmentation(
        num_frames=6,
        num_frame_per_chunk=3,
        max_timestep=100,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert randint_bounds == {"low": 900, "high": 1000}
    assert torch.allclose(sigmas, torch.full_like(sigmas, 0.1), atol=1e-6)


@pytest.mark.parametrize("max_timestep", [0, 1001])
def test_clean_augmentation_rejects_invalid_max_timestep(max_timestep):
    scheduler = CausalForcingFlowMatchScheduler(
        num_train_timesteps=1000,
        time_shift_settings={"do_time_shift": False},
    )

    with pytest.raises(ValueError, match="max_timestep must be"):
        scheduler.sample_clean_augmentation(
            num_frames=3,
            num_frame_per_chunk=3,
            max_timestep=max_timestep,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_training_cache_signature_includes_dataset_type():
    config = {
        "model": {
            "name": "wan_t2v",
            "pretrained_model_name_or_path": "/models/wan",
        },
        "data": {
            "processor": {"name": "video"},
            "train": {"name": "video_dataset", "data_path": ["train.jsonl"]},
        },
        "training": {"method": "flow_matching", "flow_matching": {}},
    }
    prompt_config = deepcopy(config)
    prompt_config["data"]["train"]["name"] = "prompt_dataset"

    assert training_cache_info(config)["signature"] != training_cache_info(prompt_config)["signature"]


def test_wan_model_structure_ignores_teacher_forcing_config():
    config = {
        "model": {
            "name": "wan_t2v",
            "pretrained_model_name_or_path": "/models/wan",
            "running_dtype": "fp32",
            "causal": False,
            "num_frame_per_chunk": 5,
            "local_attn_size": 7,
            "sink_size": 2,
            "independent_first_frame": True,
        },
        "scheduler": {"num_train_timesteps": 1000},
        "training": {
            "teacher_forcing": {
                "enabled": True,
                "num_frame_per_chunk": 3,
                "local_attn_size": 11,
                "sink_size": 4,
                "independent_first_frame": False,
            }
        },
    }
    model = WanT2VModel(config)
    model.load_components(
        load_transformer=False,
        load_vae=False,
        load_condition_encoder=False,
    )

    assert model.use_causal_transformer is False
    assert model.num_frame_per_chunk == 5
    assert model.local_attn_size == 7
    assert model.sink_size == 2
    assert model.independent_first_frame is True
