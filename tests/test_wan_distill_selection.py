from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner, get_wan_model_class
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.scheduler_factory import create_wan_scheduler, get_wan_distill_method
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan21MeanFlowStepDistillScheduler, WanStepDistillScheduler
from lightx2v.utils.input_info import T2VInputInfo
from lightx2v.utils.lockable_dict import LockableDict


def scheduler_config(**updates):
    config = {
        "model_cls": "wan2.1",
        "feature_caching": "NoCaching",
        "infer_steps": 4,
        "target_video_length": 81,
        "sample_shift": 5.0,
        "sample_guide_scale": [3.5, 3.5],
        "seq_parallel": False,
        "dim": 5120,
        "num_heads": 40,
        "denoising_step_list": [1000, 750, 500, 250],
    }
    config.update(updates)
    return config


def test_scheduler_selection():
    standard = create_wan_scheduler(scheduler_config())
    dmd2 = create_wan_scheduler(scheduler_config(model_cls="wan2.2_moe", distill_method="dmd2"))
    mean_flow = create_wan_scheduler(scheduler_config(distill_method="mean_flow"))

    assert type(standard) is WanScheduler
    assert isinstance(dmd2, WanStepDistillScheduler)
    assert isinstance(mean_flow, Wan21MeanFlowStepDistillScheduler)


def test_distill_method_validation():
    assert get_wan_distill_method({"model_cls": "wan2.1"}) is None
    assert get_wan_distill_method({"model_cls": "wan2.1", "distill_method": "mean_flow"}) == "mean_flow"
    assert get_wan_distill_method({"model_cls": "wan2.2_moe", "distill_method": "dmd2"}) == "dmd2"

    with pytest.raises(NotImplementedError, match="wan2.2_moe does not support distill_method 'mean_flow'"):
        get_wan_distill_method({"model_cls": "wan2.2_moe", "distill_method": "mean_flow"})
    with pytest.raises(NotImplementedError, match="wan2.1_vace does not support distill_method 'dmd2'"):
        get_wan_distill_method({"model_cls": "wan2.1_vace", "distill_method": "dmd2"})
    with pytest.raises(NotImplementedError, match="wan2.1 does not support distill_method 'dmd-2'"):
        create_wan_scheduler(scheduler_config(distill_method="dmd-2"))


def test_wan21_distill_scheduler_selection():
    runner = object.__new__(WanRunner)
    runner.config = scheduler_config(model_cls="wan2.1", distill_method="mean_flow")
    runner.init_scheduler()
    assert isinstance(runner.scheduler, Wan21MeanFlowStepDistillScheduler)

    runner.config["distill_method"] = "unknown"
    with pytest.raises(NotImplementedError, match="wan2.1 does not support distill_method 'unknown'"):
        runner.init_scheduler()


def test_wan21_distill_model_selection():
    assert get_wan_model_class(None) is WanModel
    assert get_wan_model_class("dmd2") is WanDistillModel


def test_multimodel_boundary_selection():
    normal = MultiModelStruct([None, None], {"boundary": 0.9})
    normal.scheduler = SimpleNamespace(step_index=0, timesteps=torch.tensor([1000, 750]))
    assert normal.uses_high_noise_model()
    assert normal.get_switch_step_index() == 1
    assert not hasattr(normal, "boundary_step_index")

    dmd2 = MultiModelStruct([None, None], {"model_cls": "wan2.2_moe", "distill_method": "dmd2", "boundary_step_index": 2})
    dmd2.scheduler = SimpleNamespace(step_index=1)
    assert dmd2.uses_high_noise_model()
    assert dmd2.get_switch_step_index() == 2
    assert not hasattr(dmd2, "boundary")


@pytest.mark.parametrize(
    ("config", "field"),
    [({}, "boundary"), ({"model_cls": "wan2.2_moe", "distill_method": "dmd2"}, "boundary_step_index")],
)
def test_multimodel_boundary_is_required(config, field):
    with pytest.raises(KeyError, match=field):
        MultiModelStruct([None, None], config)


def test_wan_input_info_excludes_infer_steps():
    input_info = T2VInputInfo()
    input_info.update({"infer_steps": 8})

    assert not hasattr(input_info, "infer_steps")


def test_wan_runner_keeps_scheduler_config_and_carries_request_frame_count():
    runner = object.__new__(WanRunner)
    runner.config = LockableDict(
        scheduler_config(
            task="t2v",
            target_height=720,
            target_width=1280,
        )
    )
    runner.supported_tasks = ("t2v",)
    runner.scheduler = SimpleNamespace(infer_steps=4)

    first = runner.create_input_info({"task": runner.config["task"], "infer_steps": 8, "target_video_length": 49})
    second = runner.create_input_info({"task": runner.config["task"]})

    assert runner.config["infer_steps"] == 4
    assert runner.scheduler.infer_steps == 4
    assert runner.config["target_video_length"] == 81
    assert first.target_video_length == 49
    assert second.target_video_length == 81


def test_wan_runner_rejects_frame_length_cached_at_startup():
    runner = object.__new__(WanRunner)
    runner.config = LockableDict(scheduler_config(task="t2v", target_height=720, target_width=1280, self_attn_1_type="svg_attn"))
    runner.supported_tasks = ("t2v",)
    with pytest.raises(ValueError, match="target_video_length"):
        runner.prepare_request({"task": runner.config["task"], "target_video_length": 49})
