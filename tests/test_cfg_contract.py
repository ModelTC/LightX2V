import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from lightx2v.models.networks.cosmos3.model import Cosmos3TransformerModel
from lightx2v.models.networks.ernie_image.model import ErnieImageTransformerModel
from lightx2v.models.networks.flux2.model import Flux2DevTransformerModel, Flux2KleinTransformerModel
from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model
from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.networks.longcat_image.model import LongCatImageTransformerModel
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.models.networks.wan.dreamzero_model import DreamZeroModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.z_image.model import ZImageTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct
from lightx2v.models.schedulers.cosmos3.scheduler import Cosmos3Scheduler
from lightx2v.models.schedulers.flux2.scheduler import Flux2Scheduler
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v.models.schedulers.lingbot_video.scheduler import LingBotVideoScheduler
from lightx2v.models.schedulers.wan.animate2.scheduler import WanAnimate2Scheduler
from lightx2v.models.schedulers.wan.scheduler import WanScheduler

STANDARD_MODELS = [
    WanModel,
    Cosmos3TransformerModel,
    Flux2KleinTransformerModel,
    HunyuanVideo15Model,
    ErnieImageTransformerModel,
    LongCatImageTransformerModel,
    QwenImageTransformerModel,
    ZImageTransformerModel,
    LTX2Model,
]


@pytest.fixture
def config(tmp_path):
    FlowMatchEulerDiscreteScheduler().save_pretrained(tmp_path / "scheduler")
    return {
        "model_path": str(tmp_path),
        "model_cls": "wan2.2_audio",
        "infer_steps": 4,
        "enable_cfg": True,
        "sample_guide_scale": 4.0,
        "sample_shift": 3.0,
        "seq_parallel": False,
        "parallel": None,
        "dim": 16,
        "num_heads": 2,
    }


@pytest.fixture
def inputs():
    return {
        "text_encoder_output": {
            "cond_input_ids": "positive",
            "uncond_input_ids": "negative",
            "prompt_embeds": "positive",
            "negative_prompt_embeds": "negative",
        }
    }


def make_model(model_cls, enabled=True, scale=4.0):
    model = object.__new__(model_cls)
    model.config = {"enable_cfg": enabled, "cfg_parallel": False, "task": "t2i"}
    model.cpu_offload = False
    model.scheduler = SimpleNamespace(latents=torch.zeros(1), sample_guide_scale=scale)
    values = [torch.tensor([10.0]), torch.tensor([2.0])]
    if model_cls is Cosmos3TransformerModel:
        values = [SimpleNamespace(vision=value, sound=None, action=None) for value in values]
    elif model_cls is LTX2Model:
        values = [(value, value * 2) for value in values]
    model._infer_cond_uncond = Mock(side_effect=values)
    return model


@pytest.mark.parametrize("model_cls", STANDARD_MODELS)
def test_enabled_cfg_rejects_unit_scale_before_forward(model_cls, inputs):
    model = make_model(model_cls, scale=1.0)
    with pytest.raises(AssertionError, match="enable_cfg=true requires sample_guide_scale != 1"):
        model.infer(inputs)
    model._infer_cond_uncond.assert_not_called()


@pytest.mark.parametrize("model_cls", STANDARD_MODELS)
def test_disabled_cfg_accepts_unit_scale_and_runs_one_branch(model_cls, inputs):
    model = make_model(model_cls, enabled=False, scale=1.0)
    model.infer(inputs)
    model._infer_cond_uncond.assert_called_once()


@pytest.mark.parametrize("scheduler_cls", [WanScheduler, Cosmos3Scheduler, Flux2Scheduler, HunyuanVideo15Scheduler])
def test_scheduler_initialization_does_not_validate_cfg(config, scheduler_cls):
    scheduler = scheduler_cls({**config, "sample_guide_scale": 1.0})
    assert scheduler.sample_guide_scale == 1.0


@pytest.mark.parametrize("model_cls", [WanModel, Cosmos3TransformerModel, Flux2KleinTransformerModel, HunyuanVideo15Model, LTX2Model])
@pytest.mark.parametrize(("enabled", "scale"), [(False, 4.0), (True, 0.5), (True, 4.0)])
def test_transformer_cfg_uses_only_the_flag(model_cls, enabled, scale, inputs):
    model = make_model(model_cls, enabled=enabled, scale=scale)
    model.infer(inputs)
    assert model._infer_cond_uncond.call_count == (2 if enabled else 1)
    expected = torch.tensor([2.0 + scale * 8 if enabled else 10.0])
    if model_cls is LTX2Model:
        torch.testing.assert_close(model.scheduler.v_noise_pred, expected)
        torch.testing.assert_close(model.scheduler.a_noise_pred, expected * 2)
    else:
        torch.testing.assert_close(model.scheduler.noise_pred, expected)


@pytest.mark.parametrize("scheduler_cls", [WanScheduler, WanAnimate2Scheduler, LingBotVideoScheduler])
def test_disagg_refresh_is_checked_when_model_runs(config, scheduler_cls, inputs):
    model = make_model(WanModel)
    model.scheduler = scheduler_cls(config)
    model.scheduler.refresh_from_config({**config, "sample_guide_scale": 1.0})
    with pytest.raises(AssertionError, match="sample_guide_scale"):
        model.infer(inputs)
    model._infer_cond_uncond.assert_not_called()


def test_wan_moe_checks_the_selected_stage_scale(config, inputs):
    wrapper = object.__new__(MultiModelStruct)
    wrapper.config = {**config, "sample_guide_scale": [4.0, 1.0], "cpu_offload": False}
    wrapper.scheduler = WanScheduler(wrapper.config)
    wrapper.model = [make_model(WanModel), make_model(WanModel)]
    for model in wrapper.model:
        model.scheduler = wrapper.scheduler
    wrapper.uses_high_noise_model = Mock(side_effect=[True, False])
    wrapper.infer(inputs)
    wrapper.model[0]._infer_cond_uncond.assert_called()
    with pytest.raises(AssertionError, match="sample_guide_scale"):
        wrapper.infer(inputs)
    wrapper.model[1]._infer_cond_uncond.assert_not_called()


@pytest.mark.parametrize("model_cls", [Cosmos3TransformerModel, LTX2Model])
@pytest.mark.parametrize("rank", [0, 1])
def test_cfg_parallel_keeps_guided_outputs(monkeypatch, model_cls, rank, inputs):
    model = make_model(model_cls)
    model.config.update(cfg_parallel=True, device_mesh=Mock())
    cond, uncond = torch.tensor([10.0]), torch.tensor([2.0])
    local = cond if rank == 0 else uncond
    model._infer_cond_uncond = Mock(return_value=(local, local * 2) if model_cls is LTX2Model else SimpleNamespace(vision=local, sound=None, action=None))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    calls = []

    def gather(outputs, value, group):
        factor = 2 if calls else 1
        calls.append(value)
        outputs[0].copy_(cond * factor)
        outputs[1].copy_(uncond * factor)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    model.infer(inputs)
    model._infer_cond_uncond.assert_called_once()
    if model_cls is LTX2Model:
        torch.testing.assert_close(model.scheduler.v_noise_pred, torch.tensor([34.0]))
        torch.testing.assert_close(model.scheduler.a_noise_pred, torch.tensor([68.0]))
    else:
        torch.testing.assert_close(model.scheduler.noise_pred, torch.tensor([34.0]))


@pytest.mark.parametrize("model_cls", [Cosmos3TransformerModel, LTX2Model, WanModel])
def test_unit_scale_fails_before_cfg_parallel_collectives(model_cls, inputs):
    model = make_model(model_cls, scale=1.0)
    model.config["cfg_parallel"] = True
    with pytest.raises(AssertionError, match="sample_guide_scale"):
        model.infer(inputs)
    model._infer_cond_uncond.assert_not_called()


def test_ltx_multimodal_guidance_keeps_its_own_scale_rules(inputs):
    model = make_model(LTX2Model, scale=1.0)
    model.scheduler.mm_guider_enabled = True
    model._infer_mm_guider_cfg = Mock()
    model.infer(inputs)
    model._infer_mm_guider_cfg.assert_called_once_with(inputs)
    model._infer_cond_uncond.assert_not_called()


def test_flux_dev_embedding_guidance_keeps_unit_scale(inputs):
    model = make_model(Flux2DevTransformerModel, scale=1.0)
    model.infer(inputs)
    model._infer_cond_uncond.assert_called_once()


@pytest.mark.parametrize("field", ["sample_guide_scale", "diff_guidance_scale"])
def test_hunyuan_image_checks_the_scale_when_combining_cfg(field):
    model = object.__new__(HunyuanImage3Model)
    model.config = {field: 4.0}
    cond, uncond = torch.tensor([10.0]), torch.tensor([2.0])
    torch.testing.assert_close(model.combine_cfg_predictions(cond, uncond), torch.tensor([34.0]))
    model.config[field] = 1.0
    with pytest.raises(AssertionError, match="guidance_scale"):
        model.combine_cfg_predictions(cond, uncond)


def test_dreamzero_checks_runtime_guide_scale():
    model = make_model(DreamZeroModel)
    model.config["sample_guide_scale"] = 4.0
    with pytest.raises(AssertionError, match="guide_scale"):
        model.infer({"guide_scale": 1.0})


def test_sr_uses_its_own_cfg_flag_and_conditioning(monkeypatch, config):
    root = Path(__file__).resolve().parents[1]
    config.update(json.loads((root / "configs/hunyuan_video_15/vsr/hy15_i2v_480p.json").read_text()))
    config["transformer_model_path"] = "/models/hunyuan/480p_i2v"
    monkeypatch.setattr(DefaultRunner, "__init__", lambda self, config: setattr(self, "config", config))
    runner = HunyuanVideo15Runner(config)
    assert runner.config["enable_cfg"] is True
    assert runner.config_sr["enable_cfg"] is False
    runner.config_sr.update(is_sr_running=True, cfg_parallel=True)
    qwen = Mock()
    byt5 = Mock()
    byt5.infer.return_value = ("features", "masks")
    runner.text_encoders = [qwen, byt5]
    output = runner.run_text_encoder(SimpleNamespace(prompt="positive", negative_prompt="negative"))
    qwen.infer.assert_called_once_with(["positive"])
    assert output["context_null"] is None


def test_cosmos_action_presets_explicitly_disable_cfg():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "configs/cosmos3").glob("cosmos3_*_omni_action_*.json"))
    assert len(paths) == 6
    for path in paths:
        config = json.loads(path.read_text())
        assert config["sample_guide_scale"] == 1.0
        assert config["enable_cfg"] is False
