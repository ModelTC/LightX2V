from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.hidream_o1_image.hidream_o1_image_runner import HidreamO1ImageRunner
from lightx2v.models.runners.lingbot_video.lingbot_video_runner import LingBotVideoRunner
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.wan.wan_audio_runner import WanAudioRunner
from lightx2v.models.schedulers.flux2.scheduler import Flux2Scheduler
from lightx2v.shot_runner.rs2v_infer import ShotRS2VPipeline
from lightx2v.utils.input_info import INPUT_INFO_TYPES, UNSET, Cosmos3InputInfo, S2VInputInfo, SekoTalkInputs, T2IInputInfo, VaceInputInfo
from lightx2v.utils.va_controller import VAController


@pytest.mark.parametrize("media", ({}, {"video_path": None, "mask_path": None, "src_ref_images": None}))
def test_vace_missing_media_reaches_processor_as_none(media):
    runner = object.__new__(DefaultRunner)
    runner.input_info = VaceInputInfo(task="vace", target_shape=[480, 832], **media)
    runner.get_target_size = Mock(return_value=(480, 832))
    runner.prepare_source = Mock(return_value=([None], [None], [None]))
    runner.run_vae_encoder = Mock(return_value=("latents", [16, 21, 60, 104]))
    runner.run_text_encoder = Mock(return_value={})
    runner.maybe_empty_cache = Mock()
    runner.get_encoder_output_i2v = Mock()
    runner._run_input_encoder_local_vace()
    runner.prepare_source.assert_called_once_with([None], [None], [None], (832, 480))


def test_vace_reference_paths_remain_a_sequence():
    runner = object.__new__(DefaultRunner)
    runner.input_info = VaceInputInfo(task="vace", video_path="clip.mp4", mask_path="mask.mp4", src_ref_images="a.png,b.png")
    runner.get_target_size = Mock(return_value=(480, 832))
    runner.prepare_source = Mock(return_value=([None], [None], [None]))
    runner.run_vae_encoder = Mock(return_value=("latents", [16, 21, 60, 104]))
    runner.run_text_encoder = Mock(return_value={})
    runner.maybe_empty_cache = Mock()
    runner.get_encoder_output_i2v = Mock()
    runner._run_input_encoder_local_vace()
    runner.prepare_source.assert_called_once_with(["clip.mp4"], ["mask.mp4"], [["a.png", "b.png"]], (832, 480))


@pytest.mark.parametrize("task", ("t2i", "t2v"))
@pytest.mark.parametrize("enable_cfg", (False, True))
def test_lingbot_warmup_preserves_conditioning_and_latent_shape(task, enable_cfg):
    runner = object.__new__(LingBotVideoRunner)
    runner.config = {"task": task, "target_video_length": 9, "enable_cfg": enable_cfg}
    runner.text_encoders = [SimpleNamespace(infer=Mock(side_effect=[{"prompt_embeds": torch.zeros(1, 3, 2), "prompt_mask": None}, {"prompt_embeds": torch.zeros(1, 2, 2), "prompt_mask": None}]))]
    runner.maybe_empty_cache = Mock()
    encoded = runner._prepare_warmup_inputs(256, 256)
    assert encoded["prompt_embeds"].shape == (1, 3, 2)
    assert ("negative_prompt_embeds" in encoded) == enable_cfg
    if enable_cfg:
        assert encoded["negative_prompt_embeds"].shape == (1, 2, 2)
    assert runner.input_info.latent_shape == (1, 16, 1 if task == "t2i" else 3, 32, 32)


@pytest.mark.parametrize("task,needs_image,needs_last", (("t2av", False, False), ("i2av", True, False), ("l2av", False, True), ("fl2av", True, True), ("ref2av", True, False)))
def test_minimax_warmup_keeps_keyframe_placement(task, needs_image, needs_last):
    runner = object.__new__(MiniMaxH3Runner)
    runner.config = {"task": task}
    runner._prepare_warmup_inputs(64, 64, 9)
    assert type(runner.input_info) is INPUT_INFO_TYPES[task]
    assert bool(getattr(runner.input_info, "image_path", "")) == needs_image
    assert bool(getattr(runner.input_info, "last_frame_path", "")) == needs_last
    assert runner.input_info.task == task
    if task == "fl2av":
        assert runner.input_info.image_path is not runner.input_info.last_frame_path


def test_mutable_context_defaults_are_not_shared():
    first, second = Cosmos3InputInfo(), Cosmos3InputInfo()
    mutable_fields = [f.name for f in fields(first) if isinstance(getattr(first, f.name), (list, dict))]
    assert mutable_fields
    for name in mutable_fields:
        assert getattr(first, name) is not getattr(second, name)
    first.vision_condition_latents = torch.ones(1)
    assert second.vision_condition_latents is None
    assert second.action_latents is None


def test_shot_omitted_values_keep_unset_semantics():
    info = SekoTalkInputs()
    info.update({"prompt": "shot", "seed": 0, "target_video_length": 17})
    assert info.prompt == "shot"
    assert info.seed == 0
    assert info.target_video_length == 17
    for name in ("negative_prompt", "image_path", "audio_path", "stream_config", "overlap_latent", "audio_clip"):
        assert getattr(info, name) is UNSET


@pytest.mark.parametrize("stream_config", (None, {}, {"enabled": True}))
def test_stream_controller_keeps_none_and_explicit_configs(stream_config):
    controller = object.__new__(VAController)
    controller.reader = None
    controller.recorder = None
    info = S2VInputInfo(target_shape=[480, 832], stream_config=stream_config)
    controller.init_base({}, info, False, False)
    assert controller.stream_config is stream_config


def test_flux2_clears_optional_text_positions_between_requests(monkeypatch):
    monkeypatch.setattr("lightx2v.models.schedulers.flux2.scheduler.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.models.schedulers.flux2.scheduler.reset_scheduler_fls_state", Mock())
    scheduler = object.__new__(Flux2Scheduler)
    scheduler.generator = None
    scheduler.dtype = torch.float32
    scheduler.set_timesteps = Mock()
    positions = torch.zeros(1, 3, 4)
    scheduler.prepare(T2IInputInfo(seed=42, latent_shape=[1, 4, 2], txt_ids=positions))
    assert scheduler.txt_ids is positions
    scheduler.prepare(T2IInputInfo(seed=42, latent_shape=[1, 4, 2]))
    assert scheduler.txt_ids is None


@pytest.mark.parametrize("request_frames,expected_frames", ((None, 81), (0, 81), (33, 33)))
def test_wan_audio_image_encoding_uses_existing_frame_resolution(monkeypatch, request_frames, expected_frames):
    monkeypatch.setattr("lightx2v.models.runners.wan.wan_audio_runner.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.models.runners.wan.wan_audio_runner.resize_image", lambda img, **kwargs: (img, 16, 16))
    runner = object.__new__(WanAudioRunner)
    runner.config = {"target_video_length": 81, "vae_stride": [4, 8, 8], "patch_size": [1, 2, 2]}
    runner.input_info = S2VInputInfo(target_video_length=request_frames)
    runner._get_image_resize_kwargs = Mock(return_value={})
    runner._resolve_patched_spatial_size = Mock(return_value=([16, 16], 1, 1))
    _, latent_shape, _ = runner.read_image_input(Image.new("RGB", (16, 16)))
    assert latent_shape == [16, (expected_frames - 1) // 4 + 1, 2, 2]


@pytest.mark.parametrize("shape,expected", ((None, None), ([], []), ([16, 4, 8, 8], [16, 9, 8, 8])))
def test_shot_updates_only_a_prepared_latent_shape(shape, expected):
    info = SekoTalkInputs(latent_shape=shape)
    ShotRS2VPipeline._update_latent_shape(info, target_len=33, vae_stride=4)
    assert info.latent_shape == expected


@pytest.mark.parametrize("layout", [UNSET, None, "", '[{"bbox": [0, 0, 32, 32]}]'])
def test_hidream_layout_request_overrides_config_before_encoding(monkeypatch, layout):
    configured = '[{"bbox": [0, 0, 64, 64]}]'
    runner = object.__new__(HidreamO1ImageRunner)
    BaseRunner.__init__(runner, {"task": "i2i", "layout_bboxes": configured})
    runner.model = SimpleNamespace(model_config={}, device="cpu")
    runner.tokenizer = runner.processor = None
    runner.dtype = torch.float32
    runner._resolve_generation_config = Mock(return_value={"enable_cfg": False})
    build_samples = Mock(return_value={"samples": [], "tgt_image_len": 0})
    monkeypatch.setattr("lightx2v.models.networks.hidream_o1_image.i2i_utils.build_i2i_samples", build_samples)
    request = {"image_path": "reference.png", "layout_bboxes": layout}
    runner.input_info = runner.prepare_request(request)

    runner._run_input_encoder_local_i2i()

    expected = configured if layout is UNSET or layout is None else layout or None
    assert build_samples.call_args.kwargs["layout_bboxes"] == expected
    assert runner.config["layout_bboxes"] == configured
    assert runner.prepare_request({}).layout_bboxes == configured
