import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from lightx2v import infer
from lightx2v.disagg.examples import infer as disagg_infer
from lightx2v.models.networks.worldplay.pose_utils import pose_to_input
from lightx2v.models.runners import runner_factory
from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner
from lightx2v.models.runners.flux2.flux2_runner import Flux2Runner
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v.models.runners.ltx2.ltx25_runner import LTX25Runner
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, PROMPT_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.runners.seedvr.seedvr_runner import SeedVRRunner
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner
from lightx2v.models.runners.wan.wan_audio_runner import WanAudioARRunner
from lightx2v.models.runners.wan.wan_dancer_runner import WanDancerRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.pipeline import LightX2VPipeline
from lightx2v.server.api import openai_images
from lightx2v.server.api.tasks import image as image_api
from lightx2v.server.schema import ImageTaskRequest, VideoTaskRequest
from lightx2v.server.services.generation.base import BaseGenerationService
from lightx2v.server.services.generation.image import ImageGenerationService
from lightx2v.server.services.inference.worker import TorchrunInferenceWorker
from lightx2v.utils import set_config as set_config_utils
from lightx2v.utils.input_info import (
    FL2AVInputInfo,
    I2AVInputInfo,
    I2IInputInfo,
    I2VAInputInfo,
    L2AVInputInfo,
    NeoppInputInfo,
    Ref2AVInputInfo,
    SRInputInfo,
    T2AVInputInfo,
    T2IInputInfo,
)
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.set_config import build_cli_inputs


class RecordingScheduler:
    def __init__(self, config=None):
        self.config = config
        self.infer_steps = config["infer_steps"] if config else None


class RecordingRunner(BaseRunner):
    supported_request_fields_by_task = {
        "t2v": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | VIDEO_OUTPUT_FIELDS,
        "t2av": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | VIDEO_OUTPUT_FIELDS,
        "t2i": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | {"target_shape"},
        "sr": COMMON_REQUEST_FIELDS | {"image_path", "video_path", "sr_ratio", "target_shape"},
        "animate": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | {"src_face_path", "src_pose_path"},
    }

    def __init__(self, task="t2v"):
        config = {
            "task": task,
            "target_video_length": 81,
            "target_height": 720,
            "target_width": 1280,
        }
        super().__init__(config)
        self.requests = []
        self.pipeline_inputs = []
        self.lifecycle = []

    def prepare_request(self, request_data):
        input_info = super().prepare_request(request_data)
        self.lifecycle.append("prepare")
        return input_info

    def init_modules(self):
        self.lifecycle.append("init")

    def create_input_info(self, request_data):
        self.requests.append(request_data)
        return super().create_input_info(request_data)

    def run_pipeline(self, input_info):
        self.pipeline_inputs.append(input_info)
        self.lifecycle.append("run")
        return input_info


class MultitaskRecordingRunner(RecordingRunner):
    supported_request_fields_by_task = {
        "t2i": COMMON_REQUEST_FIELDS | {"aspect_ratio", "prompt", "target_shape"},
        "i2i": COMMON_REQUEST_FIELDS | {"image_path", "prompt", "target_shape"},
    }

    def get_supported_tasks(self):
        return ("t2i", "i2i")


class FileOnlyRecordingRunner(RecordingRunner):
    supported_request_fields_by_task = {
        "i23d": COMMON_REQUEST_FIELDS - {"return_result_tensor"},
    }


def make_minimax_h3_runner(task):
    runner = object.__new__(MiniMaxH3Runner)
    runner.config = LockableDict(
        {
            "model_cls": "minimax_h3",
            "task": task,
            "target_video_length": 124,
            "target_height": 544,
            "target_width": 960,
        }
    )
    runner.loaded_transformer_partition = "transformer_ref" if task == "ref2av" else "transformer"
    runner.supported_tasks = runner.get_supported_tasks()
    return runner


class RequestConfigRunner(DefaultRunner):
    supported_request_fields_by_task = {
        "t2v": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | VIDEO_OUTPUT_FIELDS,
    }

    def set_init_device(self):
        pass

    def init_scheduler(self):
        self.scheduler = RecordingScheduler(self.config)


class FixedShapeWanRunner(WanRunner):
    supported_request_fields_by_task = {
        "t2v": WanRunner.supported_request_fields_by_task["t2v"] - {"target_shape"},
    }

    def set_init_device(self):
        pass

    def init_scheduler(self):
        pass


class RecordingInferenceService:
    def __init__(self):
        self.requests = []
        self.worker = SimpleNamespace(runner=SimpleNamespace(config={}))

    async def submit_task_async(self, task_data):
        self.requests.append(task_data)
        return {"status": "success", "save_result_path": task_data["save_result_path"]}


class RecordingGenerationService(BaseGenerationService):
    def get_output_extension(self):
        return ".mp4"

    def get_task_type(self):
        return "t2v"


def create_runner():
    config = LockableDict(
        {
            "infer_steps": 40,
            "target_video_length": 81,
            "target_height": 720,
            "target_width": 1280,
            "resize_mode": "adaptive",
            "video_frame_interpolation": {"algo": "rife", "target_fps": 24},
            "task": "t2v",
        }
    )
    runner = RequestConfigRunner(config)
    runner.config.lock()
    return runner


def test_request_values_are_stored_in_input_info_only():
    runner = create_runner()

    input_info = runner.create_input_info(
        {
            "task": runner.config["task"],
            "task_id": "task-1",
            "prompt": "ignored by runner config",
            "infer_steps": 20,
            "target_video_length": 49,
            "target_shape": [480, 832],
            "resize_mode": "fixed_shape",
        }
    )

    assert runner.config["infer_steps"] == 40
    assert runner.config["target_video_length"] == 81
    assert runner.config["target_height"] == 720
    assert runner.config["target_width"] == 1280
    assert runner.config["resize_mode"] == "adaptive"
    assert "task_id" not in runner.config
    assert "prompt" not in runner.config
    assert input_info.target_video_length == 49
    assert input_info.target_shape == [480, 832]
    assert input_info.prompt == "ignored by runner config"
    assert input_info.negative_prompt == ""
    assert not hasattr(input_info, "infer_steps")
    assert not hasattr(input_info, "resize_mode")


def test_each_request_gets_startup_defaults_without_state_leakage():
    runner = create_runner()
    first = runner.create_input_info({"task": runner.config["task"], "infer_steps": 20, "target_video_length": 49, "target_shape": [480, 832]})
    second = runner.create_input_info({"task": runner.config["task"], "prompt": "second request"})

    assert runner.config["infer_steps"] == 40
    assert runner.config["target_video_length"] == 81
    assert runner.config["target_height"] == 720
    assert runner.config["target_width"] == 1280
    assert runner.config["resize_mode"] == "adaptive"
    assert runner.scheduler.infer_steps == 40
    assert first.target_video_length == 49
    assert first.target_shape == [480, 832]
    assert second.target_video_length == 81
    assert second.target_shape == [720, 1280]
    assert first is not second


def test_explicit_aspect_ratio_takes_precedence_over_startup_dimensions():
    runner = MultitaskRecordingRunner(task="t2i")

    input_info = runner.prepare_request({"task": runner.config["task"], "aspect_ratio": "16:9"})

    assert input_info.aspect_ratio == "16:9"
    assert input_info.target_shape == []


def test_ernie_uses_request_aspect_ratio_over_startup_dimensions():
    runner = object.__new__(ErnieImageRunner)
    runner.config = {
        "task": "t2i",
        "target_height": 480,
        "target_width": 832,
        "vae_scale_factor": 16,
    }
    runner.supported_tasks = ("t2i",)
    runner.resolution = 1024
    runner.input_info = runner.prepare_request({"task": runner.config["task"], "aspect_ratio": "16:9"})

    runner.set_latent_shape()

    assert runner.input_info.target_shape == [768, 1360]


def test_qwen_uses_request_aspect_ratio_over_startup_dimensions():
    runner = object.__new__(QwenImageRunner)
    runner.config = {
        "task": "t2i",
        "target_height": 480,
        "target_width": 832,
    }
    runner.supported_tasks = ("t2i",)
    runner.input_info = runner.prepare_request({"task": runner.config["task"], "aspect_ratio": "16:9"})

    assert runner.get_custom_shape() == (1664, 928)


def test_video_frame_interpolation_is_startup_only():
    runner = create_runner()

    input_info = runner.create_input_info({"task": runner.config["task"], "video_frame_interpolation": {"algo": "rife", "target_fps": 30}})

    assert runner.config["video_frame_interpolation"]["target_fps"] == 24
    assert not hasattr(input_info, "video_frame_interpolation")
    with pytest.raises(TypeError, match="Dictionary is locked"):
        runner.config["video_frame_interpolation"]["target_fps"] = 60


def test_runner_rejects_startup_and_internal_fields():
    runner = create_runner()

    with pytest.raises(ValueError, match="infer_steps, latent_shape, resize_mode"):
        runner.prepare_request(
            {
                "task": runner.config["task"],
                "infer_steps": 20,
                "latent_shape": [1, 2, 3],
                "resize_mode": "fixed_shape",
            }
        )


@pytest.mark.parametrize(
    ("runner_cls", "model_cls", "task"),
    [
        (WanRunner, "wan2.1", "omni_vision_task"),
        (WanRunner, "wan2.1", "i2i"),
        (SenseNovaVisionRunner, "sensenova_vision", "t2v"),
        (NeoppRunner, "neopp", "t2v"),
        (MiniMaxH3Runner, "minimax_h3", "t2i"),
    ],
)
def test_runner_rejects_unsupported_startup_task_before_device_init(monkeypatch, runner_cls, model_cls, task):
    device_init = Mock()
    monkeypatch.setattr(runner_cls, "set_init_device", device_init)

    with pytest.raises(ValueError, match=f"{runner_cls.__name__} does not support task"):
        runner_cls({"model_cls": model_cls, "task": task})

    device_init.assert_not_called()


@pytest.mark.parametrize("task", [None, ""])
@pytest.mark.parametrize("runner_cls", [RecordingRunner, NeoppRunner, MiniMaxH3Runner])
def test_runner_requires_startup_task_even_with_task_selection_override(runner_cls, task):
    with pytest.raises(ValueError, match="task must be set"):
        if runner_cls is RecordingRunner:
            runner_cls(task=task)
        else:
            runner_cls({"task": task})


def test_task_selection_override_does_not_bypass_startup_validation():
    with pytest.raises(ValueError, match="MultitaskRecordingRunner does not support task 't2v'"):
        MultitaskRecordingRunner(task="t2v")


@pytest.mark.parametrize("task", ["t2av", "i2av", "l2av", "fl2av", "ref2av"])
def test_minimax_initializes_task_group_and_loaded_partition(monkeypatch, task):
    monkeypatch.setattr(MiniMaxH3Runner, "set_init_device", lambda self: None)
    monkeypatch.setattr(MiniMaxH3Runner, "init_scheduler", lambda self: None)

    runner = MiniMaxH3Runner({"task": task})

    assert runner.supported_tasks == (("ref2av",) if task == "ref2av" else ("t2av", "i2av", "l2av", "fl2av"))
    assert runner.loaded_transformer_partition == ("transformer_ref" if task == "ref2av" else "transformer")


def test_request_schema_has_no_generation_defaults():
    request = VideoTaskRequest()

    assert request.target_video_length is None
    assert "infer_steps" not in VideoTaskRequest.model_fields
    assert "target_fps" not in VideoTaskRequest.model_fields
    assert "resize_mode" not in VideoTaskRequest.model_fields
    assert "target_video_length" not in request.model_fields_set

    explicit_request = VideoTaskRequest(num_frames=49)
    assert explicit_request.target_video_length == 49
    assert not hasattr(explicit_request, "infer_steps")
    assert not hasattr(explicit_request, "target_fps")
    assert not hasattr(explicit_request, "resize_mode")
    with pytest.raises(ValueError, match="extra_forbidden"):
        VideoTaskRequest(infer_steps=20, target_fps=30, resize_mode="fixed_shape")


def test_generation_service_forwards_only_explicit_generation_fields(tmp_path):
    inference_service = RecordingInferenceService()
    file_service = SimpleNamespace(get_output_path=lambda path: tmp_path / path)
    service = RecordingGenerationService(file_service, inference_service)
    message = VideoTaskRequest(prompt="request", num_frames=49, target_shape=[480, 832])

    asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    task_data = inference_service.requests[0]
    assert task_data["target_video_length"] == 49
    assert task_data["target_shape"] == [480, 832]
    assert "infer_steps" not in task_data
    assert "resize_mode" not in task_data
    assert "target_fps" not in task_data


def test_image_generation_does_not_expect_video_only_fields(tmp_path):
    inference_service = RecordingInferenceService()
    file_service = SimpleNamespace(get_output_path=lambda path: tmp_path / path)
    service = RecordingGenerationService(file_service, inference_service)

    asyncio.run(service.generate_with_stop_event(ImageTaskRequest(prompt="image"), asyncio.Event()))

    assert inference_service.requests[0]["prompt"] == "image"


def test_image_generation_omits_disabled_memory_result(tmp_path):
    inference_service = RecordingInferenceService()
    file_service = SimpleNamespace(get_output_path=lambda path: tmp_path / path)
    service = ImageGenerationService(file_service, inference_service)

    asyncio.run(service.generate_with_stop_event(ImageTaskRequest(prompt="image"), asyncio.Event()))

    assert "return_result_tensor" not in inference_service.requests[0]


def test_image_form_omits_unsubmitted_optional_fields(tmp_path, monkeypatch):
    messages = []
    services = SimpleNamespace(file_service=SimpleNamespace(input_image_dir=tmp_path))
    monkeypatch.setattr(image_api, "get_services", lambda: services)
    monkeypatch.setattr(image_api.task_manager, "create_task", lambda message: messages.append(message) or message.task_id)

    asyncio.run(
        image_api.create_image_task_form(
            request=SimpleNamespace(form=AsyncMock(return_value={})),
            image_file=None,
            prompt="",
            save_result_path="",
            negative_prompt="",
            seed=42,
            aspect_ratio=None,
        )
    )

    assert {"aspect_ratio", "image_path", "negative_prompt", "prompt"}.isdisjoint(messages[0].model_fields_set)


def test_openai_image_request_omits_empty_optional_fields():
    message = openai_images._build_image_task_request(prompt="image", task="t2i")

    assert {"image_mask_path", "image_path", "negative_prompt"}.isdisjoint(message.model_fields_set)


def test_server_requests_restore_startup_config_and_input_state():
    runner = create_runner()
    requests = []

    def run_pipeline(input_info):
        requests.append(
            {
                "infer_steps": runner.config["infer_steps"],
                "target_video_length": runner.config["target_video_length"],
                "target_height": runner.config["target_height"],
                "target_width": runner.config["target_width"],
                "resize_mode": runner.config["resize_mode"],
                "target_fps": runner.config["video_frame_interpolation"]["target_fps"],
                "input_target_shape": input_info.target_shape,
                "input_target_video_length": input_info.target_video_length,
                "negative_prompt": input_info.negative_prompt,
                "input_info": input_info,
            }
        )

    runner.run_pipeline = run_pipeline
    worker = TorchrunInferenceWorker()
    worker.rank = 0
    worker.world_size = 1
    worker.runner = runner

    asyncio.run(
        worker.process_request(
            {
                "task_id": "first",
                "target_video_length": 49,
                "target_shape": [480, 832],
            }
        )
    )
    asyncio.run(worker.process_request({"task_id": "second"}))

    assert requests[0]["infer_steps"] == 40
    assert requests[0]["target_video_length"] == 81
    assert requests[0]["target_height"] == 720
    assert requests[0]["target_width"] == 1280
    assert requests[0]["resize_mode"] == "adaptive"
    assert requests[0]["target_fps"] == 24
    assert requests[0]["input_target_shape"] == [480, 832]
    assert requests[0]["input_target_video_length"] == 49
    assert requests[1]["infer_steps"] == 40
    assert requests[1]["target_video_length"] == 81
    assert requests[1]["target_height"] == 720
    assert requests[1]["target_width"] == 1280
    assert requests[1]["resize_mode"] == "adaptive"
    assert requests[1]["target_fps"] == 24
    assert requests[1]["input_target_shape"] == [720, 1280]
    assert requests[1]["input_target_video_length"] == 81
    assert requests[0]["negative_prompt"] == ""
    assert requests[1]["negative_prompt"] == ""
    assert requests[0]["input_info"] is not requests[1]["input_info"]


def test_wan_i2v_reuse_key_includes_target_shape():
    runner = object.__new__(WanRunner)
    runner.config = {
        "task": "i2v",
        "target_video_length": 81,
        "target_height": 480,
        "target_width": 832,
        "resize_mode": None,
    }
    runner.input_info = SimpleNamespace(prompt="same", negative_prompt="", image_path="same.png", target_video_length=81, target_shape=[480, 832])
    first_key = runner.reuse_key()

    runner.input_info.target_shape = [720, 1280]

    assert runner.reuse_key() != first_key


def test_hunyuan_rejects_request_shape_and_uses_request_length_for_vsr():
    runner = object.__new__(HunyuanVideo15Runner)
    runner.config = LockableDict(
        {
            "infer_steps": 50,
            "target_video_length": 121,
            "target_height": 480,
            "target_width": 832,
            "task": "i2v",
            "vae_stride": (4, 16, 16),
        }
    )
    runner.scheduler = RecordingScheduler(runner.config)
    runner.sr_version = "720p_sr_distilled"
    runner.config_sr = {
        "target_video_length": 121,
        "vae_stride": (4, 16, 16),
        "video_super_resolution": {"base_resolution": "480p"},
    }
    runner.lq_latents_shape = (32, 31, 30, 40)
    runner.build_bucket_map = lambda **kwargs: lambda size: (1280, 720)
    runner.supported_tasks = ("i2v",)

    with pytest.raises(ValueError, match="target_shape"):
        runner.prepare_request({"task": runner.config["task"], "target_shape": [720, 1280]})

    runner.input_info = runner.create_input_info({"task": runner.config["task"], "target_video_length": 49})

    assert runner.config["target_video_length"] == 121
    assert runner.get_sr_latent_shape_with_target_hw()[1] == 13


def test_cli_keeps_json_defaults_separate_from_request_values(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"hidden_size": 256}', encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"aspect_ratio": "16:9", "infer_steps": 40, "target_video_length": 81, "target_height": 720, "target_width": 1280}',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config_json=str(config_path),
        model_cls="test_model",
        model_path=str(model_path),
        task="t2v",
        target_video_length=49,
    )

    startup_config, request_data = build_cli_inputs(args)

    assert startup_config["hidden_size"] == 256
    assert startup_config["infer_steps"] == 40
    assert startup_config["target_video_length"] == 81
    assert startup_config["target_height"] == 720
    assert startup_config["target_width"] == 1280
    assert startup_config["aspect_ratio"] == "16:9"
    assert request_data["target_video_length"] == 49


def test_cli_uses_runner_as_request_field_authority(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        config_json=str(config_path),
        model_cls="test_model",
        model_path=str(model_path),
        task="t2v",
        model_specific_input="value",
    )

    startup_config, request_data = build_cli_inputs(args)

    assert "model_specific_input" not in startup_config
    assert request_data["model_specific_input"] == "value"


@pytest.mark.parametrize(
    ("cli", "frame_args", "expected_frames"),
    [(infer, ["--num_frames", "49"], 49), (disagg_infer, [], 81)],
    ids=["cli", "disagg_cli"],
)
def test_cli_passes_explicit_request_fields_to_runner(tmp_path, monkeypatch, cli, frame_args, expected_frames):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"infer_steps": 40, "target_video_length": 81, "target_height": 720, "target_width": 1280}',
        encoding="utf-8",
    )
    runner = RecordingRunner()
    monkeypatch.setattr(
        "sys.argv",
        [
            "lightx2v.infer",
            "--model_cls",
            "wan2.1",
            "--task",
            "t2v",
            "--model_path",
            str(model_path),
            "--config_json",
            str(config_path),
            *frame_args,
            "--target_shape",
            "480",
            "832",
        ],
    )
    monkeypatch.setattr(runner_factory, "RUNNER_REGISTER", {"wan2.1": lambda config: runner})
    monkeypatch.setattr(cli, "print_config", lambda config, **kwargs: None)
    monkeypatch.setattr(cli, "print_request", lambda input_info, supported_request_fields: None)
    monkeypatch.setattr(cli, "validate_config_paths", lambda config: None)
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", lambda seed: None)

    cli.main()

    assert runner.pipeline_inputs[0].target_video_length == expected_frames
    assert runner.pipeline_inputs[0].target_shape == [480, 832]
    assert runner.lifecycle == ["init", "prepare", "run"]


def test_print_request_logs_readable_effective_fields(monkeypatch):
    input_info = I2VAInputInfo(
        task="i2va",
        prompt="move forward",
        seed=7,
        target_shape=[480, 832],
        policy_image=object(),
        policy_state=object(),
    )
    logged = {}

    def record_config(config, title="config"):
        logged["config"] = config
        logged["title"] = title

    monkeypatch.setattr(set_config_utils, "print_config", record_config)

    set_config_utils.print_request(input_info, {"task", "policy_image", "policy_state", "prompt", "seed", "target_shape"})

    assert logged == {
        "config": {
            "task": "i2va",
            "seed": 7,
            "prompt": "move forward",
            "target_shape": [480, 832],
        },
        "title": "Effective request",
    }


@pytest.mark.parametrize(("requested_frames", "aligned_frames"), [(80, 81), (82, 81)])
def test_wan_aligns_request_frames_with_warning(requested_frames, aligned_frames):
    runner = object.__new__(WanRunner)
    runner.config = {"task": "t2v", "vae_stride": (4, 8, 8)}
    runner.supported_tasks = ("t2v",)
    warnings = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")

    try:
        input_info = runner.prepare_request({"task": runner.config["task"], "target_video_length": requested_frames})
    finally:
        logger.remove(sink_id)

    assert input_info.target_video_length == aligned_frames
    assert any(f"using {aligned_frames} instead of {requested_frames}" in message for message in warnings)


def test_wan_keeps_valid_request_frames_without_warning():
    runner = object.__new__(WanRunner)
    runner.config = {"task": "t2v", "vae_stride": (4, 8, 8)}
    runner.supported_tasks = ("t2v",)
    warnings = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")

    try:
        input_info = runner.prepare_request({"task": runner.config["task"], "target_video_length": 81})
    finally:
        logger.remove(sink_id)

    assert input_info.target_video_length == 81
    assert not warnings


def test_wan_does_not_normalize_fixed_frame_count():
    runner = object.__new__(WanRunner)
    runner.config = {
        "self_attn_1_type": "radial_attn",
        "task": "t2v",
        "target_video_length": 80,
        "vae_stride": (4, 8, 8),
    }
    runner.supported_tasks = ("t2v",)

    input_info = runner.prepare_request({"task": runner.config["task"]})

    assert input_info.target_video_length == 80


def test_fixed_frame_attention_preserves_runner_request_restrictions():
    runner = FixedShapeWanRunner(
        {
            "enable_cfg": False,
            "model_cls": "test_wan",
            "self_attn_1_type": "radial_attn",
            "task": "t2v",
        }
    )

    assert "target_shape" not in runner.get_supported_request_fields("t2v")
    assert "target_video_length" not in runner.get_supported_request_fields("t2v")


@pytest.mark.parametrize(
    ("runner_cls", "config", "task", "field", "supported"),
    [
        (WanRunner, {"enable_cfg": True}, "t2v", "negative_prompt", True),
        (WanRunner, {"enable_cfg": False}, "t2v", "negative_prompt", False),
        (ErnieImageRunner, {"enable_cfg": True}, "t2i", "negative_prompt", True),
        (ErnieImageRunner, {"enable_cfg": False}, "t2i", "negative_prompt", False),
        (LTX2Runner, {"enable_cfg": True}, "t2av", "negative_prompt", True),
        (LTX2Runner, {"enable_cfg": False}, "t2av", "negative_prompt", False),
        (QwenImageRunner, {"enable_cfg": True}, "t2i", "negative_prompt", True),
        (QwenImageRunner, {"enable_cfg": False}, "t2i", "negative_prompt", False),
        (HunyuanVideo15Runner, {"enable_cfg": False}, "t2v", "negative_prompt", False),
        (HunyuanVideo15Runner, {"enable_cfg": True}, "t2v", "negative_prompt", True),
        (HunyuanVideo15Runner, {"enable_cfg": False, "video_super_resolution": {"enable_cfg": True}}, "t2v", "negative_prompt", True),
        (Flux2Runner, {"enable_cfg": True}, "t2i", "negative_prompt", False),
        (Flux2Runner, {"inpaint_mask_enabled": True}, "i2i", "inpaint_blur_sigma", True),
        (Flux2Runner, {"inpaint_mask_enabled": False}, "i2i", "inpaint_blur_sigma", False),
        (SeedVRRunner, {"seq_parallel": True}, "sr", "image_path", False),
        (SeedVRRunner, {"seq_parallel": False}, "sr", "image_path", True),
        (QwenImageRunner, {"layered": True}, "i2i", "i2i_denoise_strength", False),
        (QwenImageRunner, {"layered": False}, "i2i", "i2i_denoise_strength", True),
        (WanAnimateRunner, {"replace_flag": True}, "animate", "src_bg_path", True),
        (WanAnimateRunner, {"replace_flag": False}, "animate", "src_bg_path", False),
        (WanDancerRunner, {"dancer_stage": "local"}, "s2v", "video_path", True),
        (WanDancerRunner, {"dancer_stage": "global"}, "s2v", "video_path", False),
        (WanAudioARRunner, {}, "rs2v", "target_video_length", False),
        (WanAudioARRunner, {}, "rs2v", "prompt", True),
        (WanAudioARRunner, {"prompt_travel": {"prompt_travel_text": ["scene"]}}, "rs2v", "prompt", False),
    ],
)
def test_request_fields_follow_startup_capabilities(runner_cls, config, task, field, supported):
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, {"task": task, **config})
    assert (field in runner.get_supported_request_fields(task)) is supported


def test_ltx2_upsampler_uses_final_request_shape_for_stage_one():
    runner = object.__new__(LTX2Runner)
    runner.config = {"use_upsampler": True}
    runner.input_info = SimpleNamespace(target_shape=[1024, 1536])

    runner.prepare_stage1_target_shape()

    assert runner.input_info.target_shape == [512, 768]


def test_ltx25_upsampler_validates_and_converts_final_request_shape():
    runner = object.__new__(LTX25Runner)
    runner.config = {"use_upsampler": True}
    runner.input_info = SimpleNamespace(target_shape=[1024, 1536])

    runner.prepare_stage1_target_shape()

    assert runner.input_info.target_shape == [512, 768]


def test_ltx2_v2av_keeps_source_shape_resolution_when_request_omits_shape():
    runner = object.__new__(LTX2Runner)
    runner.config = {
        "task": "v2av",
        "target_height": 768,
        "target_width": 1280,
    }
    runner.supported_tasks = ("v2av",)

    inferred_shape = runner.create_input_info({"task": runner.config["task"], "video_path": "control.mp4"})
    explicit_shape = runner.create_input_info({"task": runner.config["task"], "video_path": "control.mp4", "target_shape": [480, 832]})

    assert inferred_shape.target_shape == []
    assert explicit_shape.target_shape == [480, 832]


def test_worldplay_rejects_pose_and_frame_count_mismatch():
    with pytest.raises(ValueError, match="pose corresponds to 1 frames, num_frames must be set to 1"):
        pose_to_input({"0": {}}, latent_num=2)


def test_wan_scheduler_owns_guidance_scale_validation():
    config = {
        "dim": 16,
        "enable_cfg": False,
        "infer_steps": 4,
        "num_heads": 2,
        "sample_shift": 5.0,
        "seq_parallel": False,
    }

    scheduler = WanScheduler(config)
    assert scheduler.sample_guide_scale is None

    config["enable_cfg"] = True
    with pytest.raises(ValueError, match="sample_guide_scale"):
        WanScheduler(config)


def test_wan_validates_source_inputs_before_model_loading():
    runner = object.__new__(WanRunner)
    runner.config = {"task": "flf2v"}
    runner.supported_tasks = ("flf2v",)

    with pytest.raises(ValueError, match="last_frame_path"):
        runner.prepare_request({"task": runner.config["task"], "image_path": "first.png", "last_frame_path": ""})


@pytest.mark.parametrize("disagg_mode", ["transformer", "decode"])
def test_wan_defers_source_validation_for_downstream_disagg_roles(disagg_mode):
    runner = object.__new__(WanRunner)
    runner.config = {"task": "i2v", "disagg_mode": disagg_mode}
    runner.supported_tasks = ("i2v",)

    runner.prepare_request({"task": runner.config["task"], "image_path": ""})


def test_pipeline_passes_each_request_directly_to_runner():
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "wan2.1"
    pipeline.task = "t2v"
    pipeline.runner = RecordingRunner(task="t2v")

    pipeline.generate(seed=None, num_frames=49, target_shape=[480, 832], return_result_tensor=True)
    pipeline.generate(seed=None, return_result_tensor=True)

    first_request, second_request = pipeline.runner.requests
    assert first_request["target_video_length"] == 49
    assert first_request["target_shape"] == [480, 832]
    assert "target_video_length" not in second_request
    assert "target_shape" not in second_request


def test_pipeline_omits_default_tensor_result_request():
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "hunyuan3d"
    pipeline.task = "i23d"
    pipeline.runner = FileOnlyRecordingRunner(task="i23d")

    pipeline.generate(seed=None)

    assert "return_result_tensor" not in pipeline.runner.requests[0]


def test_multitask_runner_uses_request_task_without_changing_startup_config():
    runner = MultitaskRecordingRunner(task="t2i")

    t2i_input = runner.prepare_request({"task": runner.config["task"], "prompt": "create an image"})
    runner.run_request(t2i_input)
    i2i_input = runner.prepare_request({"task": "i2i", "image_path": "input.png", "prompt": "edit the image"})
    runner.run_request(i2i_input)

    assert isinstance(t2i_input, T2IInputInfo)
    assert isinstance(i2i_input, I2IInputInfo)
    assert t2i_input.task == "t2i"
    assert i2i_input.task == "i2i"
    assert runner.config["task"] == "t2i"
    assert t2i_input is not i2i_input

    next_input = runner.prepare_request({"task": runner.config["task"], "prompt": "create another image"})
    runner.run_request(next_input)
    assert isinstance(next_input, T2IInputInfo)
    assert next_input.task == "t2i"


def test_multitask_runner_validates_fields_for_the_selected_task():
    runner = MultitaskRecordingRunner(task="t2i")

    with pytest.raises(ValueError, match="image_path"):
        runner.prepare_request({"task": "t2i", "image_path": "input.png"})

    with pytest.raises(ValueError, match="not supported by this runner"):
        runner.prepare_request({"task": "vace"})


@pytest.mark.parametrize(
    ("startup_task", "request_task", "request_data", "input_info_cls"),
    [
        ("t2av", "i2av", {"image_path": "first.png"}, I2AVInputInfo),
        ("i2av", "l2av", {"last_frame_path": "last.png"}, L2AVInputInfo),
        ("l2av", "fl2av", {"image_path": "first.png", "last_frame_path": "last.png"}, FL2AVInputInfo),
        ("fl2av", "t2av", {}, T2AVInputInfo),
    ],
)
def test_minimax_h3_base_tasks_share_one_runner(startup_task, request_task, request_data, input_info_cls):
    runner = make_minimax_h3_runner(startup_task)

    input_info = runner.prepare_request({"task": request_task, **request_data})

    assert runner.supported_tasks == ("t2av", "i2av", "l2av", "fl2av")
    assert isinstance(input_info, input_info_cls)
    assert input_info.task == request_task
    assert runner.config["task"] == startup_task


def test_minimax_h3_rejects_unimplemented_image_controls():
    runner = make_minimax_h3_runner("t2av")

    with pytest.raises(ValueError, match="image_strength"):
        runner.prepare_request({"task": "i2av", "image_path": "input.png", "image_strength": 0.5})

    with pytest.raises(ValueError, match="image_frame_idx"):
        runner.prepare_request({"task": "i2av", "image_path": "input.png", "image_frame_idx": [10]})


def test_minimax_h3_reference_transformer_is_a_separate_task_group():
    base_runner = make_minimax_h3_runner("t2av")
    reference_runner = make_minimax_h3_runner("ref2av")

    assert reference_runner.supported_tasks == ("ref2av",)
    with pytest.raises(ValueError, match="Task 'ref2av' is not supported"):
        base_runner.prepare_request({"task": "ref2av", "image_path": "reference.png"})
    with pytest.raises(ValueError, match="Task 't2av' is not supported"):
        reference_runner.prepare_request({"task": "t2av"})


def test_minimax_h3_runtime_uses_input_info_task():
    runner = make_minimax_h3_runner("t2av")
    runner.input_info = I2AVInputInfo(task="i2av")

    with pytest.raises(ValueError, match="i2av requires exactly one"):
        runner._prepare_keyframes()

    runner.input_info = Ref2AVInputInfo(task="ref2av")
    with pytest.raises(ValueError, match="cannot switch between the base and reference"):
        runner._run_input_encoder_local_h3()


def test_minimax_h3_warmup_input_keeps_startup_task():
    runner = make_minimax_h3_runner("i2av")

    runner._prepare_warmup_inputs(480, 480, 124)

    assert runner.input_info.task == "i2av"


def test_minimax_h3_releases_request_conditioning():
    runner = object.__new__(MiniMaxH3Runner)
    runner.config = {}
    runner.model = SimpleNamespace(scheduler=SimpleNamespace(clear=lambda: None))
    runner.input_info = T2AVInputInfo()
    runner.inputs = {}
    runner.condition_video_latents = [object()]
    runner.condition_audio_latents = [object()]
    runner.keyframe_anchors = ("first",)
    runner.prepared_references = [object()]
    runner.maybe_empty_cache = lambda **kwargs: False

    runner.end_run()

    assert runner.condition_video_latents == []
    assert runner.condition_audio_latents == []
    assert runner.keyframe_anchors == ()
    assert runner.prepared_references is None
    assert runner.input_info is None
    assert not hasattr(runner, "inputs")


def test_runner_rejects_task_switching_by_default():
    runner = RecordingRunner(task="t2v")

    with pytest.raises(ValueError, match="Task 'i2v' is not supported by this runner"):
        runner.prepare_request({"task": "i2v", "image_path": "input.png"})


@pytest.mark.parametrize("runner_cls", [RecordingRunner, MultitaskRecordingRunner])
@pytest.mark.parametrize("request_data", [{}, {"task": None}])
def test_runner_defaults_task_only_for_single_task_models(runner_cls, request_data):
    runner = runner_cls(task="t2i")
    original_request = request_data.copy()

    if runner_cls is MultitaskRecordingRunner:
        with pytest.raises(ValueError, match="task is required"):
            runner.prepare_request(request_data)
    else:
        assert runner.prepare_request(request_data).task == "t2i"
    assert request_data == original_request


def test_pipeline_without_default_requires_task_on_every_call():
    pipeline = LightX2VPipeline(model_cls="test_model", support_tasks=["t2i", "i2i"])
    pipeline.runner = MultitaskRecordingRunner(task="t2i")

    with pytest.raises(ValueError, match="task is required"):
        pipeline.generate()
    assert pipeline.generate(task="i2i", image_path="input.png").task == "i2i"
    with pytest.raises(ValueError, match="task is required"):
        pipeline.generate()
    assert pipeline.generate(task="t2i").task == "t2i"
    assert pipeline.runner.config["task"] == "t2i"


def test_neopp_rejects_direct_inputs_encoded_outside_the_runner():
    runner = object.__new__(NeoppRunner)
    runner.config = {"task": "i2i"}
    runner.supported_tasks = ("i2i",)

    assert isinstance(runner.prepare_request({"task": "i2i"}), NeoppInputInfo)
    assert runner.get_supported_request_fields("i2i") == {"task", "seed", "save_result_path", "target_shape"}
    with pytest.raises(ValueError, match="image_path"):
        runner.prepare_request({"task": runner.config["task"], "image_path": "input.png"})


def test_neopp_rejects_non_image_tasks():
    with pytest.raises(ValueError, match="does not support task 't2v'"):
        NeoppRunner({"task": "t2v"})


def test_pipeline_leaves_minimax_h3_output_path_unset():
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "minimax_h3"
    pipeline.task = "t2av"
    pipeline.runner = RecordingRunner(task="t2av")

    result = pipeline.generate(seed=None)

    assert result.save_result_path is None


@pytest.mark.parametrize(
    ("task", "video_path"),
    [
        ("t2v", None),
        ("t2i", None),
        ("sr", "input.mp4"),
        ("sr", None),
    ],
)
def test_pipeline_leaves_output_path_unset_for_all_tasks(task, video_path):
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "test_model"
    pipeline.task = task
    pipeline.runner = RecordingRunner(task=task)

    result = pipeline.generate(seed=None, video_path=video_path)

    assert result.save_result_path is None


@pytest.mark.parametrize("enable_cfg", [False, True])
@pytest.mark.parametrize("request_data", [{}, {"negative_prompt": ""}, {"negative_prompt": "blur"}])
def test_wan_negative_prompt_requires_cfg(enable_cfg, request_data):
    runner = object.__new__(WanRunner)
    BaseRunner.__init__(runner, {"task": "t2v", "enable_cfg": enable_cfg, "vae_stride": (4, 8, 8)})
    original_request = request_data.copy()
    if not enable_cfg and "negative_prompt" in request_data:
        with pytest.raises(ValueError, match="negative_prompt"):
            runner.prepare_request({"task": runner.config["task"], **request_data})
    else:
        assert runner.prepare_request({"task": runner.config["task"], **request_data}).negative_prompt == request_data.get("negative_prompt", "")
    assert request_data == original_request
    assert runner.config["enable_cfg"] is enable_cfg


@pytest.mark.parametrize("request_data", [{}, {"save_result_path": None}])
def test_default_runner_skips_saving_without_output_path(request_data):
    runner = object.__new__(DefaultRunner)
    runner.config = {"task": "t2v"}
    runner.input_info = runner.create_input_info({"task": "t2v", **request_data})
    runner.gen_video_final = object()

    assert runner.process_images_after_vae_decoder() == {"video": None}
    assert runner.gen_video_final is None


def test_pipeline_forwards_request_task_without_changing_its_default(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    pipeline = LightX2VPipeline(
        model_path=str(model_path),
        model_cls="test_model",
        task="t2i",
    )
    monkeypatch.setattr("lightx2v.pipeline.validate_config_paths", lambda config: None)
    monkeypatch.setattr(
        "lightx2v.pipeline.build_runner",
        lambda config: MultitaskRecordingRunner(task=config["task"]),
    )
    pipeline.create_generator(config_json=str(config_path))

    default_result = pipeline.generate(seed=None, return_result_tensor=True)
    result = pipeline.generate(task="i2i", image_path="input.png", seed=None, return_result_tensor=True)
    next_result = pipeline.generate(seed=None, return_result_tensor=True)

    assert default_result.task == "t2i"
    assert result.task == "i2i"
    assert next_result.task == "t2i"
    assert pipeline.task == "t2i"
    assert pipeline.runner.config["task"] == "t2i"
    assert "support_tasks" not in pipeline.startup_config
    assert "support_tasks" not in pipeline.runner.config


def test_runner_applies_request_seed(monkeypatch):
    seeds = []
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seeds.append)

    runner = RecordingRunner()
    input_info = runner.prepare_request({"task": "t2v", "seed": 123})
    runner.run_request(input_info)

    assert seeds == [123]

    input_info = runner.prepare_request({"task": "t2v", "seed": 456})
    runner.run_request(input_info)

    assert seeds == [123, 456]


def test_seedvr_keeps_detected_fps_in_request_context():
    runner = object.__new__(SeedVRRunner)
    runner.config = {"fps": 16}
    first_request = SRInputInfo()
    runner.input_info = first_request

    runner.set_output_fps(24)

    assert first_request.output_fps == 24
    assert runner.get_output_fps() == 24
    assert runner.config["fps"] == 16

    runner.input_info = SRInputInfo()

    assert runner.get_output_fps() == 16


def test_pipeline_forwards_task_specific_inputs():
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "wan2.2_animate"
    pipeline.task = "animate"
    pipeline.runner = RecordingRunner(task="animate")

    pipeline.generate(seed=None, src_pose_path="pose.mp4", src_face_path="face.mp4")

    request = pipeline.runner.requests[0]
    assert request["src_pose_path"] == "pose.mp4"
    assert request["src_face_path"] == "face.mp4"


def test_pipeline_copies_startup_defaults_to_input_info():
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.model_cls = "ltx2_5"
    pipeline.task = "t2av"
    pipeline.runner = RecordingRunner(task="t2av")

    input_info = pipeline.generate(seed=None, return_result_tensor=True)

    assert input_info.target_video_length == 81


def test_pipeline_preserves_explicit_task_when_loading_model_config(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    json_model_path = tmp_path / "json-model"
    json_model_path.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        f'{{"model_cls": "json_model", "model_path": "{json_model_path}", "task": "i2v", "infer_steps": 20, "target_video_length": 49}}',
        encoding="utf-8",
    )
    pipeline = LightX2VPipeline(
        model_path=str(model_path),
        model_cls="constructor_model",
        task="t2v",
    )
    monkeypatch.setattr("lightx2v.pipeline.validate_config_paths", lambda config: None)
    monkeypatch.setattr("lightx2v.pipeline.build_runner", lambda config: SimpleNamespace(config=config))

    pipeline.create_generator(config_json=str(config_path))

    assert pipeline.model_cls == "json_model"
    assert pipeline.model_path == str(json_model_path)
    assert pipeline.task == "t2v"

    with pytest.raises(RuntimeError, match="already been created"):
        pipeline.create_generator()


def test_pipeline_manual_startup_fields(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    pipeline = LightX2VPipeline(
        model_path=str(model_path),
        model_cls="test_model",
        task="t2v",
    )
    monkeypatch.setattr("lightx2v.pipeline.validate_config_paths", lambda config: None)
    monkeypatch.setattr("lightx2v.pipeline.build_runner", lambda config: SimpleNamespace(config=config))

    pipeline.create_generator(
        infer_steps=17,
        num_frames=49,
        height=480,
        width=832,
        resize_mode="adaptive",
    )

    assert pipeline.runner.config["infer_steps"] == 17
    assert pipeline.runner.config["target_video_length"] == 49
    assert pipeline.runner.config["target_height"] == 480
    assert pipeline.runner.config["target_width"] == 832
    assert pipeline.runner.config["resize_mode"] == "adaptive"
