import asyncio
import json
from dataclasses import fields
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Runner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.flux2.flux2_runner import Flux2Runner
from lightx2v.models.runners.hidream_o1_image.hidream_o1_image_runner import HidreamO1ImageRunner
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v.models.runners.motus.motus_runner import MotusRunner
from lightx2v.models.runners.runner_factory import RUNNER_MODULES
from lightx2v.models.runners.wan.wan_animate2_runner import WanAnimate2Runner
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner
from lightx2v.models.runners.wan.wan_runner import LingbotRunner, WanRunner
from lightx2v.models.runners.wan.wan_s2v_runner import WanS2VRunner
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner
from lightx2v.server.api.tasks import image as image_api
from lightx2v.server.api.tasks import video as video_api
from lightx2v.server.schema import ImageTaskRequest, SenseNovaVisionTaskRequest, TalkObject, VideoTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.generation.image import ImageGenerationService
from lightx2v.server.services.generation.video import VideoGenerationService
from lightx2v.server.services.inference.worker import TorchrunInferenceWorker
from lightx2v.utils.input_info import INPUT_INFO_TYPES
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@pytest.mark.parametrize("model_cls,module", RUNNER_MODULES.items())
def test_native_schema_covers_runner_fields(monkeypatch, model_cls, module):
    monkeypatch.setattr(DefaultRunner, "__del__", lambda self: None)
    import_module(module)
    runner_cls = RUNNER_REGISTER[model_cls]
    image_tasks = {"t2i", "i2i", "ti2i"}
    video_tasks = {"t2v", "i2v", "flf2v", "vace", "s2v", "rs2v", "animate", "sr", "t2av", "i2av", "l2av", "fl2av", "ref2av", "i2va", "v2av", "ltx2_s2v"}
    internal_fields = {"return_result_tensor", "policy_image", "policy_state"}
    variants = (
        {},
        {"inpaint_mask_enabled": True},
        {"image_size": [512, 512]},
        {"video_super_resolution": {"enable_cfg": True}},
        {"layered": True},
        {"seq_parallel": True},
        {"replace_flag": True},
        {"dancer_stage": "local"},
        {"prompt_travel": {"prompt_travel_text": ["scene"]}},
        {"self_attn_1_type": "radial_attn"},
        {"resize_mode": "fixed_shape"},
    )
    for task in runner_cls.supported_request_fields_by_task:
        input_info_cls = runner_cls.input_info_cls_by_task.get(task) or INPUT_INFO_TYPES[task]
        input_fields = {field.name for field in fields(input_info_cls)}
        if task in image_tasks:
            schema = ImageTaskRequest
        elif task in video_tasks:
            schema = VideoTaskRequest
        else:
            schema = None
        for enable_cfg in (False, True):
            for variant in variants:
                runner = object.__new__(runner_cls)
                BaseRunner.__init__(runner, {"task": task, "enable_cfg": enable_cfg, **variant})
                request_fields = runner.get_supported_request_fields(task)
                assert not request_fields - input_fields, (model_cls, task, variant)
                if schema is not None:
                    assert not request_fields - schema.model_fields.keys() - internal_fields, (model_cls, task, variant)


@pytest.mark.parametrize(
    "runner_cls,task,payload",
    [
        (WanS2VRunner, "s2v", {"src_pose_path": "pose.mp4"}),
        (WanAnimateRunner, "animate", {"src_pose_path": "pose.mp4", "src_face_path": "face.mp4", "src_ref_images": ["reference.png"]}),
        (WanAnimate2Runner, "animate", {"prompt_ref": "reference motion", "video_path": "driving.mp4", "src_pose_path": "pose.mp4"}),
        (WanVaceRunner, "vace", {"video_path": "source.mp4", "mask_path": "mask.mp4", "src_ref_images": ["reference.png"]}),
        (
            LTX2Runner,
            "v2av",
            {
                "image_path": "first.png,last.png",
                "image_strength": [0.0, 0.8],
                "image_frame_idx": [0, 16],
                "reference_video_strength": 0.5,
                "reference_video_frame_cap": 17,
                "mux_audio_video_path": "audio.mp4",
                "video_path": "reference.mp4",
            },
        ),
        (LingbotRunner, "i2v", {"image_path": "input.png", "pose": "w-1", "action_path": "controls"}),
        (
            Cosmos3Runner,
            "i2va",
            {"action_mode": "policy", "domain_name": "agibotworld", "view_point": "front", "action_path": "actions.npy", "state_path": "state.npy", "save_action_path": "actions.json"},
        ),
        (MotusRunner, "i2v", {"image_path": "input.png", "state_path": "state.json", "save_action_path": "actions.json"}),
        (HidreamO1ImageRunner, "i2i", {"keep_original_aspect": False, "layout_bboxes": '[{"bbox": [0, 0, 100, 100]}]'}),
        (HunyuanImage3Runner, "i2i", {"infer_align_image_size": False}),
    ],
)
def test_service_preserves_model_request_fields(tmp_path, monkeypatch, runner_cls, task, payload):
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", lambda seed: None)
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, {"task": task, "enable_cfg": True, "vae_stride": (4, 8, 8)})
    runner.run_pipeline = Mock(return_value=None)
    worker = TorchrunInferenceWorker()
    worker.rank = 0
    worker.world_size = 1
    worker.runner = runner
    inference = SimpleNamespace(worker=worker, submit_task_async=worker.process_request)
    is_image = task == "i2i"
    schema = ImageTaskRequest if is_image else VideoTaskRequest
    service_cls = ImageGenerationService if is_image else VideoGenerationService
    service = service_cls(FileService(tmp_path), inference)
    message = schema(task=task, **payload)

    response = asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    assert response.task_status == "completed"
    input_info = runner.run_pipeline.call_args.args[0]
    for field, value in payload.items():
        assert getattr(input_info, field) == (",".join(value) if field == "src_ref_images" else value)


@pytest.mark.parametrize("field", ["src_pose_path", "image_strength"])
def test_known_but_unsupported_fields_reach_runner_validation(tmp_path, monkeypatch, field):
    runner = object.__new__(WanRunner)
    BaseRunner.__init__(runner, {"task": "t2v", "vae_stride": (4, 8, 8)})
    message = VideoTaskRequest(**{field: "pose.mp4" if field == "src_pose_path" else 0.5})
    service = VideoGenerationService(FileService(tmp_path), None)
    request_data = service.prepare_task_data(message)
    request_data.pop("task_id")

    with pytest.raises(ValueError, match=field):
        runner.prepare_request(request_data)


@pytest.fixture
def native_api(tmp_path, monkeypatch):
    def build(api):
        messages = []
        file_service = FileService(tmp_path)
        monkeypatch.setattr(api, "get_services", lambda: SimpleNamespace(file_service=file_service))
        monkeypatch.setattr(api.task_manager, "create_task", lambda message: messages.append(message) or message.task_id)
        app = FastAPI()
        app.include_router(api.router)
        return TestClient(app), messages

    return build


@pytest.mark.parametrize("encoding", ["json", "form"])
@pytest.mark.parametrize(
    "api,payload",
    [
        (video_api, {"src_pose_path": "pose.mp4", "src_face_path": "face.mp4", "src_bg_path": "background.mp4", "mask_path": "mask.mp4", "prompt_ref": "motion"}),
        (video_api, {"image_strength": [0.0, 0.8], "image_frame_idx": [0, 16], "reference_video_strength": 0.0, "reference_video_frame_cap": 17, "mux_audio_video_path": "audio.mp4"}),
        (video_api, {"pose": {"0": {"camera": "forward"}}, "target_shape": [480, 832], "src_ref_images": ["reference.png"], "num_frames": 17, "video_duration": 2.5}),
        (video_api, {"action_mode": "policy", "action_path": "actions.npy", "state_path": "state.json", "save_action_path": "output.npy", "domain_name": "robot", "view_point": "front"}),
        (image_api, {"keep_original_aspect": False, "layout_bboxes": '[{"bbox": [0, 0, 1, 1]}]', "infer_align_image_size": False, "target_shape": [512, 768], "prompt": "[keep this text]"}),
    ],
)
def test_http_preserves_schema_fields(native_api, encoding, api, payload):
    client, messages = native_api(api)
    with client:
        if encoding == "json":
            response = client.post("/", json=payload)
        else:
            form = {key: json.dumps(value) if isinstance(value, (list, dict)) else str(value) for key, value in payload.items()}
            response = client.post("/form", data=form)

    assert response.status_code == 200, response.text
    for field, value in payload.items():
        field = "target_video_length" if field == "num_frames" else field
        assert getattr(messages[0], field) == value
        assert field in messages[0].model_fields_set
    assert messages[0].save_result_path is None


@pytest.mark.parametrize("encoding", ["json", "form"])
@pytest.mark.parametrize("inpaint_enabled", [False, True])
def test_http_flux2_inpaint_fields_follow_deployment(native_api, tmp_path, monkeypatch, encoding, inpaint_enabled):
    client, messages = native_api(image_api)
    payload = {"inpaint_blur_sigma": 0.5, "inpaint_blur_size": 3}
    with client:
        response = client.post("/", json=payload) if encoding == "json" else client.post("/form", data=payload)
    assert response.status_code == 200, response.text

    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", lambda seed: None)
    runner = object.__new__(Flux2Runner)
    BaseRunner.__init__(runner, {"task": "i2i", "inpaint_mask_enabled": inpaint_enabled})
    runner.run_pipeline = Mock(return_value=None)
    worker = TorchrunInferenceWorker()
    worker.rank = 0
    worker.world_size = 1
    worker.runner = runner
    inference = SimpleNamespace(worker=worker, submit_task_async=worker.process_request)
    service = ImageGenerationService(FileService(tmp_path), inference)

    if inpaint_enabled:
        response = asyncio.run(service.generate_with_stop_event(messages[0], asyncio.Event()))
        assert response.task_status == "completed"
        input_info = runner.run_pipeline.call_args.args[0]
        assert input_info.inpaint_blur_sigma == 0.5
        assert input_info.inpaint_blur_size == 3
    else:
        with pytest.raises(RuntimeError, match="inpaint_blur_sigma.*inpaint_blur_size") as error:
            asyncio.run(service.generate_with_stop_event(messages[0], asyncio.Event()))
        assert error.value.original_error_type == "ValueError"
        runner.run_pipeline.assert_not_called()


@pytest.mark.parametrize("api", [image_api, video_api])
@pytest.mark.parametrize("encoding", ["json", "form"])
@pytest.mark.parametrize("field", ["unknown_option", "infer_steps", "warmup", "resize_mode", "return_result_tensor"])
def test_http_rejects_unknown_fields(native_api, api, encoding, field):
    client, messages = native_api(api)
    with client:
        response = client.post("/", json={field: True}) if encoding == "json" else client.post("/form", data={field: "true"})

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert messages == []


@pytest.mark.parametrize("api,payload", [(video_api, {"target_shape": "[480,"}), (image_api, {"keep_original_aspect": "maybe"})])
def test_form_rejects_invalid_values(native_api, api, payload):
    client, messages = native_api(api)
    with client:
        response = client.post("/form", data=payload)
    assert response.status_code == 422, response.text
    assert not messages


def test_form_rejects_unknown_upload(native_api):
    client, messages = native_api(video_api)
    with client:
        response = client.post("/form", files={"unknown_file": ("file.mp4", b"video", "video/mp4")})
    assert response.status_code == 422, response.text
    assert not messages


@pytest.mark.parametrize("schema,payload", [(TalkObject, {"audio": "audio.wav", "mask": "mask.png"}), (SenseNovaVisionTaskRequest, {"task": "depth"})])
def test_nested_and_special_requests_reject_unknown_fields(schema, payload):
    with pytest.raises(ValueError, match="extra_forbidden"):
        schema(**payload, unknown_option=True)
