import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.server.api import openai_images
from lightx2v.server.api.tasks import image as image_api
from lightx2v.server.api.tasks import video as video_api
from lightx2v.server.schema import ImageTaskRequest, SenseNovaVisionTaskRequest, TalkObject, VideoTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.generation.image import ImageGenerationService
from lightx2v.server.services.generation.sensenova_vision import SenseNovaVisionGenerationService
from lightx2v.server.services.generation.video import VideoGenerationService
from lightx2v.server.services.inference import worker as worker_module
from lightx2v.server.services.inference.worker import TorchrunInferenceWorker
from lightx2v.server.task_manager import TaskManager
from lightx2v.utils.input_info import SenseNovaVisionInputInfo
from lightx2v.utils.lockable_dict import LockableDict


class RecordingRunner(BaseRunner):
    supported_request_fields_by_task = {task: COMMON_REQUEST_FIELDS | {"prompt", "negative_prompt", "target_shape"} for task in ("t2i", "t2v")}

    def __init__(self, config):
        super().__init__(config)
        self.requests = []

    def run_pipeline(self, input_info):
        self.requests.append(input_info)


class SenseNovaRecordingRunner(RecordingRunner):
    input_info_cls_by_task = {"omni_vision_task": SenseNovaVisionInputInfo}
    supported_request_fields_by_task = {"omni_vision_task": COMMON_REQUEST_FIELDS | {"prompt", "image_path", "omni_vision_subtask", "raw_output_path", "glb_output_path", "postprocess_predictions"}}

    def run_pipeline(self, input_info):
        super().run_pipeline(input_info)
        return {"text": "ready"}


def make_service(tmp_path, monkeypatch, task, config):
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", lambda seed: None)
    runner_cls = SenseNovaRecordingRunner if task == "omni_vision_task" else RecordingRunner
    runner = runner_cls(LockableDict({"task": task, **config}))
    runner.config.lock()
    worker = TorchrunInferenceWorker()
    worker.world_size = 1
    worker.rank = 0
    worker.runner = runner
    inference = SimpleNamespace(worker=worker, submit_task_async=worker.process_request)
    cls = {"t2i": ImageGenerationService, "t2v": VideoGenerationService, "omni_vision_task": SenseNovaVisionGenerationService}[task]
    if task == "omni_vision_task":

        async def resolve_images(self, sources):
            return sources

        monkeypatch.setattr(cls, "_resolve_images", resolve_images)
    return cls(FileService(tmp_path), inference), runner


@pytest.mark.parametrize(("task", "request_cls", "suffix"), [("t2i", ImageTaskRequest, ".png"), ("t2v", VideoTaskRequest, ".mp4")])
def test_server_restores_output_specs_between_requests(tmp_path, monkeypatch, task, request_cls, suffix):
    config = {"target_shape": [512, 768], "enable_cfg": True}
    service, runner = make_service(tmp_path, monkeypatch, task, config)
    first = request_cls(prompt="first", seed=0, negative_prompt="explicit", save_result_path="override" + suffix, target_shape=[480, 832])
    second = request_cls(prompt="second")
    assert "seed" not in second.model_fields_set
    assert "save_result_path" not in second.model_fields_set

    first_response = asyncio.run(service.generate_with_stop_event(first, asyncio.Event()))
    second_response = asyncio.run(service.generate_with_stop_event(second, asyncio.Event()))

    assert [request.seed for request in runner.requests] == [0, 42]
    assert [request.negative_prompt for request in runner.requests] == ["explicit", ""]
    assert [request.target_shape for request in runner.requests] == [[480, 832], [512, 768]]
    assert Path(first_response.save_result_path) == Path(runner.requests[0].save_result_path).absolute()
    assert second_response.save_result_path is None
    assert runner.requests[1].save_result_path is None
    assert runner.config == {"task": task, **config}


@pytest.mark.parametrize(("task", "request_cls"), [("t2i", ImageTaskRequest), ("t2v", VideoTaskRequest)])
@pytest.mark.parametrize("output_fields", [{}, {"save_result_path": None}])
def test_server_does_not_invent_output_paths(tmp_path, monkeypatch, task, request_cls, output_fields):
    service, runner = make_service(tmp_path, monkeypatch, task, {})
    message = request_cls(prompt="image", **output_fields)

    response = asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    assert runner.requests[0].seed == 42
    assert runner.requests[0].save_result_path is None
    assert response.save_result_path is None
    assert response.task_status == "completed"


def test_sync_image_returns_memory_result_without_output_path(tmp_path, monkeypatch):
    service, runner = make_service(tmp_path, monkeypatch, "t2i", {})
    original_run = runner.run_pipeline

    def run(input_info):
        original_run(input_info)
        return {"images": [Image.new("RGB", (4, 4), "red")]}

    monkeypatch.setattr(runner, "run_pipeline", run)
    message = ImageTaskRequest(prompt="image", prefer_memory_result=True)

    response = asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    assert runner.requests[0].save_result_path is None
    assert runner.requests[0].return_result_tensor is True
    assert response.save_result_path is None
    assert response.result_png.startswith(b"\x89PNG")
    assert not list(service.file_service.output_video_dir.iterdir())


def test_single_task_post_uses_startup_task_and_rejects_conflicting_task(tmp_path, monkeypatch):
    service, runner = make_service(tmp_path, monkeypatch, "t2i", {})

    asyncio.run(service.generate_with_stop_event(ImageTaskRequest(prompt="image"), asyncio.Event()))
    assert runner.requests[-1].task == "t2i"
    with pytest.raises(RuntimeError, match="Task 't2v' is not supported"):
        asyncio.run(service.generate_with_stop_event(ImageTaskRequest(task="t2v", prompt="image"), asyncio.Event()))
    assert len(runner.requests) == 1


def test_multitask_post_requires_and_preserves_explicit_task(tmp_path, monkeypatch):
    service, runner = make_service(tmp_path, monkeypatch, "t2i", {})
    runner.supported_tasks = ("t2i", "t2v")
    worker = service.inference_service.worker
    worker.lora_dir = tmp_path
    lora_changes = []
    monkeypatch.setattr(worker, "switch_lora", lambda name, strength: lora_changes.append((name, strength)))

    with pytest.raises(RuntimeError, match="task is required"):
        asyncio.run(service.generate_with_stop_event(ImageTaskRequest(prompt="image"), asyncio.Event()))
    assert lora_changes == []
    for task in ("t2v", "t2i"):
        asyncio.run(service.generate_with_stop_event(ImageTaskRequest(task=task, prompt="image"), asyncio.Event()))
        assert runner.requests[-1].task == task
    assert len(lora_changes) == 2
    assert runner.config["task"] == "t2i"


@pytest.mark.parametrize("task", ["t2i", "t2v", "omni_vision_task"])
@pytest.mark.parametrize(
    ("config", "request_fields", "expected"),
    [
        ({}, {}, 42),
        ({}, {"seed": None}, 42),
        ({"seed": None}, {}, 42),
        ({"seed": None}, {"seed": None}, 42),
        ({"seed": 73}, {}, 42),
        ({"seed": 73}, {"seed": None}, 42),
        ({"seed": 73}, {"seed": 0}, 0),
    ],
)
def test_server_seed_defaults_are_resolved_by_runner(tmp_path, monkeypatch, task, config, request_fields, expected):
    config = {"model_cls": "sensenova_vision", **config} if task == "omni_vision_task" else config
    service, runner = make_service(tmp_path, monkeypatch, task, config)
    if task == "omni_vision_task":
        message = SenseNovaVisionTaskRequest(task="understanding", images=["reference.png"], visualize=False, **request_fields)
    else:
        request_cls = ImageTaskRequest if task == "t2i" else VideoTaskRequest
        message = request_cls(prompt="image", **request_fields)

    asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    assert runner.requests[0].seed == expected
    assert runner.config == {"task": task, **config}


@pytest.mark.parametrize("task", ["t2i", "t2v", "omni_vision_task"])
def test_server_forwards_explicit_seed_to_runner(tmp_path, monkeypatch, task):
    config = {"seed": 73, "model_cls": "sensenova_vision"} if task == "omni_vision_task" else {"seed": 73}
    service, runner = make_service(tmp_path, monkeypatch, task, config)
    if task == "omni_vision_task":
        message = SenseNovaVisionTaskRequest(task="understanding", images=["reference.png"], visualize=False, seed=12)
    else:
        request_cls = ImageTaskRequest if task == "t2i" else VideoTaskRequest
        message = request_cls(prompt="image", seed=12)
    seeded = []
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seeded.append)
    submitted = []
    submit = service.inference_service.submit_task_async

    async def submit_task(task_data):
        submitted.append(task_data["seed"])
        return await submit(task_data)

    service.inference_service.submit_task_async = submit_task

    asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    assert submitted == [12]
    assert runner.requests[0].seed == 12
    assert seeded == [12]
    assert runner.config["seed"] == 73


@pytest.mark.parametrize(("config", "expected"), [({}, False), ({"use_compile": True}, False), ({"use_compile": True, "warmup": True}, True)])
def test_server_warmup_is_owned_by_startup_config(tmp_path, monkeypatch, config, expected):
    config_path = tmp_path / "deployment.json"
    config_path.write_text(json.dumps(config))
    captured = []
    monkeypatch.setattr(worker_module, "build_runner", lambda config: captured.append(config) or object())
    worker = TorchrunInferenceWorker()
    worker.world_size = 1
    args = SimpleNamespace(model_path=str(tmp_path), config_json=str(config_path), model_cls="test_model", task="t2v")

    assert worker.init(args)
    assert captured[0]["warmup"] is expected
    assert "warmup" not in ImageTaskRequest.model_fields
    assert "warmup" not in VideoTaskRequest.model_fields


@pytest.mark.parametrize("seed", [None, 0])
@pytest.mark.parametrize("task", ["image", "video"])
def test_form_preserves_seed_omission_and_zero(tmp_path, monkeypatch, task, seed):
    api = image_api if task == "image" else video_api
    messages = []
    monkeypatch.setattr(api, "get_services", lambda: SimpleNamespace(file_service=FileService(tmp_path)))
    monkeypatch.setattr(api.task_manager, "create_task", lambda message: messages.append(message) or message.task_id)
    fields = dict(request=SimpleNamespace(form=AsyncMock(return_value={"prompt": "prompt"})), image_file=None, prompt="prompt", save_result_path="", negative_prompt="", seed=seed)
    if task == "image":
        fields["aspect_ratio"] = None
        asyncio.run(api.create_image_task_form(**fields))
    else:
        fields.update(last_frame_file=None, target_video_length=None, audio_file=None, video_duration=None)
        asyncio.run(api.create_video_task_form(**fields))

    assert ("seed" in messages[0].model_fields_set) == (seed is not None)
    assert messages[0].seed == seed


@pytest.mark.parametrize("task", ["image", "video"])
def test_form_uploads_files_and_reports_full_queue(tmp_path, monkeypatch, task):
    api = image_api if task == "image" else video_api
    file_service = FileService(tmp_path)
    manager = TaskManager(max_queue_size=1)
    monkeypatch.setattr(api, "get_services", lambda: SimpleNamespace(file_service=file_service))
    monkeypatch.setattr(api, "task_manager", manager)
    fields = dict(
        request=SimpleNamespace(form=AsyncMock(return_value={"prompt": "prompt"})),
        image_file=UploadFile(file=BytesIO(b"first"), filename="first.jpg"),
        prompt="prompt",
        save_result_path="",
        negative_prompt="",
        seed=None,
    )
    expected = {"image_path": (file_service.input_image_dir, ".jpg", b"first")}
    if task == "image":
        fields["aspect_ratio"] = None
        create_form = api.create_image_task_form
    else:
        fields.update(
            last_frame_file=UploadFile(file=BytesIO(b"last"), filename="last.png"),
            audio_file=UploadFile(file=BytesIO(b""), filename="audio.wav"),
            target_video_length=None,
            video_duration=None,
        )
        expected.update(last_frame_path=(file_service.input_image_dir, ".png", b"last"), audio_path=(file_service.input_audio_dir, ".wav", b""))
        create_form = api.create_video_task_form

    response = asyncio.run(create_form(**fields))
    message = manager.get_task(response.task_id).message
    assert response.task_status == "pending"
    for field, (directory, suffix, content) in expected.items():
        path = Path(getattr(message, field))
        assert path.parent == directory
        assert path.suffix == suffix
        assert path.read_bytes() == content

    for field in ("image_file", "last_frame_file", "audio_file"):
        if field in fields:
            fields[field] = None
    with pytest.raises(HTTPException) as exc:
        asyncio.run(create_form(**fields))
    assert exc.value.status_code == 503
    assert "queue is full" in exc.value.detail


@pytest.mark.parametrize(("filename", "content", "suffix"), [("image.jpeg", b"image", ".jpeg"), ("image", b"image", ".png"), ("", b"image", None), ("image.png", b"", None)])
def test_openai_edit_upload_contract(tmp_path, monkeypatch, filename, content, suffix):
    file_service = FileService(tmp_path)
    monkeypatch.setattr(openai_images, "get_services", lambda: SimpleNamespace(file_service=file_service))
    generate = AsyncMock(return_value=(b"result", None))
    monkeypatch.setattr(openai_images, "_run_sync_image_task", generate)
    request = openai_images.create_openai_image_edit(
        request=SimpleNamespace(form=AsyncMock(return_value={})),
        image=[UploadFile(file=BytesIO(content), filename=filename)],
        prompt="edit",
        mask=None,
        model=None,
        n=1,
        size=None,
        response_format="b64_json",
        user=None,
        negative_prompt="",
        seed=None,
        i2i_denoise_strength=None,
    )
    if suffix is None:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(request)
        assert exc.value.status_code == 400
        assert not list(file_service.input_image_dir.iterdir())
        generate.assert_not_awaited()
    else:
        response = asyncio.run(request)
        path = Path(generate.await_args.args[1].image_path)
        assert path.parent == file_service.input_image_dir
        assert path.suffix == suffix
        assert path.read_bytes() == content
        assert response.data[0]["b64_json"] == base64.b64encode(b"result").decode("utf-8")


@pytest.mark.parametrize("task", ["image", "video"])
@pytest.mark.parametrize(
    "text_fields", [{}, {"prompt": "", "negative_prompt": "", "save_result_path": ""}, {"prompt": "cat", "negative_prompt": "blur", "save_result_path": "chosen.png"}, {"task": "i2v", "prompt": "cat"}]
)
def test_http_form_preserves_submitted_text(tmp_path, monkeypatch, task, text_fields):
    api = image_api if task == "image" else video_api
    messages = []
    monkeypatch.setattr(api, "get_services", lambda: SimpleNamespace(file_service=FileService(tmp_path)))
    monkeypatch.setattr(api.task_manager, "create_task", lambda message: messages.append(message) or message.task_id)
    app = FastAPI()
    app.include_router(api.router)

    with TestClient(app) as client:
        response = client.post("/form", data=text_fields, files={"image_file": ("input.png", b"image", "image/png")})

    assert response.status_code == 200
    message = messages[0]
    for key in ("task", "prompt", "negative_prompt", "save_result_path"):
        assert (key in message.model_fields_set) == (key in text_fields)
        assert getattr(message, key) == text_fields.get(key, None if key in {"task", "save_result_path"} else "")


@pytest.mark.parametrize("image_field", ["image", "image[]"])
@pytest.mark.parametrize("negative_prompt", [None, "", "blur"])
def test_http_openai_edit_preserves_negative_prompt(tmp_path, monkeypatch, image_field, negative_prompt):
    monkeypatch.setattr(openai_images, "get_services", lambda: SimpleNamespace(file_service=FileService(tmp_path)))
    generate = AsyncMock(return_value=(b"result", None))
    monkeypatch.setattr(openai_images, "_run_sync_image_task", generate)
    app = FastAPI()
    app.include_router(openai_images.router)
    data = {"prompt": "edit", "response_format": "b64_json"}
    if negative_prompt is not None:
        data["negative_prompt"] = negative_prompt

    with TestClient(app) as client:
        response = client.post("/edits", data=data, files={image_field: ("input.png", b"image", "image/png")})

    assert response.status_code == 200
    message = generate.await_args.args[1]
    assert message.task == "i2i"
    assert ("negative_prompt" in message.model_fields_set) == (negative_prompt is not None)
    assert message.negative_prompt == ("" if negative_prompt is None else negative_prompt)


def test_openai_generation_selects_task_without_a_public_task_parameter(monkeypatch):
    generate = AsyncMock(return_value=(b"result", None))
    monkeypatch.setattr(openai_images, "_run_sync_image_task", generate)
    app = FastAPI()
    app.include_router(openai_images.router)

    with TestClient(app) as client:
        response = client.post("/generations", json={"prompt": "cat", "response_format": "b64_json"})

    assert response.status_code == 200
    assert generate.await_args.args[1].task == "t2i"


@pytest.mark.parametrize("source_type", ["local", "base64", "url"])
def test_video_resolves_frames_and_talk_masks(tmp_path, source_type):
    file_service = FileService(tmp_path)
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4)).save(image_path)
    source = {"local": str(image_path), "base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"), "url": "https://example.test/image.png"}[source_type]
    file_service.download_image = AsyncMock(return_value=image_path)
    submit = AsyncMock(return_value={"status": "success", "save_result_path": "output.mp4"})
    inference = SimpleNamespace(worker=SimpleNamespace(runner=SimpleNamespace(config={})), submit_task_async=submit)
    service = VideoGenerationService(file_service, inference)
    message = VideoTaskRequest(image_path=source, last_frame_path=source, talk_objects=[TalkObject(audio="audio.wav", mask=source)])

    asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    payload = submit.await_args.args[0]
    talk_objects = json.loads((Path(payload["audio_path"]) / "config.json").read_text())["talk_objects"]
    for path in (payload["image_path"], payload["last_frame_path"], talk_objects[0]["mask"]):
        assert Path(path).read_bytes() == image_path.read_bytes()
    assert talk_objects[0]["audio"] == "audio.wav"
    assert "talk_objects" not in payload


def test_image_resolves_and_aligns_mask(tmp_path):
    file_service = FileService(tmp_path)
    image_path = tmp_path / "source.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (4, 4)).save(image_path)
    Image.new("RGB", (2, 2)).save(mask_path)
    submit = AsyncMock(return_value={"status": "success", "save_result_path": "output.png"})
    inference = SimpleNamespace(worker=SimpleNamespace(runner=SimpleNamespace(config={})), submit_task_async=submit)
    service = ImageGenerationService(file_service, inference)
    message = ImageTaskRequest(image_path=str(image_path), image_mask_path=base64.b64encode(mask_path.read_bytes()).decode("utf-8"))

    asyncio.run(service.generate_with_stop_event(message, asyncio.Event()))

    payload = submit.await_args.args[0]
    files = list(Path(payload["image_path"]).glob("*.png"))
    assert len(files) == 2
    assert "image_mask_path" not in payload
    for path in files:
        with Image.open(path) as image:
            assert image.size == (4, 4)
