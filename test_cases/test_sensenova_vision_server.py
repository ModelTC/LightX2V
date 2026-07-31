import asyncio
import threading
from pathlib import Path

from PIL import Image

from lightx2v.server.api.router import create_api_router
from lightx2v.server.schema import SenseNovaVisionTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.generation.sensenova_vision import (
    SenseNovaVisionGenerationService,
    normalize_sensenova_task,
    validate_sensenova_request,
)


class _FakeRunner:
    def __init__(self):
        self.config = {
            "model_cls": "sensenova_vision",
            "sensenova_source_path": "/does/not/matter/when/visualization/is/disabled",
        }


class _FakeWorker:
    def __init__(self):
        self.runner = _FakeRunner()


class _FakeInferenceService:
    def __init__(self):
        self.worker = _FakeWorker()
        self.calls = []

    async def submit_task_async(self, task_data):
        self.calls.append(dict(task_data))
        save_path = Path(task_data["save_result_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        task = task_data["sensenova_task"]
        if task == "raw_query":
            text = "A cat is sitting on a chair."
            save_path.write_text(text, encoding="utf-8")
            pipeline_return = {"text": text, "images": []}
        else:
            Image.new("RGB", (8, 8), (10, 20, 30)).save(save_path)
            pipeline_return = {"text": "", "images": []}
        return {
            "status": "success",
            "save_result_path": str(save_path),
            "pipeline_return": pipeline_return,
        }


def test_sensenova_server_task_aliases_and_validation():
    assert normalize_sensenova_task("binary-seg") == "binary_segmentation"
    task, spec, mode = validate_sensenova_request(
        SenseNovaVisionTaskRequest(
            task="object_detection",
            images=["image.jpg"],
            prompt="bird, boat",
        )
    )
    assert task == "object_detection"
    assert spec.runner_task == "bbox_detection"
    assert mode == "understanding"


def test_sensenova_server_reuses_runner_and_returns_heterogeneous_artifacts(tmp_path):
    file_service = FileService(tmp_path / "cache")
    inference_service = _FakeInferenceService()
    service = SenseNovaVisionGenerationService(file_service, inference_service)
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(source)
    runner_identity = id(inference_service.worker.runner)

    depth_result = asyncio.run(
        service.generate_with_stop_event(
            SenseNovaVisionTaskRequest(
                task_id="depth-task",
                task="depth",
                images=[str(source)],
                visualize=False,
                seed=42,
            ),
            threading.Event(),
        )
    )
    text_result = asyncio.run(
        service.generate_with_stop_event(
            SenseNovaVisionTaskRequest(
                task_id="text-task",
                task="understanding",
                images=[str(source)],
                visualize=False,
                seed=42,
            ),
            threading.Event(),
        )
    )

    assert id(inference_service.worker.runner) == runner_identity
    assert [call["sensenova_task"] for call in inference_service.calls] == ["depth", "raw_query"]
    assert [call["sensenova_mode"] for call in inference_service.calls] == ["dense_perception", "understanding"]
    assert depth_result.result_data["artifacts"][0]["kind"] == "raw_image"
    assert text_result.result_data["text"] == "A cat is sitting on a chair."
    assert text_result.result_data["artifacts"][0]["kind"] == "text"
    assert (file_service.output_video_dir / depth_result.save_result_path).is_file()
    assert (file_service.output_video_dir / text_result.save_result_path).is_file()


def test_sensenova_server_routes_are_registered():
    paths = {route.path for route in create_api_router().routes}
    assert "/v1/tasks/sensenova-vision/" in paths
    assert "/v1/tasks/sensenova-vision/sync" in paths
    assert "/v1/tasks/sensenova-vision/{task_id}/result" in paths
