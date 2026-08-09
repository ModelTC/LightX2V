import asyncio
import threading
from pathlib import Path

import pytest
from PIL import Image

from lightx2v.server.api.router import create_api_router
from lightx2v.server.schema import SenseNovaVisionTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.generation.sensenova_vision import (
    SenseNovaVisionGenerationService,
    build_official_vgd_prompt,
    normalize_sensenova_task,
    validate_sensenova_request,
    validate_vgd_output_text,
)
from lightx2v.server.services.inference.worker import TorchrunInferenceWorker


class _FakeRunner:
    def __init__(self):
        self.config = {
            "model_cls": "sensenova_vision",
            "sensenova_source_path": "/does/not/matter/when/visualization/is/disabled",
            "task": "omni_vision_task",
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
        task = task_data["omni_vision_subtask"]
        if task == "understanding":
            text = "A cat is sitting on a chair."
            save_path.write_text(text, encoding="utf-8")
            pipeline_return = {"text": text, "images": []}
        else:
            Image.new("RGB", (8, 8), (10, 20, 30)).save(save_path)
            text = "<p>object1<color>(10,20,30)</color></p>" if task == "vgd_segmentation" else ""
            pipeline_return = {"text": text, "images": []}
        return {
            "status": "success",
            "save_result_path": str(save_path),
            "pipeline_return": pipeline_return,
        }


class _ResidentRunner:
    def __init__(self):
        self.config = {"model_cls": "sensenova_vision", "task": "omni_vision_task"}
        self.calls = []

    def set_config(self, task_data):
        pass

    def run_pipeline(self, input_info):
        self.calls.append((id(input_info), input_info.omni_vision_subtask))
        return {"images": [], "text": input_info.omni_vision_subtask}


def _worker_task(task_id, subtask):
    return {
        "task_id": task_id,
        "omni_vision_subtask": subtask,
        "_return_pipeline_result": True,
    }


def test_sensenova_worker_reuses_loaded_runner_across_subtasks():
    worker = TorchrunInferenceWorker()
    worker.runner = _ResidentRunner()
    runner_identity = id(worker.runner)

    depth = asyncio.run(worker.process_request(_worker_task("depth-task", "depth")))
    understanding = asyncio.run(worker.process_request(_worker_task("understanding-task", "understanding")))

    assert id(worker.runner) == runner_identity
    assert [subtask for _, subtask in worker.runner.calls] == ["depth", "understanding"]
    assert worker.runner.calls[0][0] != worker.runner.calls[1][0]
    assert depth["pipeline_return"]["text"] == "depth"
    assert understanding["pipeline_return"]["text"] == "understanding"


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


def test_sensenova_server_builds_official_vgd_prompt_and_validates_ar_output():
    short_prompt = "<image> Find matching objects for <p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>."
    official_prompt = build_official_vgd_prompt(short_prompt)
    assert official_prompt == (
        "<image> Identify all objects belonging to the same classes as the visually provided "
        "<p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>. Generate an instance segmentation "
        "visualization and each identified category <p>object1</p> is colored different. First, "
        "enumerate each visible <p>object1</p> instance mentioned in the request and assign each "
        "<p>object1</p> a different color. Reformat them in the EXACT format: "
        "<p>object1<color>(R,G,B)</color></p>. Then respond with interleaved instance segmentation "
        "masks using those instance labels and colors."
    )
    validate_vgd_output_text("<p>object1<color>(10,20,30)</color></p>")
    with pytest.raises(RuntimeError, match="official colored-instance format"):
        validate_vgd_output_text("<p>object1</p><bbox>[0.1, 0.2, 0.3, 0.4]</bbox>")


def test_sensenova_server_rejects_non_official_vgd_reference():
    with pytest.raises(ValueError, match="exact form"):
        build_official_vgd_prompt("segment all matching objects")
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        build_official_vgd_prompt("<p>object1</p><bbox>[1, 2, 3, 4]</bbox>")


def test_sensenova_server_passes_official_vgd_prompt_to_resident_runner(tmp_path):
    file_service = FileService(tmp_path / "cache")
    inference_service = _FakeInferenceService()
    service = SenseNovaVisionGenerationService(file_service, inference_service)
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(source)
    result = asyncio.run(
        service.generate_with_stop_event(
            SenseNovaVisionTaskRequest(
                task_id="vgd-task",
                task="vgd_segmentation",
                images=[str(source)],
                prompt="<p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>",
                visualize=False,
                seed=42,
            ),
            threading.Event(),
        )
    )

    call = inference_service.calls[-1]
    assert call["omni_vision_subtask"] == "vgd_segmentation"
    assert call["prompt"] == build_official_vgd_prompt("<p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>")
    assert result.result_data["status"] == "completed"
    assert result.result_data["text"] == "<p>object1<color>(10,20,30)</color></p>"


def test_sensenova_server_loads_default_panoptic_prompt_from_official_source(tmp_path):
    source_root = tmp_path / "SenseNova-Vision"
    example_path = source_root / "inference" / "example_visualize.py"
    example_path.parent.mkdir(parents=True)
    example_path.write_text(
        'EXAMPLE_08_PANOPTIC_QUESTION = "<image> official panoptic prompt"\n',
        encoding="utf-8",
    )

    file_service = FileService(tmp_path / "cache")
    inference_service = _FakeInferenceService()
    inference_service.worker.runner.config["sensenova_source_path"] = str(source_root)
    service = SenseNovaVisionGenerationService(file_service, inference_service)
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(source)
    result = asyncio.run(
        service.generate_with_stop_event(
            SenseNovaVisionTaskRequest(
                task_id="panoptic-task",
                task="panoptic_segmentation",
                images=[str(source)],
                visualize=False,
                seed=42,
            ),
            threading.Event(),
        )
    )

    call = inference_service.calls[-1]
    assert call["omni_vision_subtask"] == "panoptic_segmentation"
    assert call["prompt"] == "<image> official panoptic prompt"
    assert result.result_data["status"] == "completed"


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
    assert [call["omni_vision_subtask"] for call in inference_service.calls] == ["depth", "understanding"]
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
