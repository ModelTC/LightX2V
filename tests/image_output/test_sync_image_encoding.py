"""Exercise synchronous image encoding and response formats without GPUs."""

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from lightx2v.server.api import openai_images
from lightx2v.server.api.tasks import image as image_api
from lightx2v.server.schema import ImageTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.inference.pipeline_image_encode import encode_pipeline_return_to_image_bytes
from lightx2v.server.task_manager import TaskManager
from lightx2v.utils.image_output import image_format_for_bytes


@pytest.fixture
def image():
    return Image.fromarray(np.random.default_rng(5).integers(0, 256, (64, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize("suffix,fmt", [(None, "png"), ("", "png"), (".png", "png"), (".JPG", "jpeg"), (".jpeg", "jpeg"), (".WEBP", "webp")])
@pytest.mark.parametrize("quality", [0, 40, 100])
def test_encodes_selected_format_and_quality(image, suffix, fmt, quality):
    path = "result" + suffix if suffix is not None else None
    encoded = encode_pipeline_return_to_image_bytes({"images": [image]}, path, quality)
    assert image_format_for_bytes(encoded) == fmt
    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.size == image.size
        if fmt == "png":
            assert np.array_equal(decoded, image)
    if fmt != "png":
        expected = BytesIO()
        image.save(expected, format=fmt.upper(), quality=quality)
        assert encoded == expected.getvalue()


@pytest.mark.parametrize("schema", [ImageTaskRequest, openai_images.OpenAIImageGenerationRequest, openai_images.OpenAIImageEditRequest])
@pytest.mark.parametrize("value", [-1, 101, True, 1.5, "1.5"])
def test_rejects_invalid_quality(schema, value):
    with pytest.raises(ValidationError, match="output_compression"):
        schema(prompt="edit", output_compression=value)


def test_form_integer_and_request_forwarding():
    from lightx2v.server.api.request_parsing import parse_form_request

    body = parse_form_request(openai_images.OpenAIImageEditRequest, {"prompt": "edit", "save_result_path": "out.webp", "output_compression": "0"})
    message = openai_images._build_image_task_request(body, task="i2i")
    assert message.save_result_path == "out.webp" and message.output_compression == 0
    assert "output_compression" not in openai_images._build_image_task_request(openai_images.OpenAIImageGenerationRequest(prompt="x"), task="t2i").model_fields_set


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_response_download_and_upload_match_encoded_bytes(tmp_path, monkeypatch, image, fmt):
    raw = encode_pipeline_return_to_image_bytes({"images": [image]}, "out." + fmt, 30)
    files = FileService(tmp_path)
    services = SimpleNamespace(file_service=files)
    monkeypatch.setattr(openai_images, "get_services", lambda: services)
    monkeypatch.setattr(image_api, "get_services", lambda: services)
    request = Request({"type": "http", "scheme": "http", "server": ("testserver", 80), "path": "/", "root_path": "", "headers": []})
    b64 = openai_images._build_openai_response(request, "test", raw, "b64_json")
    assert b64.output_format == fmt and base64.b64decode(b64.data[0]["b64_json"]) == raw
    response = openai_images._build_openai_response(request, "test", raw, "url")
    assert response.output_format == fmt and response.data[0]["url"].endswith("test." + fmt)
    assert (files.output_video_dir / ("test." + fmt)).read_bytes() == raw
    direct = image_api._build_image_response(raw)
    assert direct.body == raw and direct.media_type == "image/" + fmt
    assert direct.headers["content-disposition"].endswith(f'result.{fmt}"')
    uploads = {}

    async def upload(**kwargs):
        uploads.update(kwargs)

    files.upload_to_presigned_url = upload
    message = ImageTaskRequest(presigned_url="https://example.com/upload")
    asyncio.run(image_api._upload_sync_result_if_needed(message, raw))
    assert uploads["file_content"] == raw and uploads["content_type"] == "image/" + fmt
    manager = TaskManager(result_image_keep_count=1)
    task_id = manager.create_task(message)
    manager.complete_task(task_id, result_image=raw)
    assert manager.get_task_result_image(task_id) == raw


def test_generation_endpoint_encodes_requested_format(monkeypatch, image):
    async def run(request, message):
        return encode_pipeline_return_to_image_bytes({"images": [image]}, message.save_result_path, message.output_compression), None

    monkeypatch.setattr(openai_images, "_run_sync_image_task", run)
    app = FastAPI()
    app.include_router(openai_images.router, prefix="/v1/images")
    client = TestClient(app)
    result = client.post("/v1/images/generations", json={"prompt": "x", "save_result_path": "out.webp", "output_compression": 20})
    assert result.status_code == 200
    body = result.json()
    assert body["output_format"] == "webp"
    assert image_format_for_bytes(base64.b64decode(body["data"][0]["b64_json"])) == "webp"
    assert client.post("/v1/images/generations", json={"prompt": "x", "output_compression": -1}).status_code == 422
    assert client.post("/v1/images/generations", json={"prompt": "x", "save_result_path": "out.gif"}).status_code == 400


@pytest.mark.parametrize("suffix,fmt", [(".jpg", "jpeg"), (".webp", "webp"), (".png", "png")])
def test_worker_uses_effective_request_quality(image, suffix, fmt):
    from lightx2v.server.services.inference.worker import TorchrunInferenceWorker

    class Runner:
        def prepare_request(self, data):
            return SimpleNamespace(output_compression=0)

        def set_reuse(self, *args):
            pass

        def run_request(self, input_info):
            return {"images": [image]}

        def compute_usage(self, **kwargs):
            return None

    worker = TorchrunInferenceWorker.__new__(TorchrunInferenceWorker)
    worker.world_size = 1
    worker.rank = 0
    worker.lora_dir = None
    worker.runner = Runner()
    result = asyncio.run(worker.process_request({"task_id": "test", "return_result_tensor": True, "save_result_path": "out" + suffix}))
    assert result["status"] == "success"
    expected = encode_pipeline_return_to_image_bytes({"images": [image]}, "out" + suffix, 0)
    assert result["result_image"] == expected and image_format_for_bytes(result["result_image"]) == fmt


@pytest.mark.parametrize("suffix,fmt", [(".jpg", "JPEG"), (".webp", "WEBP")])
@pytest.mark.parametrize("kind", ["bgr", "bgra", "gray", "gray_channel", "uint16_bgr", "uint16_gray", "float255", "float01", "noncontiguous"])
def test_numpy_direct_encoding_matches_previous_pixels(monkeypatch, suffix, fmt, kind):
    import cv2

    from lightx2v.server.services.inference import pipeline_image_encode as encoder

    array = np.random.default_rng(8).integers(0, 256, (20, 24, 3), dtype=np.uint8)
    if kind == "bgra":
        array = np.concatenate([array, np.full((20, 24, 1), 37, dtype=np.uint8)], axis=-1)
    elif kind in ("gray", "gray_channel"):
        array = array[:, :, 0] if kind == "gray" else array[:, :, :1]
    elif kind == "uint16_bgr":
        array = array.astype(np.uint16) * 257
    elif kind == "uint16_gray":
        array = array[:, :, 0].astype(np.uint16) * 257
    elif kind == "float255":
        array = array.astype(np.float32) * 1.25 - 20.5
    elif kind == "float01":
        array = array.astype(np.float32) / 255
    elif kind == "noncontiguous":
        array = array[:, ::2, :]
    # Compare against the actual previous PNG roundtrip, including OpenCV's
    # BGR convention and its handling of uint16/float arrays.
    png = encoder._image_to_png_bytes(array)
    with Image.open(BytesIO(png)) as decoded:
        reference = decoded.convert("RGB")
    expected = BytesIO()
    reference.save(expected, format=fmt, quality=40)

    def forbid_png(*args, **kwargs):
        raise AssertionError("Tensor/NumPy output must not encode or decode an intermediate PNG")

    monkeypatch.setattr(cv2, "imencode", forbid_png)
    monkeypatch.setattr(Image, "open", forbid_png)
    result = encoder.encode_pipeline_return_to_image_bytes(({"images": [[array]]}, None), "out" + suffix, 40)
    assert result == expected.getvalue()


@pytest.mark.parametrize("suffix,fmt", [(".jpg", "JPEG"), (".webp", "WEBP")])
@pytest.mark.parametrize("kind", ["chw", "hwc", "batch", "float01", "float255", "gray", "rgba", "noncontiguous"])
def test_tensor_direct_encoding_preserves_rgb_pixels(monkeypatch, suffix, fmt, kind):
    import cv2
    import torch

    from lightx2v.server.services.inference import pipeline_image_encode as encoder

    pixels = np.random.default_rng(9).integers(0, 256, (20, 24, 3), dtype=np.uint8)
    if kind == "gray":
        pixels = pixels[:, :, :1]
    elif kind == "rgba":
        pixels = np.concatenate([pixels, np.full((20, 24, 1), 37, dtype=np.uint8)], axis=-1)
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
    if kind == "hwc":
        tensor = tensor.permute(1, 2, 0)
    elif kind == "batch":
        tensor = torch.stack([tensor, torch.zeros_like(tensor)])
    elif kind == "float01":
        tensor = tensor.float() / 255
    elif kind == "float255":
        tensor = tensor.float() * 1.25 - 20.5
        pixels = np.clip(np.rint(pixels.astype(np.float32) * 1.25 - 20.5), 0, 255).astype(np.uint8)
    elif kind == "noncontiguous":
        tensor = tensor[:, :, ::2]
        pixels = pixels[:, ::2, :]
    reference = Image.fromarray(pixels[:, :, 0] if pixels.shape[-1] == 1 else pixels).convert("RGB")
    expected = BytesIO()
    reference.save(expected, format=fmt, quality=40)

    def forbid_png(*args, **kwargs):
        raise AssertionError("Tensor/NumPy output must not encode or decode an intermediate PNG")

    monkeypatch.setattr(cv2, "imencode", forbid_png)
    monkeypatch.setattr(Image, "open", forbid_png)
    assert encoder.encode_pipeline_return_to_image_bytes({"images": tensor}, "out" + suffix, 40) == expected.getvalue()


def test_tensor_png_preserves_rgba_channel_order():
    import torch

    from lightx2v.server.services.inference.pipeline_image_encode import encode_pipeline_return_to_png_bytes

    pixels = np.zeros((12, 16, 4), dtype=np.uint8)
    pixels[:, :] = [200, 70, 20, 37]
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
    encoded = encode_pipeline_return_to_png_bytes({"images": tensor})
    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGBA"
        assert np.array_equal(decoded, pixels)


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_openai_url_download_serves_actual_image_type(tmp_path, monkeypatch, image, fmt):
    from lightx2v.server.api import files

    services = SimpleNamespace(file_service=FileService(tmp_path))
    monkeypatch.setattr(openai_images, "get_services", lambda: services)
    monkeypatch.setattr(files, "get_services", lambda: services)
    app = FastAPI()
    app.include_router(files.router, prefix="/v1/files")
    client = TestClient(app)
    request = Request({"type": "http", "scheme": "http", "server": ("testserver", 80), "path": "/", "root_path": "", "headers": []})
    raw = encode_pipeline_return_to_image_bytes({"images": [image]}, "output." + fmt, 40)
    generated = openai_images._build_openai_response(request, "download-test", raw, "url")
    downloaded = client.get(generated.data[0]["url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/" + fmt
    assert downloaded.content == raw
    assert generated.output_format == fmt
    assert downloaded.headers["content-disposition"].endswith(f'download-test.{fmt}"')
