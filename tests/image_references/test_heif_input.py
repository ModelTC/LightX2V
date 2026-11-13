"""Decode real HEIF references through local and service input paths."""

import asyncio
import base64
import threading
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pillow_heif import from_pillow

from lightx2v.models.input_encoders.hf.minimax_h3_image.preprocess import prepare_reference_images
from lightx2v.server.media.image import is_base64_image, save_base64_image
from lightx2v.server.schema import ImageTaskRequest
from lightx2v.server.services.file_service import FileService
from lightx2v.server.services.generation.image import ImageGenerationService
from lightx2v.utils.image_input import is_heif_header
from lightx2v.utils.image_input import load_rgb_reference as load_rgb_image


@pytest.fixture
def heif_bytes():
    image = Image.new("RGB", (96, 64), (210, 35, 55))
    buf = BytesIO()
    from_pillow(image).save(buf, quality=95)
    return buf.getvalue()


@pytest.mark.parametrize("suffix", [".heic", ".heif", ".HEIC", ".bin"])
def test_decodes_heif_by_content(tmp_path, heif_bytes, suffix):
    path = tmp_path / ("reference" + suffix)
    path.write_bytes(heif_bytes)
    image = load_rgb_image(path)
    assert image.mode == "RGB" and image.size == (96, 64)
    assert np.abs(np.asarray(image).astype(float).mean(axis=(0, 1)) - [210, 35, 55]).max() < 6


@pytest.mark.parametrize("prefix", ["", "data:image/heic;base64,", "data:image/heif;base64,"])
def test_heif_base64_roundtrip(tmp_path, heif_bytes, prefix):
    value = prefix + base64.b64encode(heif_bytes).decode()
    assert is_base64_image(value)
    path = save_base64_image(value, str(tmp_path))
    assert load_rgb_image(path).size == (96, 64)


def test_header_does_not_match_other_containers():
    assert not is_heif_header(b"\0\0\0\x18ftypavif\0\0\0\0avif")
    assert not is_heif_header(b"\0\0\0\x08ftypheic")
    assert not is_heif_header(b"not an image")


def test_corrupt_heif_reports_decode_error(tmp_path):
    path = tmp_path / "broken.heic"
    path.write_bytes(b"invalid image")
    with pytest.raises(ValueError, match="Cannot decode reference image"):
        load_rgb_image(path)


def test_existing_exif_orientation_stays_correct(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40), "red").save(path, exif=exif)
    assert load_rgb_image(path).size == (40, 80)


def test_service_resolves_mixed_reference_inputs_in_order(tmp_path, heif_bytes):
    files = FileService(tmp_path)
    local = tmp_path / "local.heif"
    local.write_bytes(heif_bytes)
    uploaded = files.save_uploaded_file(heif_bytes, "upload.heic")
    url = "https://example.com/reference.heic"
    downloaded = tmp_path / "download.heic"
    downloaded.write_bytes(heif_bytes)

    async def download_image(value):
        assert value == url
        return downloaded

    files.download_image = download_image
    captured = {}

    async def submit(data):
        captured.update(data)
        return {"status": "success", "save_result_path": None}

    service = ImageGenerationService(files, SimpleNamespace(submit_task_async=submit))
    bare = base64.b64encode(heif_bytes).decode()
    message = ImageTaskRequest(task="i2i", prompt="edit", image_path=[str(local), str(uploaded), url, bare, "data:image/heif;base64," + bare])
    asyncio.run(service.generate_with_stop_event(message, threading.Event()))
    resolved = captured["image_path"].split(",")
    assert resolved[:3] == [str(local), str(uploaded), str(downloaded)]
    assert len(resolved) == 5
    assert all(load_rgb_image(path).size == (96, 64) for path in resolved)


def test_heif_orientation_is_applied_once(tmp_path):
    image = Image.new("RGB", (96, 64), "red")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.heic"
    from_pillow(image).save(path, quality=95, exif=exif.tobytes())
    assert load_rgb_image(path).size == (64, 96)


def test_heif_reference_reaches_edit_preprocessor(tmp_path, heif_bytes):
    path = tmp_path / "reference.heif"
    path.write_bytes(heif_bytes)
    references = prepare_reference_images(str(path))
    assert len(references) == 1
    assert references[0].qwen_image.size == (1536, 1024)
    assert references[0].vae_image.size == (1536, 1024)


@pytest.mark.parametrize("image_format", ["PNG", "HEIF"])
@pytest.mark.parametrize("position", [0, 7])
def test_memory_reference_preserves_caller_stream(heif_bytes, image_format, position):
    source = BytesIO()
    if image_format == "HEIF":
        source.write(heif_bytes)
    else:
        Image.new("RGB", (96, 64), "red").save(source, format="PNG")
    source.seek(position)
    image = load_rgb_image(source)
    assert image.mode == "RGB" and image.size == (96, 64)
    assert not source.closed and source.tell() == position
    # Reusing the same stream must not depend on the previous decoder position.
    assert load_rgb_image(source).size == (96, 64)
    assert not source.closed and source.tell() == position


@pytest.mark.parametrize("image_format,suffix", [("PNG", ".png"), ("HEIF", ".heic")])
@pytest.mark.parametrize("position", [0, 7])
def test_open_file_reference_stays_open(tmp_path, heif_bytes, image_format, suffix, position):
    path = tmp_path / ("reference" + suffix)
    if image_format == "HEIF":
        path.write_bytes(heif_bytes)
    else:
        Image.new("RGB", (96, 64), "red").save(path)
    with path.open("rb") as source:
        source.seek(position)
        assert load_rgb_image(source).size == (96, 64)
        assert not source.closed and source.tell() == position


def test_invalid_memory_reference_restores_stream():
    source = BytesIO(b"not a valid image")
    source.seek(5)
    with pytest.raises(ValueError, match="Cannot decode reference image"):
        load_rgb_image(source)
    assert not source.closed and source.tell() == 5


def test_bytes_path_reference_stays_supported(tmp_path):
    import os

    path = tmp_path / "reference.png"
    Image.new("RGB", (96, 64), "red").save(path)
    assert load_rgb_image(os.fsencode(path)).size == (96, 64)


@pytest.mark.parametrize("image_format", ["PNG", "HEIF"])
def test_nonseekable_reference_stays_supported(heif_bytes, image_format):
    from io import UnsupportedOperation

    buffer = BytesIO()
    if image_format == "HEIF":
        buffer.write(heif_bytes)
    else:
        Image.new("RGB", (96, 64), "red").save(buffer, format="PNG")
    buffer.seek(0)

    class NonSeekable:
        def read(self, size=-1):
            return buffer.read(size)

        def seek(self, *args):
            raise UnsupportedOperation("not seekable")

        def tell(self):
            raise UnsupportedOperation("not seekable")

    assert load_rgb_image(NonSeekable()).size == (96, 64)
    assert not buffer.closed


def test_memory_reference_reaches_edit_preprocessor():
    source = BytesIO()
    Image.new("RGB", (96, 64), "red").save(source, format="PNG")
    source.seek(9)
    references = prepare_reference_images(source)
    assert references[0].qwen_image.size == (1536, 1024)
    assert references[0].vae_image.size == (1536, 1024)
    assert not source.closed and source.tell() == 9
